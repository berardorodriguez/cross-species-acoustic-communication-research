from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


OBS_C1 = 0.8364381324904796
OBS_C2 = 117 / 127
PRIMARY_START = 10000
PRIMARY_N = 500
EXTENSION_START = 10500
EXTENSION_END = 10999
RESERVE_START = 11000


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def now_iso() -> str:
    return pd.Timestamp.now().isoformat()


def validate_counts(manifest: pd.DataFrame) -> bool:
    counts = manifest["subdomain"].astype(str).str.lower().value_counts().to_dict()
    return counts == {"gc": 5, "lfc": 5, "hfc": 5, "psc": 5}


def combine_anchor_with_frozen(
    historical_full_features: Path,
    ensemble_features: Path,
    output_csv: Path,
) -> tuple[pd.DataFrame, list[str]]:
    hist = pd.read_csv(historical_full_features)
    new_anchor = pd.read_csv(ensemble_features)

    # The public feature table omits audio filenames and paths. Keep the
    # generated source manifest private to the run and remove those fields
    # before combining the regenerated anchors with the distributed table.
    path_metadata = {
        "file_name",
        "relative_path",
        "original_name",
        "analysis_name",
        "relative_path_original",
        "relative_path_processed",
        "resolved_audio_path",
    }
    new_anchor = new_anchor.drop(
        columns=[column for column in path_metadata if column in new_anchor.columns]
    )

    anchor_mask = hist["domain"].astype(str).str.lower() == "bovine_core_ensemble"
    hist_anchor = hist.loc[anchor_mask].copy()

    missing_cols = sorted(set(hist.columns) - set(new_anchor.columns))
    extra_cols = sorted(set(new_anchor.columns) - set(hist.columns))
    expected_missing = {"base_version", "condition", "fragment_id"}

    if set(missing_cols) != expected_missing or extra_cols:
        raise RuntimeError(
            f"Unexpected schema mismatch. Missing={missing_cols}; extra={extra_cols}"
        )

    for c in missing_cols:
        new_anchor[c] = np.nan

    new_anchor = new_anchor[hist.columns]

    if len(hist_anchor) != 20 or len(new_anchor) != 20:
        raise RuntimeError("Anchor row count mismatch.")
    if not new_anchor["id"].is_unique:
        raise RuntimeError("Rebuilt anchor IDs are not unique.")
    if set(new_anchor["id"]) != set(hist_anchor["id"]):
        raise RuntimeError("Rebuilt anchor IDs differ from frozen historical anchor IDs.")

    lookup = {str(r["id"]): r.copy() for _, r in new_anchor.iterrows()}
    rows = []

    for _, row in hist.iterrows():
        if str(row["domain"]).lower() == "bovine_core_ensemble":
            ident = str(row["id"])
            rebuilt = lookup[ident].copy()
            rebuilt["id"] = ident
            rows.append(rebuilt)
        else:
            rows.append(row.copy())

    combined = pd.DataFrame(rows, columns=hist.columns)

    expected_shape = (549, len(hist.columns))
    if combined.shape != expected_shape:
        raise RuntimeError(f"Unexpected combined shape: {combined.shape}")
    if not combined["id"].notna().all() or combined["id"].nunique() != 549:
        raise RuntimeError("Combined ID integrity check failed.")

    combined.to_csv(output_csv, index=False)
    return combined, missing_cols


def compute_iteration_statistics(
    pca_variance_path: Path,
    projection_path: Path,
    c2_ids: set[str],
) -> tuple[float, float, int]:
    pv = pd.read_csv(pca_variance_path)
    if len(pv) != 10:
        raise RuntimeError(f"Expected 10 PCA rows, got {len(pv)}")

    c1 = float(pv.loc[:2, "explained_variance_ratio"].astype(float).sum())

    projected = pd.read_csv(projection_path)
    c2_rows = projected[projected["id"].astype(str).isin(c2_ids)].copy()

    if len(c2_rows) != 127:
        raise RuntimeError(f"Expected 127 C2 rows, got {len(c2_rows)}")

    n_gc = int(
        (c2_rows["nearest_bovine_class"].astype(str).str.upper() == "GC").sum()
    )
    c2 = n_gc / 127.0
    return c1, c2, n_gc


def save_iteration_audit(
    audit_dir: Path,
    base_seed: int,
    planned_index: int | None,
    status: str,
    error: str | None,
    manifest: pd.DataFrame | None,
    feature_manifest_path: Path | None,
    pca_variance_path: Path | None,
    centroids_path: Path | None,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "base_seed": base_seed,
        "planned_index": planned_index,
        "status": status,
        "error": error,
        "timestamp": now_iso(),
    }
    (audit_dir / "iteration_status.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if manifest is not None:
        manifest.to_csv(audit_dir / "bovine_ensembles_180s.csv", index=False)

    for src, name in [
        (feature_manifest_path, "feature_manifest.csv"),
        (pca_variance_path, "pca_variance.csv"),
        (centroids_path, "bovine_subdomain_centroids.csv"),
    ]:
        if src is not None and src.exists():
            shutil.copy2(src, audit_dir / name)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description=(
            "Run the prespecified MICRO–MACRO random-assembly Null. "
            "Source recordings must be available locally under appropriate access rights."
        )
    )
    p.add_argument(
        "--audio-corpus-root",
        type=Path,
        required=True,
        help="Local corpus root containing analysis_ready_clean/BOVINE_CORE/.",
    )
    p.add_argument(
        "--source-metadata",
        type=Path,
        default=repo_root / "data" / "metadata" / "Supplementary_Dataset_D1_Corpus_Metadata.csv",
        help="Corpus metadata CSV; defaults to the sanitized repository copy.",
    )
    p.add_argument(
        "--historical-features",
        type=Path,
        default=repo_root / "data" / "derived" / "oaas_core" / "features_MACRO_derived.csv",
        help="Frozen MACRO feature table used as the baseline comparison set.",
    )
    p.add_argument(
        "--c2-ids-csv",
        type=Path,
        default=repo_root / "data" / "derived" / "assembly_null" / "c2_frozen_micro_hfc_obonuco_ids.csv",
        help="Frozen 127 Obonuco fragment IDs selected as HFC-nearest in MICRO.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "assembly_null",
        help="Directory for iteration results, audit summaries, and temporary files.",
    )
    p.add_argument(
        "--stage",
        choices=["primary", "extension"],
        default="primary",
        help="primary = planned seeds 10000-10499; extension = 10500-10999.",
    )
    p.add_argument(
        "--max_new_valid",
        type=int,
        default=None,
        help="Optional cap for this invocation only. Useful for running in batches.",
    )
    args = p.parse_args()

    corpus_root = args.audio_corpus_root.resolve()
    analysis_root = args.output_dir.resolve()
    audit_root = analysis_root / "ITERATION_AUDIT"
    work_root = analysis_root / "_work"

    analysis_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    generator_py = repo_root / "scripts" / "oaas" / "make_bovine_ensembles_180s.py"
    extractor_py = repo_root / "scripts" / "oaas" / "extract_oaas_features.py"
    oaas_py = repo_root / "scripts" / "oaas" / "fit_oaas_model.py"
    distance_py = repo_root / "scripts" / "oaas" / "compute_oaas_distances.py"
    source_metadata = args.source_metadata.resolve()
    historical_full_features = args.historical_features.resolve()
    c2_ids_csv = args.c2_ids_csv.resolve()

    required_paths = [
        source_metadata,
        historical_full_features,
        c2_ids_csv,
        generator_py,
        extractor_py,
        oaas_py,
        distance_py,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Required input or script file is missing: " + ", ".join(missing_paths))
    if not (corpus_root / "analysis_ready_clean" / "BOVINE_CORE").is_dir():
        raise FileNotFoundError(
            "The audio corpus root must contain analysis_ready_clean/BOVINE_CORE/."
        )

    results_csv = analysis_root / "null_iteration_results.csv"
    run_log = analysis_root / "run_log.jsonl"

    extractor = load_module(extractor_py, "extract_oaas_features_null_primary")

    c2_table = pd.read_csv(c2_ids_csv)
    if "id" not in c2_table.columns:
        raise ValueError("The frozen C2 ID file must contain an 'id' column.")
    c2_ids = set(c2_table["id"].astype(str))
    if len(c2_ids) != 127:
        raise RuntimeError(f"Frozen C2 reference set should contain 127 IDs, got {len(c2_ids)}")

    if results_csv.exists():
        results = pd.read_csv(results_csv)
    else:
        results = pd.DataFrame(
            columns=[
                "attempt_order",
                "stage",
                "planned_index",
                "base_seed",
                "seed_role",
                "status",
                "valid_index",
                "c1",
                "c2",
                "c2_gc_n",
                "elapsed_sec",
                "error",
                "timestamp",
            ]
        )

    attempted = set(pd.to_numeric(results.get("base_seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    valid_existing = results[results["status"].astype(str) == "VALID"] if len(results) else results.copy()

    if args.stage == "primary":
        planned_seeds = list(range(PRIMARY_START, PRIMARY_START + PRIMARY_N))
        target_valid_total = PRIMARY_N
    else:
        planned_seeds = list(range(EXTENSION_START, EXTENSION_END + 1))
        target_valid_total = 1000

    # For extension, primary stage must already have at least 500 valid runs.
    if args.stage == "extension":
        primary_valid = int(
            ((results["status"].astype(str) == "VALID") & (results["stage"].astype(str) == "primary")).sum()
        ) if len(results) else 0
        if primary_valid < 500:
            raise RuntimeError("Extension stage requested before 500 valid primary realizations exist.")

    planned_queue = [s for s in planned_seeds if s not in attempted]
    reserve_seed = RESERVE_START
    while reserve_seed in attempted:
        reserve_seed += 1

    new_valid_this_call = 0
    attempt_order = len(results) + 1

    def append_log(obj: dict) -> None:
        with run_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    while True:
        current_valid_total = int((results["status"].astype(str) == "VALID").sum()) if len(results) else 0
        if args.stage == "primary":
            current_stage_valid = int(
                ((results["status"].astype(str) == "VALID") & (results["stage"].astype(str) == "primary")).sum()
            ) if len(results) else 0
            if current_stage_valid >= 500:
                break
        else:
            if current_valid_total >= 1000:
                break

        if args.max_new_valid is not None and new_valid_this_call >= args.max_new_valid:
            break

        if planned_queue:
            base_seed = planned_queue.pop(0)
            seed_role = "PLANNED"
            planned_index = (
                base_seed - PRIMARY_START + 1
                if args.stage == "primary"
                else base_seed - EXTENSION_START + 501
            )
        else:
            while reserve_seed in attempted:
                reserve_seed += 1
            base_seed = reserve_seed
            reserve_seed += 1
            seed_role = "TECHNICAL_REPLACEMENT"
            planned_index = None

        attempted.add(base_seed)
        tag = f"seed_{base_seed}"
        work_dir = work_root / tag
        ensemble_dir = work_dir / "ensembles"
        feature_dir = work_dir / "features"
        oaas_dir = work_dir / "oaas"
        manifest_dir = work_dir / "manifests"

        if work_dir.exists():
            shutil.rmtree(work_dir)
        for d in (ensemble_dir, feature_dir, oaas_dir, manifest_dir):
            d.mkdir(parents=True, exist_ok=True)

        ensemble_manifest = manifest_dir / "bovine_ensembles_180s.csv"
        ensemble_features = feature_dir / "features_ensemble_anchor.csv"
        combined_features = feature_dir / "features_full_with_obonuco_complete.csv"

        t0 = time.time()
        status = "INVALID_TECHNICAL"
        error = None
        c1 = np.nan
        c2 = np.nan
        c2_gc_n = np.nan
        manifest = None

        try:
            run_cmd(
                [
                    sys.executable,
                    str(generator_py),
                    "--metadata_csv",
                    str(source_metadata),
                    "--corpus_root",
                    str(corpus_root),
                    "--output_dir",
                    str(ensemble_dir),
                    "--output_csv",
                    str(ensemble_manifest),
                    "--target_sec",
                    "180",
                    "--sample_rate",
                    "22050",
                    "--ensembles_per_class",
                    "5",
                    "--seed",
                    str(base_seed),
                    "--trim_db",
                    "30",
                    "--target_peak",
                    "0.99",
                ]
            )

            wavs = sorted(ensemble_dir.glob("*.wav"))
            if len(wavs) != 20:
                raise RuntimeError(f"Expected 20 WAVs, got {len(wavs)}")

            manifest = pd.read_csv(ensemble_manifest)
            if len(manifest) != 20 or not validate_counts(manifest):
                raise RuntimeError("Ensemble manifest class/count validation failed.")

            ens_df = pd.read_csv(ensemble_manifest)
            extracted = extractor.extract_from_metadata(
                df=ens_df,
                audio_root=ensemble_dir,
                output_csv=ensemble_features,
                sample_rate=22050,
                mode="ensemble",
            )
            if len(extracted) != 20:
                raise RuntimeError(f"Expected 20 extracted feature rows, got {len(extracted)}")

            combined, _ = combine_anchor_with_frozen(
                historical_full_features,
                ensemble_features,
                combined_features,
            )

            run_cmd(
                [
                    sys.executable,
                    str(oaas_py),
                    "--features_csv",
                    str(combined_features),
                    "--output_dir",
                    str(oaas_dir),
                    "--anchor_domain",
                    "bovine_core_ensemble",
                    "--n_components",
                    "10",
                ]
            )

            run_cmd(
                [
                    sys.executable,
                    str(distance_py),
                    "--oaas_dir",
                    str(oaas_dir),
                ]
            )

            feature_manifest = pd.read_csv(oaas_dir / "feature_manifest.csv")
            pca_variance = pd.read_csv(oaas_dir / "pca_variance.csv")
            centroids = pd.read_csv(oaas_dir / "bovine_subdomain_centroids.csv")
            projected = pd.read_csv(oaas_dir / "OAAS_other_with_distances.csv")

            if len(feature_manifest) != 208:
                raise RuntimeError(f"Expected 208 OAAS features, got {len(feature_manifest)}")
            if len(pca_variance) != 10:
                raise RuntimeError(f"Expected 10 PCs, got {len(pca_variance)}")
            if len(centroids) != 4:
                raise RuntimeError(f"Expected 4 centroids, got {len(centroids)}")
            if len(projected) != 529:
                raise RuntimeError(f"Expected 529 projected objects, got {len(projected)}")

            obu = projected[projected["domain"].astype(str).str.lower() == "obonuco"]
            if len(obu) != 145:
                raise RuntimeError(f"Expected 145 Obonuco objects, got {len(obu)}")

            if not np.isfinite(
                projected[["dist_to_GC", "dist_to_LFC", "dist_to_HFC", "dist_to_PSC"]]
                .to_numpy(dtype=float)
            ).all():
                raise RuntimeError("Non-finite OAAS distances detected.")

            if not projected["nearest_bovine_class"].notna().all():
                raise RuntimeError("Missing nearest-bovine assignments.")

            if not c2_ids.issubset(set(projected["id"].astype(str))):
                raise RuntimeError("One or more frozen C2 IDs missing from projected data.")

            c1, c2, c2_gc_n = compute_iteration_statistics(
                oaas_dir / "pca_variance.csv",
                oaas_dir / "OAAS_other_with_distances.csv",
                c2_ids,
            )

            if not np.isfinite(c1) or not np.isfinite(c2):
                raise RuntimeError("Non-finite C1/C2 statistic.")

            status = "VALID"

        except Exception as e:
            error = repr(e)

        elapsed = time.time() - t0

        valid_index = np.nan
        if status == "VALID":
            current_valid_total = int((results["status"].astype(str) == "VALID").sum()) if len(results) else 0
            valid_index = current_valid_total + 1
            new_valid_this_call += 1

        row = pd.DataFrame(
            [{
                "attempt_order": attempt_order,
                "stage": args.stage,
                "planned_index": planned_index,
                "base_seed": base_seed,
                "seed_role": seed_role,
                "status": status,
                "valid_index": valid_index,
                "c1": c1,
                "c2": c2,
                "c2_gc_n": c2_gc_n,
                "elapsed_sec": elapsed,
                "error": error,
                "timestamp": now_iso(),
            }]
        )

        results = pd.concat([results, row], ignore_index=True)
        atomic_write_csv(results, results_csv)

        audit_dir = audit_root / f"{attempt_order:04d}_seed_{base_seed}"
        save_iteration_audit(
            audit_dir=audit_dir,
            base_seed=base_seed,
            planned_index=planned_index,
            status=status,
            error=error,
            manifest=manifest,
            feature_manifest_path=oaas_dir / "feature_manifest.csv",
            pca_variance_path=oaas_dir / "pca_variance.csv",
            centroids_path=oaas_dir / "bovine_subdomain_centroids.csv",
        )

        append_log(
            {
                "attempt_order": attempt_order,
                "base_seed": base_seed,
                "stage": args.stage,
                "seed_role": seed_role,
                "status": status,
                "elapsed_sec": elapsed,
                "timestamp": now_iso(),
            }
        )

        # Do not print C1/C2 values during execution.
        print(
            f"[{attempt_order:04d}] seed={base_seed} role={seed_role} "
            f"status={status} elapsed={elapsed:.1f}s"
        )

        if work_dir.exists():
            shutil.rmtree(work_dir)

        attempt_order += 1

    valid_total = int((results["status"].astype(str) == "VALID").sum()) if len(results) else 0
    primary_valid = int(
        ((results["status"].astype(str) == "VALID") & (results["stage"].astype(str) == "primary")).sum()
    ) if len(results) else 0

    print("\n=== PACKAGE C NULL EXECUTION CHECKPOINT ===")
    print(f"Primary valid realizations: {primary_valid}/500")
    print(f"Total valid realizations:   {valid_total}")
    print(f"New valid this invocation:  {new_valid_this_call}")
    print(f"Results checkpoint:         {results_csv}")
    print("C1/C2 values were stored for later locked summarization and were not printed.")

    # No p-values or distribution summaries are computed here.


if __name__ == "__main__":
    main()
