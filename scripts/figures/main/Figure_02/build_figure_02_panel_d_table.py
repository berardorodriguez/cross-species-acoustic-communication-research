from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CLASSES = ["GC", "LFC", "HFC", "PSC"]
EXPECTED = {
    ("GC", "GC"): 1,
    ("HFC", "GC"): 117,
    ("HFC", "LFC"): 4,
    ("HFC", "HFC"): 4,
    ("HFC", "PSC"): 2,
    ("PSC", "GC"): 2,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_class(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper()
            .str.replace("DIST_TO_", "", regex=False)
            .str.replace("NEAREST_", "", regex=False)
            .str.replace("CLASS_", "", regex=False))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild the exact-ID 130-fragment source table for Figure 2D."
    )
    ap.add_argument("--micro", required=True, type=Path)
    ap.add_argument("--macro", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    required = {"id", "domain", "subdomain", "function", "analysis_unit", "nearest_bovine_class"}
    micro = pd.read_csv(args.micro)
    macro = pd.read_csv(args.macro)
    for label, df in (("MICRO", micro), ("MACRO", macro)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} source is missing columns: {sorted(missing)}")
    micro = micro.loc[micro["domain"].astype(str).str.lower().eq("obonuco")].copy()
    macro = macro.loc[macro["domain"].astype(str).str.lower().eq("obonuco")].copy()
    if micro["id"].duplicated().any() or macro["id"].duplicated().any():
        raise ValueError("Duplicate Obonuco IDs found; exact one-to-one join is not valid.")
    if set(micro["id"]) != set(macro["id"]):
        raise ValueError("MICRO and MACRO Obonuco ID sets differ.")

    left = micro[["id", "subdomain", "function", "analysis_unit", "nearest_bovine_class"]].rename(
        columns={"nearest_bovine_class": "class_micro"}
    )
    right = macro[["id", "subdomain", "function", "analysis_unit", "nearest_bovine_class"]].rename(
        columns={
            "subdomain": "subdomain_macro",
            "function": "function_macro",
            "analysis_unit": "analysis_unit_macro",
            "nearest_bovine_class": "class_macro",
        }
    )
    pairs = left.merge(right, on="id", how="inner", validate="one_to_one")
    for col in ("subdomain", "function", "analysis_unit"):
        if not (pairs[col].astype(str) == pairs[f"{col}_macro"].astype(str)).all():
            raise ValueError(f"MICRO/MACRO {col} differs for one or more matched IDs.")

    pairs["class_micro"] = normalize_class(pairs["class_micro"])
    pairs["class_macro"] = normalize_class(pairs["class_macro"])
    if not pairs["class_micro"].isin(CLASSES).all() or not pairs["class_macro"].isin(CLASSES).all():
        raise ValueError("Unexpected nearest-reference class in Obonuco pairs.")
    fragments = pairs.loc[pairs["analysis_unit"].astype(str).str.lower().eq("fragment"),
                          ["id", "subdomain", "function", "analysis_unit", "class_micro", "class_macro"]].copy()
    if len(pairs) != 145 or len(fragments) != 130 or fragments["id"].duplicated().any():
        raise ValueError(f"Expected 145 exact matches and 130 unique fragments; got {len(pairs)} and {len(fragments)}.")

    observed = {(str(row.class_micro), str(row.class_macro)): int(row.n)
                for row in fragments.groupby(["class_micro", "class_macro"]).size()
                .rename("n").reset_index().itertuples(index=False)}
    if observed != EXPECTED:
        raise ValueError(f"Fragment correspondence differs from the expected packaged table: {observed}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    table_path = args.outdir / "Figure02_D_Obonuco_130_fragments_exact_id.csv"
    fragments.sort_values("id", kind="stable").to_csv(table_path, index=False)
    manifest = {
        "figure": 2,
        "panel": "D",
        "status": "EXACT_ID_TABLE_REBUILT_AND_AGGREGATE_VERIFIED",
        "join_key": "id",
        "domain_filter": "domain == obonuco",
        "fragment_filter": "analysis_unit == fragment",
        "micro_source": {
            "path": str(args.micro), "sha256": sha256(args.micro),
            "bytes": args.micro.stat().st_size, "obonuco_rows": len(micro),
        },
        "macro_source": {
            "path": str(args.macro), "sha256": sha256(args.macro),
            "bytes": args.macro.stat().st_size, "obonuco_rows": len(macro),
        },
        "matched_obonuco_objects": len(pairs),
        "unmatched_ids": 0,
        "duplicate_ids": 0,
        "matched_fragments": len(fragments),
        "fragment_correspondence_counts": {
            f"{a}->{b}": n for (a, b), n in sorted(observed.items())
        },
        "derived_table": str(table_path),
        "derived_table_sha256": sha256(table_path),
        "note": "Only identifiers, matching metadata, analysis-unit label, and two nearest-reference classes are retained; source feature matrices and audio are not copied.",
    }
    manifest_path = args.outdir / "Figure02_D_Obonuco_130_fragments_exact_id_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Matched Obonuco objects: {len(pairs)}; unmatched IDs: 0")
    print(f"Exact-ID fragment rows: {len(fragments)}")
    print(f"Derived table: {table_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
