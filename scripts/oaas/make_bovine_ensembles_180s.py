from __future__ import annotations
import argparse, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import librosa, numpy as np, pandas as pd, soundfile as sf

VALID_CLASSES = ["gc", "lfc", "hfc", "psc"]

def parse_args():
    p = argparse.ArgumentParser(description="Build 180-second bovine ensembles by class.")
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--corpus_root", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--target_sec", type=float, default=180.0)
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--ensembles_per_class", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trim_db", type=float, default=30.0)
    p.add_argument("--target_peak", type=float, default=0.99)
    return p.parse_args()

def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"domain", "subdomain"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    return df

def include_row(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "1", "true", "yes", "y"}:
            return True
        if v in {"0", "false", "no", "n"}:
            return False
        return True
    try:
        return int(value) != 0
    except Exception:
        return True

def infer_clean_audio_path(row: pd.Series, corpus_root: Path) -> Path:
    rel_processed = str(row.get("relative_path_processed", "") or "").strip()
    analysis_name = str(row.get("analysis_name", "") or "").strip()
    if rel_processed:
        clean_rel = rel_processed.replace("analysis_ready/", "analysis_ready_clean/", 1)
        cand = corpus_root / Path(clean_rel)
        if cand.exists():
            return cand
    if analysis_name:
        cand = corpus_root / "analysis_ready_clean" / "BOVINE_CORE" / analysis_name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve clean path for {analysis_name!r}")

def gather_bovine_items(df: pd.DataFrame, corpus_root: Path) -> Dict[str, List[Tuple[Path, str]]]:
    bov = df[df["domain"] == "bovine_core"].copy()
    if "include" in bov.columns:
        bov = bov[bov["include"].apply(include_row)]
    bov["subdomain"] = bov["subdomain"].astype(str).str.strip().str.lower()
    bov = bov[bov["subdomain"].isin(VALID_CLASSES)]
    by_class = defaultdict(list)
    for _, row in bov.iterrows():
        cls = row["subdomain"]
        by_class[cls].append((infer_clean_audio_path(row, corpus_root), str(row.get("id", ""))))
    for cls in VALID_CLASSES:
        if not by_class.get(cls):
            raise ValueError(f"No files found for class: {cls}")
    return by_class

def read_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y.astype(np.float32, copy=False)

def trim_silence(y: np.ndarray, top_db: float) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if yt.size > 0 else y

def peak_normalize(y: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y / m * peak if m > 0 else y

def build_ensemble(items, target_duration_sec, sr, seed, trim_db, target_peak):
    rng = random.Random(seed)
    target_samples = int(round(target_duration_sec * sr))
    pool = items.copy()
    rng.shuffle(pool)
    idx = 0
    chunks, used_ids, total = [], [], 0
    while total < target_samples:
        if idx >= len(pool):
            rng.shuffle(pool)
            idx = 0
        path, source_id = pool[idx]
        idx += 1
        y = trim_silence(read_audio_mono(path, sr), trim_db)
        if y.size == 0:
            continue
        remaining = target_samples - total
        if y.size <= remaining:
            chunk = y
        else:
            start_max = max(0, y.size - remaining)
            start = rng.randint(0, start_max) if start_max > 0 else 0
            chunk = y[start:start + remaining]
        chunks.append(chunk)
        used_ids.append(source_id)
        total += chunk.size
    ens = peak_normalize(np.concatenate(chunks, axis=0), target_peak)
    return ens, used_ids

def make_output_name(subdomain: str, idx: int) -> str:
    return f"BCORE_{subdomain.upper()}_ENS_{idx:03d}.wav"

def main():
    args = parse_args()
    corpus_root = args.corpus_root.resolve()
    df = load_metadata(args.metadata_csv.resolve())
    by_class = gather_bovine_items(df, corpus_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in VALID_CLASSES:
        items = by_class[cls]
        for i in range(1, args.ensembles_per_class + 1):
            ensemble_seed = args.seed + (1000 * VALID_CLASSES.index(cls)) + i
            y, used_ids = build_ensemble(items, args.target_sec, args.sample_rate, ensemble_seed, args.trim_db, args.target_peak)
            out_name = make_output_name(cls, i)
            out_path = args.output_dir / out_name
            sf.write(out_path.as_posix(), y, args.sample_rate, subtype="PCM_16")
            rel_out = out_path.resolve().relative_to(corpus_root).as_posix()
            rows.append({
                "id": f"BCORE_{cls.upper()}_ENS_{i:03d}",
                "file_name": out_name,
                "relative_path": rel_out,
                "domain": "bovine_core_ensemble",
                "subdomain": cls,
                "tradition": "bovine_ensemble",
                "function": cls,
                "signal_type": "animal_vocal",
                "recording_type": "derived",
                "processing_level": "concatenated_peak_normalized",
                "analysis_unit": f"ensemble_{int(args.target_sec)}s",
                "certainty": "high",
                "include": 1,
                "notes": f"Built from bovine_core/{cls}; target_sec={args.target_sec}; seed={ensemble_seed}; n_sources={len(used_ids)}; source_ids=" + "|".join(used_ids),
                "original_name": "",
                "analysis_name": out_name,
                "relative_path_original": "",
                "relative_path_processed": rel_out,
            })
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print("[OK] Bovine source counts by class:")
    for cls in VALID_CLASSES:
        print(f"  - {cls}: {len(by_class[cls])} source files")
    print(f"[OK] Ensembles written: {len(rows)}")
    print(f"[OK] Audio output dir:  {args.output_dir}")
    print(f"[OK] Metadata CSV:      {args.output_csv}")

if __name__ == "__main__":
    main()
