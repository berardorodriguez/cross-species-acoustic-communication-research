"""Compute five-second acoustic-information windows from 13 O1/O2 fragments.

The script creates derived window-level data and a count manifest. It does not
copy or redistribute the source audio files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

WINDOW_SECONDS = 5.0
HOP_SECONDS = 5.0
NFFT = 2048
LOW_BAND_HZ = (0, 2000)
HIGH_BAND_HZ = (2000, 8000)
GINI_BANDS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_window_metrics(segment: np.ndarray, sample_rate: int) -> dict[str, float]:
    frequencies, _, spectrum = stft(
        segment,
        fs=sample_rate,
        nperseg=NFFT,
        noverlap=NFFT // 2,
    )
    magnitude = np.abs(spectrum)
    aci = np.abs(np.diff(magnitude, axis=1)).sum() / (magnitude.sum() + 1e-12)

    band_edges = np.linspace(0, len(frequencies), GINI_BANDS + 1, dtype=int)
    band_energy = np.array([
        magnitude[band_edges[i]:band_edges[i + 1], :].sum()
        for i in range(GINI_BANDS)
    ])
    ordered_energy = np.sort(band_energy)
    n_bands = len(ordered_energy)
    gini = (
        np.sum((2 * np.arange(1, n_bands + 1) - n_bands - 1) * ordered_energy)
        / (n_bands * ordered_energy.sum() + 1e-12)
    )

    high_mask = (frequencies >= HIGH_BAND_HZ[0]) & (frequencies < HIGH_BAND_HZ[1])
    low_mask = (frequencies >= LOW_BAND_HZ[0]) & (frequencies < LOW_BAND_HZ[1])
    high_magnitude = magnitude[high_mask, :].sum()
    low_magnitude = magnitude[low_mask, :].sum()
    normalized_difference = (high_magnitude - low_magnitude) / (
        high_magnitude + low_magnitude + 1e-12
    )

    return {
        "ACI": float(aci),
        "AEI": float(gini),
        "NDSI": float(normalized_difference),
        "BI": float(high_magnitude),
    }


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio), int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Folder containing O1/O1_fragments and O2/O2_fragments.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    source_files: list[dict] = []
    for work in ("O1", "O2"):
        audio_dir = args.source_root / work / f"{work}_fragments"
        paths = sorted(audio_dir.glob("*.wav"))
        if len(paths) != 13:
            raise ValueError(f"Expected 13 {work} fragments in {audio_dir}; found {len(paths)}")
        for path in paths:
            audio, sample_rate = read_mono(path)
            duration = len(audio) / sample_rate
            n_windows = 0
            for start in np.arange(0, duration, HOP_SECONDS):
                end = start + WINDOW_SECONDS
                if end > duration:
                    break
                segment = audio[int(start * sample_rate):int(end * sample_rate)]
                row = {
                    "work": work,
                    "file": path.name,
                    "window_start_s": float(start),
                    "window_end_s": float(end),
                }
                row.update(compute_window_metrics(segment, sample_rate))
                rows.append(row)
                n_windows += 1
            source_files.append({
                "work": work,
                "file": path.name,
                "sha256": sha256(path),
                "sample_rate_hz": sample_rate,
                "channels_in_source": int(sf.info(path).channels),
                "duration_s": duration,
                "complete_5s_windows": n_windows,
            })

    result = pd.DataFrame(rows)
    expected = {"O1": 276, "O2": 276}
    actual = result.groupby("work").size().to_dict()
    if actual != expected:
        raise ValueError(f"Unexpected window counts: {actual}; expected {expected}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False, float_format="%.15g")

    script_path = Path(__file__).resolve()
    manifest = {
        "description": "Recomputed 5-s non-overlapping acoustic-information windows from 13 source fragments per work.",
        "audio_files_included": False,
        "source_root": str(args.source_root.resolve()),
        "source_files": source_files,
        "total_source_duration_s": {
            work: float(sum(item["duration_s"] for item in source_files if item["work"] == work))
            for work in ("O1", "O2")
        },
        "windows_per_work": actual,
        "total_windows": len(result),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "stft_nperseg": NFFT,
        "stft_default_boundary_and_padding": True,
        "mono_conversion": "arithmetic mean across channels when the source is multichannel",
        "operational_formulas": {
            "ACI": "sum(abs(diff(abs(STFT magnitude), time axis))) / sum(abs(STFT magnitude))",
            "AEI": "Gini coefficient of summed STFT magnitude across 20 linearly partitioned frequency-index bands",
            "NDSI": "(sum magnitude 2-8 kHz - sum magnitude 0-2 kHz) / (sum magnitude 2-8 kHz + sum magnitude 0-2 kHz)",
            "BI": "sum STFT magnitude from 2 kHz to below 8 kHz; reader-facing name: 2-8 kHz spectral magnitude",
        },
        "source_rebuild_script": str(script_path),
        "source_rebuild_script_sha256": sha256(script_path),
        "derived_csv": str(args.out_csv.resolve()),
        "derived_csv_sha256": sha256(args.out_csv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"windows": actual, "total": len(result), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
