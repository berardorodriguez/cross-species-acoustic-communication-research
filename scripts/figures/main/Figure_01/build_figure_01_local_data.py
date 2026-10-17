"""Build CC BY 4.0 Figure 1 acoustic tables from author-collected GC/PSC audio.

The source WAV files are supplied locally by the author and are not copied into
this repository. The script deliberately filters the recorded Figure 1 input
manifest to GC and PSC, the Colombian recordings collected by the authors; it
does not process the external BovineTalk LFC/HFC recordings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import get_window, savgol_filter


SR = 48_000
N_FFT = 1024
HOP = 256
N_FFT_PANEL_C = 2048
HOP_PANEL_C = 512
MAX_DURATION_S = 4.0
NORM_TIME = 300
FMAX = 3500
BANDS = {
    "Low (0–1 kHz)": (0, 1000),
    "Mid (1–3 kHz)": (1000, 3000),
    "High (3–8 kHz)": (3000, 8000),
}
EXAMPLES = {
    "GC": Path("TP_TF_vocalizations/TP1.wav"),
    "PSC": Path("TP_TF_vocalizations/TF1.wav"),
}
LOCAL_CATEGORIES = {"GC", "PSC"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_verified_manifest(path: Path, audio_root: Path, categories: set[str]) -> list[tuple[str, Path]]:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            category = row["Category"]
            if category not in categories:
                continue
            source = audio_root / "BOV_WAV_48K" / row["RelativePath"]
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.stat().st_size != int(row["Bytes"]):
                raise ValueError(f"Size mismatch: {source}")
            if sha256(source) != row["SHA256"].upper():
                raise ValueError(f"SHA-256 mismatch: {source}")
            records.append((category, source))
    if not records:
        raise ValueError("No local GC/PSC recordings found in the input manifest")
    return records


def load_example(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        y = soxr.resample(y, sr, SR, quality="HQ")
    return y[: int(MAX_DURATION_S * SR)]


def representative_spectrogram(y: np.ndarray) -> np.ndarray:
    centered = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(centered, N_FFT)[::HOP]
    window = get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=-1)).T
    reference = max(float(magnitude.max()), 1e-5)
    decibels = 20 * np.log10(np.maximum(magnitude, 1e-5) / reference)
    decibels = np.maximum(decibels, decibels.max() - 80.0)
    bin_count = int(FMAX // (SR / N_FFT)) + 1
    return decibels[:bin_count]


def band_envelopes(path: Path) -> dict[str, np.ndarray]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR or y.ndim != 1:
        raise ValueError(f"Unexpected source format for {path}: sr={sr}, shape={y.shape}")
    centered = np.pad(y, (N_FFT_PANEL_C // 2, N_FFT_PANEL_C // 2), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(centered, N_FFT_PANEL_C)[::HOP_PANEL_C]
    window = get_window("hann", N_FFT_PANEL_C, fftbins=True).astype(np.float32)
    power = (np.abs(np.fft.rfft(frames * window, n=N_FFT_PANEL_C, axis=-1)) ** 2).T
    if power.shape[1] < NORM_TIME:
        power = np.pad(power, ((0, 0), (0, NORM_TIME - power.shape[1])))
    else:
        power = power[:, :NORM_TIME]
    frequencies = np.fft.rfftfreq(N_FFT_PANEL_C, d=1 / sr)
    output = {}
    for label, (low, high) in BANDS.items():
        indices = np.where((frequencies >= low) & (frequencies <= high))[0]
        curve = power[indices, :].mean(axis=0)
        output[label] = curve / (curve.max() + 1e-12)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Author audio root with BOV_WAV_48K and TP_TF_vocalizations")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/derived/figure_01/local_colombian"))
    parser.add_argument("--input-manifest", type=Path,
                        default=Path("figures/main/Figure_01/02_PROVENANCE/SOURCE_MANIFESTS/Figure01_PanelC_audio_input_manifest.csv"))
    args = parser.parse_args()
    audio_root = args.audio_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = read_verified_manifest(args.input_manifest.resolve(), audio_root, LOCAL_CATEGORIES)
    profiles: dict[str, dict[str, list[np.ndarray]]] = {
        category: {band: [] for band in BANDS} for category in LOCAL_CATEGORIES
    }
    counts = {category: 0 for category in LOCAL_CATEGORIES}
    for category, path in records:
        for band, values in band_envelopes(path).items():
            profiles[category][band].append(values)
        counts[category] += 1

    profile_path = output / "Figure01_local_GC_PSC_band_profiles.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "normalized_time", "band", "relative_energy"])
        for category in ("GC", "PSC"):
            for band in BANDS:
                mean = np.mean(np.vstack(profiles[category][band]), axis=0)
                mean = savgol_filter(mean, 21, 3)
                for index, value in enumerate(mean):
                    writer.writerow([category, f"{index / (NORM_TIME - 1):.8f}", band, f"{value:.12g}"])

    spec_path = output / "Figure01_local_GC_PSC_representative_spectrograms.csv"
    bin_count = int(FMAX // (SR / N_FFT)) + 1
    bin_names = [f"db_bin_{index:03d}" for index in range(bin_count)]
    with spec_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "frame_index", *bin_names])
        for category in ("GC", "PSC"):
            source = audio_root / EXAMPLES[category]
            if not source.is_file():
                raise FileNotFoundError(source)
            source_manifest = args.input_manifest.resolve().parent / "Figure01_PanelsAB_representative_audio_manifest.csv"
            with source_manifest.open("r", encoding="utf-8-sig", newline="") as manifest_stream:
                matches = [row for row in csv.DictReader(manifest_stream) if row["Panel"] == category]
            if len(matches) != 1:
                raise ValueError(f"Expected one representative audio manifest row for {category}")
            record = matches[0]
            if source.stat().st_size != int(record["Bytes"]) or sha256(source) != record["SHA256"].upper():
                raise ValueError(f"Representative audio differs from its provenance record: {category}")
            values = representative_spectrogram(load_example(source))
            for frame in range(values.shape[1]):
                writer.writerow([category, frame, *[f"{x:.8g}" for x in values[:, frame]]])

    metadata = {
        "description": "Derived acoustic plot values from author-collected Colombian GC and PSC recordings only.",
        "source_audio_included": False,
        "third_party_audio_processed": False,
        "categories": {"GC": "Grazing call", "PSC": "Pre-slaughter call"},
        "panel_c_source_recording_counts": counts,
        "sample_rate_hz": SR,
        "representative_spectrogram": {"fft_size": N_FFT, "hop_length_samples": HOP,
            "maximum_plotted_frequency_hz": FMAX, "minimum_display_level_db": -80.0,
            "maximum_display_level_db": 0.0, "frequency_bins": bin_count},
        "panel_c_processing": {"fft_size": N_FFT_PANEL_C, "hop_length_samples": HOP_PANEL_C,
            "normalized_frames": NORM_TIME, "frequency_bands_hz": {k: list(v) for k, v in BANDS.items()},
            "per_recording_and_band_maximum_normalization": True,
            "category_mean_smoothing": {"method": "Savitzky-Golay", "window_length": 21, "polynomial_order": 3}},
        "files": [profile_path.name, spec_path.name],
    }
    (output / "Figure01_local_GC_PSC_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_counts": counts, "included_audio": False,
                      "third_party_audio_processed": False}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
