from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import librosa

EPS = 1e-10


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


def rms_normalize(y: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + EPS
    y = y * (target_rms / rms)
    peak = np.max(np.abs(y)) + EPS
    if peak > 0.99:
        y = y / peak * 0.99
    return y


def preprocess_audio(path: Path, sr: int) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path.as_posix(), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")

    y = y - np.mean(y)

    # Trim conservador
    yt, _ = librosa.effects.trim(y, top_db=30)
    if yt.size > 0:
        y = yt

    if y.size < 64:
        raise ValueError(f"Audio too short after preprocessing: {path}")

    y = librosa.effects.preemphasis(y, coef=0.97)
    y = rms_normalize(y, target_rms=0.08)
    return y, sr


def choose_fft_params(y_len: int) -> Tuple[int, int]:
    if y_len < 64:
        raise ValueError("Signal too short for spectral analysis.")

    n_fft = min(1024, y_len)

    # Mayor potencia de 2 <= n_fft
    pow2 = 2 ** int(np.floor(np.log2(max(n_fft, 32))))
    n_fft = max(32, pow2)

    hop_length = max(16, n_fft // 4)
    return n_fft, hop_length


def summarize_vector(prefix: str, x: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def summarize_matrix(prefix: str, x: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(x.shape[0]):
        out[f"{prefix}{i+1:02d}_mean"] = float(np.mean(x[i]))
        out[f"{prefix}{i+1:02d}_std"] = float(np.std(x[i]))
    return out


def hz_stats_from_f0(f0: np.ndarray) -> Dict[str, float]:
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        return {
            "f0_mean": np.nan,
            "f0_std": np.nan,
            "f0_median": np.nan,
            "f0_p10": np.nan,
            "f0_p90": np.nan,
            "voiced_ratio": 0.0,
            "jitter_proxy": np.nan,
        }

    diffs = np.abs(np.diff(voiced))
    jitter_proxy = np.mean(diffs) / (np.mean(voiced) + EPS) if voiced.size > 1 else 0.0

    return {
        "f0_mean": float(np.mean(voiced)),
        "f0_std": float(np.std(voiced)),
        "f0_median": float(np.median(voiced)),
        "f0_p10": float(np.percentile(voiced, 10)),
        "f0_p90": float(np.percentile(voiced, 90)),
        "voiced_ratio": float(voiced.size / max(1, f0.size)),
        "jitter_proxy": float(jitter_proxy),
    }


def compute_modulation_features(y: np.ndarray, sr: int, hop_length: int) -> Dict[str, float]:
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.size < 3:
        return {
            "env_mod_mean_absdiff": np.nan,
            "env_mod_std": np.nan,
            "env_mod_rate_proxy": np.nan,
        }

    d = np.diff(rms)
    zero_cross = np.sum(np.diff(np.signbit(d)) != 0)
    duration_sec = len(y) / sr

    return {
        "env_mod_mean_absdiff": float(np.mean(np.abs(d))),
        "env_mod_std": float(np.std(d)),
        "env_mod_rate_proxy": float(zero_cross / max(duration_sec, EPS)),
    }


def compute_band_energy_features(S_power: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    def band_energy(lo: float, hi: float) -> float:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.mean(np.sum(S_power[idx, :], axis=0)))

    e_0_1k = band_energy(0, 1000)
    e_1_3k = band_energy(1000, 3000)
    e_3_8k = band_energy(3000, 8000)
    e_total = e_0_1k + e_1_3k + e_3_8k + EPS

    return {
        "energy_0_1k": e_0_1k,
        "energy_1_3k": e_1_3k,
        "energy_3_8k": e_3_8k,
        "energy_ratio_0_1k": float(e_0_1k / e_total),
        "energy_ratio_1_3k": float(e_1_3k / e_total),
        "energy_ratio_3_8k": float(e_3_8k / e_total),
    }


def extract_features_for_file(path: Path, sr: int) -> Dict[str, float]:
    y, sr = preprocess_audio(path, sr)
    n_fft, hop_length = choose_fft_params(len(y))

    duration_sec = len(y) / sr

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # spectral_contrast puede fallar en señales muy cortas o con n_fft pequeño
    try:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    except Exception:
        contrast = np.full((7, 1), np.nan)

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=n_fft)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )
    except Exception:
        f0 = np.array([np.nan])

    try:
        chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=hop_length)
    except Exception:
        chroma = np.full((12, 1), np.nan)

    try:
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    except Exception:
        tonnetz = np.full((6, 1), np.nan)

    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_rms = np.sqrt(np.mean(y_harm ** 2))
        perc_rms = np.sqrt(np.mean(y_perc ** 2))
        hp_ratio = float(harm_rms / (perc_rms + EPS))
        hp_share_harm = float(harm_rms / (harm_rms + perc_rms + EPS))
        hp_share_perc = float(perc_rms / (harm_rms + perc_rms + EPS))
    except Exception:
        hp_ratio = np.nan
        hp_share_harm = np.nan
        hp_share_perc = np.nan

    mean_spec = np.mean(S_power, axis=1)
    mean_spec = mean_spec / (np.sum(mean_spec) + EPS)
    spectral_entropy = float(-np.sum(mean_spec * np.log(mean_spec + EPS)))

    out: Dict[str, float] = {
        "duration_sec": float(duration_sec),
        "sample_rate": float(sr),
        "n_samples": float(len(y)),
        "peak_abs": float(np.max(np.abs(y))),
        "rms_global": float(np.sqrt(np.mean(y ** 2))),
        "onset_strength_mean": float(np.mean(onset_env)) if onset_env.size else np.nan,
        "onset_strength_std": float(np.std(onset_env)) if onset_env.size else np.nan,
        "spectral_entropy": spectral_entropy,
        "hp_ratio_harm_over_perc": hp_ratio,
        "hp_share_harm": hp_share_harm,
        "hp_share_perc": hp_share_perc,
    }

    out.update(hz_stats_from_f0(f0))
    out.update(summarize_matrix("mfcc", mfcc))
    out.update(summarize_matrix("mfcc_delta", mfcc_delta))
    out.update(summarize_matrix("mfcc_delta2", mfcc_delta2))
    out.update(summarize_matrix("contrast", contrast))
    out.update(summarize_matrix("chroma", chroma))
    out.update(summarize_matrix("tonnetz", tonnetz))
    out.update(summarize_vector("centroid", centroid))
    out.update(summarize_vector("bandwidth", bandwidth))
    out.update(summarize_vector("rolloff", rolloff))
    out.update(summarize_vector("flatness", flatness))
    out.update(summarize_vector("zcr", zcr))
    out.update(summarize_vector("rms_frame", rms))
    out.update(compute_modulation_features(y, sr, hop_length=hop_length))
    out.update(compute_band_energy_features(S_power, freqs))

    return out


def resolve_main_audio_path(audio_root: Path, row: pd.Series) -> Path:
    rel_processed = str(row.get("relative_path_processed", "") or "").strip()
    analysis_name = str(row.get("analysis_name", "") or "").strip()

    candidates = []

    if rel_processed:
        clean_rel = rel_processed.replace("analysis_ready/", "", 1)
        candidates.append(audio_root / Path(clean_rel))
        candidates.append((audio_root / Path(clean_rel)).with_suffix(".wav"))

    if analysis_name:
        candidates.append(audio_root / Path("BOVINE_CORE") / analysis_name)
        candidates.append(audio_root / Path(analysis_name))

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Main audio not found for id={row.get('id', '')}")


def resolve_ensemble_audio_path(audio_root: Path, row: pd.Series) -> Path:
    rel = str(row.get("relative_path", "") or "").strip()
    file_name = str(row.get("file_name", "") or "").strip()

    candidates = []
    if rel:
        candidates.append(audio_root.parent / Path(rel))
        candidates.append(audio_root / Path(rel).name)
    if file_name:
        candidates.append(audio_root / file_name)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Ensemble audio not found for id={row.get('id', '')}")


def extract_from_metadata(
    df: pd.DataFrame,
    audio_root: Path,
    output_csv: Path,
    sample_rate: int,
    mode: str,
) -> pd.DataFrame:
    rows = []
    errors = []

    df_use = df.copy()
    if "include" in df_use.columns:
        df_use = df_use[df_use["include"].apply(include_row)].copy()

    for _, row in df_use.iterrows():
        try:
            if mode == "main":
                audio_path = resolve_main_audio_path(audio_root, row)
            elif mode == "ensemble":
                audio_path = resolve_ensemble_audio_path(audio_root, row)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            feats = extract_features_for_file(audio_path, sr=sample_rate)

            base = row.to_dict()
            base["resolved_audio_path"] = str(audio_path)
            base.update(feats)
            rows.append(base)

            print(f"[OK] {row.get('id', '')} -> {audio_path.name}")

        except Exception as e:
            errors.append({
                "id": row.get("id", ""),
                "file_name": row.get("file_name", ""),
                "relative_path": row.get("relative_path", ""),
                "error": str(e),
            })
            print(f"[ERR] {row.get('id', '')}: {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    if errors:
        err_csv = output_csv.with_name(output_csv.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_csv, index=False)
        print(f"[WARN] Wrote error log to: {err_csv}")

    print(f"[OK] Wrote feature CSV: {output_csv}")
    print(f"[OK] Rows written: {len(out_df)}")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OAAS-ready classical acoustic features.")
    parser.add_argument("--main_metadata", type=Path, required=True)
    parser.add_argument("--main_audio_root", type=Path, required=True)
    parser.add_argument("--main_output_csv", type=Path, required=True)
    parser.add_argument("--ensemble_metadata", type=Path, default=None)
    parser.add_argument("--ensemble_audio_root", type=Path, default=None)
    parser.add_argument("--ensemble_output_csv", type=Path, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    main_df = pd.read_csv(args.main_metadata)
    extract_from_metadata(
        df=main_df,
        audio_root=args.main_audio_root,
        output_csv=args.main_output_csv,
        sample_rate=args.sample_rate,
        mode="main",
    )

    if args.ensemble_metadata is not None:
        if args.ensemble_audio_root is None or args.ensemble_output_csv is None:
            raise ValueError(
                "If --ensemble_metadata is provided, you must also provide "
                "--ensemble_audio_root and --ensemble_output_csv."
            )

        ens_df = pd.read_csv(args.ensemble_metadata)
        extract_from_metadata(
            df=ens_df,
            audio_root=args.ensemble_audio_root,
            output_csv=args.ensemble_output_csv,
            sample_rate=args.sample_rate,
            mode="ensemble",
        )


if __name__ == "__main__":
    main()
