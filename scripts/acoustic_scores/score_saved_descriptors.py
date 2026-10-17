import argparse
import os
import glob
import numpy as np
import pandas as pd
import librosa

# ---------------- Utils ----------------
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def safe_div(a, b, eps=1e-8):
    return a / (b + eps)

def hz_to_bin(freq, sr, n_fft):
    return int(np.clip(np.round(freq * n_fft / sr), 0, n_fft//2))

def band_energy(S, sr, n_fft, f_lo, f_hi):
    i0 = hz_to_bin(f_lo, sr, n_fft)
    i1 = hz_to_bin(f_hi, sr, n_fft)
    i0, i1 = min(i0, i1), max(i0, i1)
    band = S[i0:i1+1, :]
    return float(np.sum(band))

def spectral_tilt_db_per_oct(S, freqs_hz):
    mag = np.mean(S, axis=1) + 1e-12
    log_mag = 20*np.log10(mag)
    log_freq = np.log2(freqs_hz + 1e-6)
    slope, _ = np.polyfit(log_freq, log_mag, 1)
    return float(slope)  # dB/oct aprox

def hnr_proxy(y, sr):
    y_h = librosa.effects.harmonic(y)
    y_p = y - y_h
    Eh = float(np.sum(y_h**2)); En = float(np.sum(y_p**2))
    return 10*np.log10((Eh + 1e-9) / (En + 1e-9))

def roughness_proxy(S):
    flatness = librosa.feature.spectral_flatness(S=np.abs(S)).mean()
    return float(flatness)

def onset_density(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))
    dens = len(onsets) / max(duration_s, 1e-6)
    return float(np.asarray(tempo).item()), float(dens)

def staccato_ratio(rms_series):
    thr = 0.5 * float(np.median(rms_series))
    short_frames = int(np.sum(rms_series < thr))
    return safe_div(short_frames, len(rms_series))

def snr_proxy(y, sr):
    hop = 512; frame = 2048
    S = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop).flatten()
    noise_floor = float(np.percentile(S, 10))
    signal_level = float(np.percentile(S, 90))
    return 20*np.log10((signal_level + 1e-9) / (noise_floor + 1e-9))

# ------------- Métricas por audio -------------
def compute_metrics(audio_path, target_sr=22050):
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))
    n_fft = 4096; hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())
    rolloff85 = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    rms_series = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop).flatten()
    rms = float(rms_series.mean())

    E_low = band_energy(S, sr, n_fft, 0, 1000)
    E_mid = band_energy(S, sr, n_fft, 1000, 3000)
    E_high = band_energy(S, sr, n_fft, 3000, 8000 if sr/2>8000 else sr/2)
    ratio_mid_low = safe_div(E_mid, E_low)
    ratio_high_low = safe_div(E_high, E_low)

    tilt = spectral_tilt_db_per_oct(S, freqs)
    hnr = hnr_proxy(y, sr)
    rough = roughness_proxy(S)

    tempo, dens = onset_density(y, sr)
    stacc = float(staccato_ratio(rms_series))
    snr_val = float(snr_proxy(y, sr))

    return {
        "file": os.path.basename(audio_path),
        "duration_s": duration_s,
        "spec_centroid_hz": centroid,
        "spec_bandwidth_hz": bandwidth,
        "rolloff_85_hz": rolloff85,
        "zcr_mean": zcr,
        "rms_mean": rms,
        "E_low_0_1k": E_low,
        "E_mid_1_3k": E_mid,
        "E_high_3_8k": E_high,
        "ratio_mid_low": ratio_mid_low,
        "ratio_high_low": ratio_high_low,
        "spectral_tilt_db_per_oct": tilt,
        "hnr_mean_db": hnr,
        "roughness": rough,
        "bpm": tempo,
        "onsets_per_s": dens,
        "staccato_ratio": stacc,
        "snr_db": snr_val,
    }

# ------------- IQR loading -------------
def load_iqr_csv(iqr_csv):
    if not iqr_csv or (not os.path.exists(iqr_csv)):
        return {}
    df = pd.read_csv(iqr_csv)
    out = {}
    for _, r in df.iterrows():
        metric = str(r["metric"])
        q1 = float(r.get("q1", np.nan)); med = float(r.get("median", np.nan)); q3 = float(r.get("q3", np.nan))
        if not np.isnan(q1) and not np.isnan(med) and not np.isnan(q3):
            out[metric] = {"q1": q1, "median": med, "q3": q3, "iqr": max(q3-q1, 1e-9)}
    return out

# Direcciones (qué lado es "peor")
DIRECTIONS = {
    "spec_centroid_hz": "higher_is_worse",
    "spec_bandwidth_hz": "higher_is_worse",
    "rolloff_85_hz": "higher_is_worse",
    "zcr_mean": "higher_is_worse",
    "rms_mean": "higher_is_worse",
    "ratio_mid_low": "higher_is_worse",
    "ratio_high_low": "higher_is_worse",
    "spectral_tilt_db_per_oct": "higher_is_worse",
    "hnr_mean_db": "lower_is_worse",
    "roughness": "higher_is_worse",
    "bpm": "higher_is_worse",
    "onsets_per_s": "higher_is_worse",
    "staccato_ratio": "higher_is_worse",
    "snr_db": "lower_is_worse",
}

# --------- Scoring base y ajuste híbrido ---------
def s_score(value, ref_med, ref_iqr, direction):
    if direction == "higher_is_worse":
        return max(0.0, (value - ref_med) / max(ref_iqr, 1e-9))
    else:  # lower_is_worse
        return max(0.0, (ref_med - value) / max(ref_iqr, 1e-9))

def in_iqr(val, q1, q3):
    return 1.0 if (val >= q1 and val <= q3) else 0.0

def beyond_hf(val, q1, q3, direction):
    if direction == "higher_is_worse":
        return 1.0 if val > q3 else 0.0
    else:  # lower_is_worse
        return 1.0 if val < q1 else 0.0

def compute_scores(m, iqr_local, iqr_lf=None, iqr_hf=None,
                   lambda_hf=0.08, mu_hf=0.08, lambda_lf=0.06, mu_lf=0.06):
    # 1) s-scores vs referencia local
    s = {}
    for key, dire in DIRECTIONS.items():
        if key not in iqr_local:
            s[key] = 0.0
            continue
        med = iqr_local[key]["median"]; iqr = iqr_local[key]["iqr"]
        val = m.get(key, None)
        if val is None or np.isnan(val):
            s[key] = 0.0
        else:
            s[key] = float(np.clip(s_score(val, med, iqr, dire), 0.0, 3.0))

    # 2) V/A base
    zA = (0.20*s["rms_mean"] + 0.18*s["onsets_per_s"] + 0.15*s["bpm"]
          + 0.12*s["spec_centroid_hz"] + 0.10*s["zcr_mean"]
          + 0.08*s["spec_bandwidth_hz"] + 0.07*s["ratio_high_low"]
          + 0.05*s["spectral_tilt_db_per_oct"] + 0.05*s["snr_db"])
    A = logistic(zA)

    zV = -(0.20*s["roughness"] + 0.15*s["spec_centroid_hz"]
           + 0.12*s["ratio_high_low"] + 0.12*s["zcr_mean"]
           + 0.10*s["rms_mean"] + 0.08*s["spec_bandwidth_hz"]
           + 0.08*s["spectral_tilt_db_per_oct"] + 0.10*s["hnr_mean_db"])
    V = logistic(zV)

    # 3) Ajuste híbrido con LF/HF
    reward_lf = 0.0; penalty_hf = 0.0; n_keys = 0
    for key, dire in DIRECTIONS.items():
        val = m.get(key, None)
        if val is None or np.isnan(val): 
            continue
        n_keys += 1
        if iqr_lf and (key in iqr_lf):
            reward_lf += in_iqr(val, iqr_lf[key]["q1"], iqr_lf[key]["q3"])
        if iqr_hf and (key in iqr_hf):
            penalty_hf += beyond_hf(val, iqr_hf[key]["q1"], iqr_hf[key]["q3"], dire)
    if n_keys > 0:
        reward_lf /= n_keys
        penalty_hf /= n_keys

    A_adj = float(np.clip(A + lambda_hf*penalty_hf - lambda_lf*reward_lf, 0.0, 1.0))
    V_adj = float(np.clip(V - mu_hf*penalty_hf + mu_lf*reward_lf, 0.0, 1.0))

    return V, A, V_adj, A_adj, s, reward_lf, penalty_hf

# ------------- Main CLI -------------
def main():
    ap = argparse.ArgumentParser(description="Pipeline V-A con normalización local y ajuste LF/HF opcional.")
    ap.add_argument("--iqr_local", type=str, required=True, help="CSV IQR de referencia local (pastoreo).")
    ap.add_argument("--iqr_lf", type=str, default=None, help="CSV IQR BovineTalk LF (opcional).")
    ap.add_argument("--iqr_hf", type=str, default=None, help="CSV IQR BovineTalk HF (opcional).")
    ap.add_argument("--dir", type=str, default=None, help="Carpeta con audios.")
    ap.add_argument("--out", type=str, default="valence_arousal_scores_hybrid.csv", help="CSV de salida.")
    ap.add_argument("inputs", nargs="*", help="Archivos individuales.")
    args = ap.parse_args()

    audio_files = []
    if args.dir:
        for ext in ("*.wav","*.mp3","*.flac","*.ogg","*.m4a"):
            audio_files.extend(glob.glob(os.path.join(args.dir, ext)))
    audio_files.extend(args.inputs)
    audio_files = sorted(list(dict.fromkeys(audio_files)))
    if not audio_files:
        print("No se encontraron audios.")
        return

    iqr_local = load_iqr_csv(args.iqr_local)
    iqr_lf = load_iqr_csv(args.iqr_lf) if args.iqr_lf else None
    iqr_hf = load_iqr_csv(args.iqr_hf) if args.iqr_hf else None

    rows = []
    for path in audio_files:
        try:
            m = compute_metrics(path)
            V, A, V_adj, A_adj, s, reward_lf, penalty_hf = compute_scores(m, iqr_local, iqr_lf, iqr_hf)
            m.update({
                "V_base": V, "A_base": A,
                "V_adj": V_adj, "A_adj": A_adj,
                "reward_lf": reward_lf, "penalty_hf": penalty_hf
            })
            rows.append(m)
            print(f"[OK] {os.path.basename(path)} -> V={V_adj:.3f}, A={A_adj:.3f} (base V={V:.3f}, A={A:.3f})")
        except Exception as e:
            print(f"[ERR] {path}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[DONE] Guardado en: {args.out}")

if __name__ == "__main__":
    main()

