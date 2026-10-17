from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


META_COLS = {
    "id",
    "file_name",
    "relative_path",
    "domain",
    "subdomain",
    "tradition",
    "function",
    "signal_type",
    "recording_type",
    "processing_level",
    "analysis_unit",
    "certainty",
    "include",
    "notes",
    "original_name",
    "analysis_name",
    "relative_path_original",
    "relative_path_processed",
    "resolved_audio_path",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bovine-centred OAAS from feature CSV.")
    p.add_argument("--features_csv", required=True, help="Input features CSV")
    p.add_argument("--output_dir", required=True, help="Output folder")
    p.add_argument("--anchor_domain", default="bovine_core", help="Anchor domain for PCA fitting")
    p.add_argument("--n_components", type=int, default=10, help="Max PCA components")
    p.add_argument(
        "--min_non_na_ratio",
        type=float,
        default=0.80,
        help="Keep feature only if bovine has at least this non-NA ratio",
    )
    return p.parse_args()


def get_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def select_clean_features(
    df: pd.DataFrame,
    anchor_domain: str,
    min_non_na_ratio: float,
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    if "domain" not in df.columns:
        raise ValueError("Input CSV must contain a 'domain' column.")

    bov = df[df["domain"] == anchor_domain].copy()
    if bov.empty:
        raise ValueError(f"No rows found for anchor domain: {anchor_domain}")

    numeric_cols = get_numeric_feature_cols(df)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")

    keep_cols = []
    bov_medians = {}

    for c in numeric_cols:
        non_na_ratio = bov[c].notna().mean()
        if non_na_ratio < min_non_na_ratio:
            continue

        med = bov[c].median()
        if pd.isna(med):
            continue

        bov_col = bov[c].fillna(med)
        if np.isclose(bov_col.std(ddof=0), 0.0):
            continue

        keep_cols.append(c)
        bov_medians[c] = med

    if not keep_cols:
        raise ValueError("No usable feature columns remained after filtering.")

    medians = pd.Series(bov_medians)
    return bov, keep_cols, medians


def build_feature_matrices(
    df: pd.DataFrame,
    keep_cols: List[str],
    medians: pd.Series,
    anchor_domain: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df2 = df.copy()

    for c in keep_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2[c] = df2[c].fillna(medians[c])

    bovine_df = df2[df2["domain"] == anchor_domain].copy()
    other_df = df2[df2["domain"] != anchor_domain].copy()

    X_bovine = bovine_df[keep_cols].to_numpy(dtype=float)
    X_other = other_df[keep_cols].to_numpy(dtype=float) if not other_df.empty else np.empty((0, len(keep_cols)))

    return bovine_df, other_df, X_bovine, X_other


def main() -> None:
    args = parse_args()

    features_csv = Path(args.features_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)

    bov_raw, keep_cols, medians = select_clean_features(
        df=df,
        anchor_domain=args.anchor_domain,
        min_non_na_ratio=args.min_non_na_ratio,
    )

    bovine_df, other_df, X_bovine, X_other = build_feature_matrices(
        df=df,
        keep_cols=keep_cols,
        medians=medians,
        anchor_domain=args.anchor_domain,
    )

    scaler = StandardScaler()
    X_bovine_scaled = scaler.fit_transform(X_bovine)
    X_other_scaled = scaler.transform(X_other) if X_other.shape[0] else X_other

    n_components = min(args.n_components, X_bovine_scaled.shape[0], X_bovine_scaled.shape[1])
    if n_components < 2:
        raise ValueError("Not enough bovine data/features to compute PCA.")

    pca = PCA(n_components=n_components)
    X_bovine_pca = pca.fit_transform(X_bovine_scaled)
    X_other_pca = pca.transform(X_other_scaled) if X_other_scaled.shape[0] else X_other_scaled

    bovine_out = bovine_df.copy()
    other_out = other_df.copy()

    for i in range(n_components):
        pc = f"PC{i+1}"
        bovine_out[pc] = X_bovine_pca[:, i]
        if X_other_pca.shape[0]:
            other_out[pc] = X_other_pca[:, i]

    # Save projections
    bovine_out.to_csv(output_dir / "OAAS_bovine_projection.csv", index=False)
    other_out.to_csv(output_dir / "OAAS_other_projection.csv", index=False)

    # Save explained variance
    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(output_dir / "pca_variance.csv", index=False)

    # Save loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=keep_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "feature"})
    loadings.to_csv(output_dir / "pca_loadings.csv", index=False)

    # Save bovine class centroids
    if "subdomain" in bovine_out.columns:
        pc_cols = [f"PC{i+1}" for i in range(n_components)]
        centroids = bovine_out.groupby("subdomain")[pc_cols].mean().reset_index()
        counts = bovine_out.groupby("subdomain").size().reset_index(name="n")
        centroids = centroids.merge(counts, on="subdomain", how="left")
        centroids.to_csv(output_dir / "bovine_subdomain_centroids.csv", index=False)

    # Save feature manifest
    pd.DataFrame({
        "feature": keep_cols,
        "bovine_median_for_imputation": [medians[c] for c in keep_cols],
    }).to_csv(output_dir / "feature_manifest.csv", index=False)

    # Save models
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(pca, output_dir / "pca.joblib")

    print("[OK] OAAS construido correctamente")
    print(f"[OK] Anchor domain: {args.anchor_domain}")
    print(f"[OK] Bovine rows:   {len(bovine_out)}")
    print(f"[OK] Other rows:    {len(other_out)}")
    print(f"[OK] Features used: {len(keep_cols)}")
    print("[OK] Varianza acumulada primeras PCs:")
    for i, v in enumerate(np.cumsum(pca.explained_variance_ratio_)[:5], start=1):
        print(f"  - PC1..PC{i}: {v:.4f}")


if __name__ == "__main__":
    main()
