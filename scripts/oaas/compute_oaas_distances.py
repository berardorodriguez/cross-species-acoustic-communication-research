from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compute distances to bovine OAAS centroids.")
    p.add_argument("--oaas_dir", required=True, help="OAAS output directory, e.g. .\\OAAS_MAIN")
    return p.parse_args()


def euclidean_to_centroid(row, centroid, pc_cols):
    v = row[pc_cols].to_numpy(dtype=float)
    c = centroid[pc_cols].to_numpy(dtype=float)
    return float(np.linalg.norm(v - c))


def main():
    args = parse_args()
    oaas_dir = Path(args.oaas_dir)

    bovine_path = oaas_dir / "OAAS_bovine_projection.csv"
    other_path = oaas_dir / "OAAS_other_projection.csv"
    centroids_path = oaas_dir / "bovine_subdomain_centroids.csv"

    bov = pd.read_csv(bovine_path)
    oth = pd.read_csv(other_path)
    cent = pd.read_csv(centroids_path)

    pc_cols = [c for c in bov.columns if c.startswith("PC")]
    if not pc_cols:
        raise ValueError("No PC columns found.")

    centroid_map = {row["subdomain"]: row for _, row in cent.iterrows()}

    target_classes = ["gc", "lfc", "hfc", "psc"]
    missing = [c for c in target_classes if c not in centroid_map]
    if missing:
        raise ValueError(f"Missing bovine centroids for: {missing}")

    # distancias para otros
    for cls in target_classes:
        oth[f"dist_to_{cls.upper()}"] = oth.apply(
            lambda r: euclidean_to_centroid(r, centroid_map[cls], pc_cols), axis=1
        )

    dist_cols = [f"dist_to_{cls.upper()}" for cls in target_classes]
    oth["nearest_bovine_class"] = oth[dist_cols].idxmin(axis=1).str.replace("dist_to_", "", regex=False)

    # también para bovino, útil como referencia interna
    for cls in target_classes:
        bov[f"dist_to_{cls.upper()}"] = bov.apply(
            lambda r: euclidean_to_centroid(r, centroid_map[cls], pc_cols), axis=1
        )

    bov["nearest_bovine_class"] = bov[dist_cols].idxmin(axis=1).str.replace("dist_to_", "", regex=False)

    # guardar archivo por archivo
    bov.to_csv(oaas_dir / "OAAS_bovine_with_distances.csv", index=False)
    oth.to_csv(oaas_dir / "OAAS_other_with_distances.csv", index=False)

    # resumen por domain
    if "domain" in oth.columns:
        summary_domain = (
            oth.groupby("domain")[dist_cols]
            .mean()
            .reset_index()
        )
        counts = oth.groupby("domain").size().reset_index(name="n")
        summary_domain = summary_domain.merge(counts, on="domain", how="left")
        summary_domain.to_csv(oaas_dir / "OAAS_summary_by_domain.csv", index=False)

        nearest_domain = (
            oth.groupby(["domain", "nearest_bovine_class"])
            .size()
            .reset_index(name="n")
        )
        nearest_domain.to_csv(oaas_dir / "OAAS_nearest_class_by_domain.csv", index=False)

    # resumen por tradition
    if "tradition" in oth.columns:
        summary_tradition = (
            oth.groupby("tradition")[dist_cols]
            .mean()
            .reset_index()
        )
        counts = oth.groupby("tradition").size().reset_index(name="n")
        summary_tradition = summary_tradition.merge(counts, on="tradition", how="left")
        summary_tradition.to_csv(oaas_dir / "OAAS_summary_by_tradition.csv", index=False)

        nearest_tradition = (
            oth.groupby(["tradition", "nearest_bovine_class"])
            .size()
            .reset_index(name="n")
        )
        nearest_tradition.to_csv(oaas_dir / "OAAS_nearest_class_by_tradition.csv", index=False)

    print("[OK] Distances computed")
    print(f"[OK] Wrote: {oaas_dir / 'OAAS_bovine_with_distances.csv'}")
    print(f"[OK] Wrote: {oaas_dir / 'OAAS_other_with_distances.csv'}")
    if "domain" in oth.columns:
        print(f"[OK] Wrote: {oaas_dir / 'OAAS_summary_by_domain.csv'}")
        print(f"[OK] Wrote: {oaas_dir / 'OAAS_nearest_class_by_domain.csv'}")
    if "tradition" in oth.columns:
        print(f"[OK] Wrote: {oaas_dir / 'OAAS_summary_by_tradition.csv'}")
        print(f"[OK] Wrote: {oaas_dir / 'OAAS_nearest_class_by_tradition.csv'}")


if __name__ == "__main__":
    main()