"""Build  Supplementary Figure S5 from preserved Figure 8 OAAS outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

DPI = 300
WIDTH_MM = 180.0
HEIGHT_MM = 165.0
LINE_PT = 2.0
COLORS = {"O1": "#7040A0", "O2": "#138B52", "stem": "#666666", "GC": "#333333", "HFC": "#777777", "link": "#A7ADB4"}
ORDER = ["O1_full", "O1_noVox", "O1_vox_stem", "O2_full", "O2_noVox"]
LABELS = {"O1_full": "O1 full", "O1_noVox": "O1 no-vocal", "O1_vox_stem": "Isolated bovine vocal stem", "O2_full": "O2 full", "O2_noVox": "O2 no-vocal"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true", help="Write exports to the final-output directory")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    package = root / "figures" / "supplementary" / "Figure_S5"
    input_path = root / "data" / "derived" / "supplementary_figure_s5" / "FigureS5_configuration_coordinates_and_distances.csv"
    outdir = package / ("00_FINAL" if args.release else "00_REBUILD")
    provdir = package / "02_PROVENANCE"
    outdir.mkdir(parents=True, exist_ok=True)
    provdir.mkdir(parents=True, exist_ok=True)

    plot = pd.read_csv(input_path).set_index("version")
    distances = plot
    if set(plot.index) != set(ORDER):
        raise ValueError("Expected the five packaged Obonuco configurations")
    for col in ("PCA1", "PCA2", "PCA3", "dist_to_GC", "dist_to_HFC"):
        if col not in plot:
            raise ValueError(f"Missing packaged S5 distance reference: {col}")
    data = plot.loc[ORDER, ["configuration", "family", "PCA1", "PCA2", "PCA3", "dist_to_GC", "dist_to_HFC"]]
    data = data.reset_index(names="version")
    source_csv = input_path

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.text(-0.095, 1.08, "(A)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    # Pair links indicate representational displacement between full and no-vocal mixes.
    for full, novox in (("O1_full", "O1_noVox"), ("O2_full", "O2_noVox")):
        a, b = plot.loc[full], plot.loc[novox]
        ax.plot([a.PCA1, b.PCA1], [a.PCA2, b.PCA2], color=COLORS["link"], linewidth=LINE_PT, zorder=1)
    for version in ORDER:
        row = plot.loc[version]
        family = "stem" if version.endswith("vox_stem") else version[:2]
        marker = "D" if family == "stem" else ("o" if version.endswith("full") else "s")
        ax.scatter(row.PCA1, row.PCA2, s=80, marker=marker, color=COLORS[family], edgecolors="white", linewidths=LINE_PT, zorder=3)
    # Direct labels keep the shared stem separate while the inset resolves the close mixes.
    ax.annotate("Isolated bovine vocal stem", (plot.loc["O1_vox_stem", "PCA1"], plot.loc["O1_vox_stem", "PCA2"]),
                xytext=(-8, 13), textcoords="offset points", ha="right", fontsize=8,
                arrowprops={"arrowstyle": "-", "color": COLORS["stem"], "lw": LINE_PT})
    ax.set_xlim(0, 25)
    ax.set_ylim(-14.5, -8.0)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.tick_params(width=LINE_PT, length=4, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_PT)
    inset = ax.inset_axes([0.61, 0.48, 0.34, 0.43])
    for full, novox in (("O1_full", "O1_noVox"), ("O2_full", "O2_noVox")):
        a, b = plot.loc[full], plot.loc[novox]
        inset.plot([a.PCA1, b.PCA1], [a.PCA2, b.PCA2], color=COLORS["link"], linewidth=LINE_PT, zorder=1)
    for version in ("O1_full", "O1_noVox", "O2_full", "O2_noVox"):
        row = plot.loc[version]
        inset.scatter(row.PCA1, row.PCA2, s=75, marker="o" if version.endswith("full") else "s",
                      color=COLORS[version[:2]], edgecolors="white", linewidths=LINE_PT, zorder=3)
    label_positions = {"O1_full": (3.98, -11.0), "O1_noVox": (3.03, -9.45), "O2_full": (3.03, -8.95), "O2_noVox": (3.03, -9.95)}
    for version, position in label_positions.items():
        row = plot.loc[version]
        inset.annotate(LABELS[version], (row.PCA1, row.PCA2), xytext=position, textcoords="data", fontsize=8,
                       arrowprops={"arrowstyle": "-", "color": "#555555", "lw": LINE_PT})
    inset.set_xlim(2.35, 4.55)
    inset.set_ylim(-11.55, -8.55)
    inset.set_xlabel("PCA1", fontsize=8, labelpad=1)
    inset.set_ylabel("PCA2", fontsize=8, labelpad=1)
    inset.tick_params(width=LINE_PT, length=3, labelsize=8)
    for spine in inset.spines.values():
        spine.set_linewidth(LINE_PT)

    bx = fig.add_subplot(gs[1])
    bx.text(-0.095, 1.12, "(B)", transform=bx.transAxes, fontsize=12, fontweight="bold", va="top")
    y = np.arange(len(ORDER))
    # Offset references vertically so both exact 3D distances and their labels remain legible.
    for idx, version in enumerate(ORDER):
        family = "stem" if version.endswith("vox_stem") else version[:2]
        row = distances.loc[version]
        bx.plot([0, row.dist_to_GC], [idx - 0.11, idx - 0.11], color=COLORS[family], linewidth=LINE_PT, alpha=0.55, zorder=1)
        bx.plot([0, row.dist_to_HFC], [idx + 0.11, idx + 0.11], color=COLORS[family], linewidth=LINE_PT, alpha=0.28, linestyle=(0, (4, 3)), zorder=1)
        bx.scatter(row.dist_to_GC, idx - 0.11, marker="o", s=55, color=COLORS[family], edgecolors="white", linewidths=LINE_PT, zorder=3)
        bx.scatter(row.dist_to_HFC, idx + 0.11, marker="D", s=55, color=COLORS[family], edgecolors="white", linewidths=LINE_PT, zorder=3)
        bx.text(row.dist_to_GC + 0.35, idx - 0.11, f"{row.dist_to_GC:.2f}", va="center", fontsize=8)
        bx.text(row.dist_to_HFC + 0.35, idx + 0.11, f"{row.dist_to_HFC:.2f}", va="center", fontsize=8)
    bx.set_yticks(y, ["Isolated vocal stem" if v.endswith("vox_stem") else LABELS[v] for v in ORDER])
    bx.invert_yaxis()
    bx.set_xlim(0, 31)
    bx.set_xlabel("Distance to bovine reference centroid (3D PCA)")
    bx.tick_params(axis="both", width=LINE_PT, length=4, labelsize=8)
    for spine in bx.spines.values():
        spine.set_linewidth(LINE_PT)
    handles = [Line2D([], [], marker="o", color="#444444", markerfacecolor="#444444", markersize=7, label="GC reference"),
               Line2D([], [], marker="D", color="#777777", markerfacecolor="#777777", markersize=7, label="HFC reference")]
    bx.legend(handles=handles, loc="lower right", frameon=False, ncol=2, fontsize=8, bbox_to_anchor=(1, 1.02))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.94, bottom=0.10)

    outputs = {"PNG": outdir / "Supplementary_Figure_S5.png",
               "JPEG": outdir / "Supplementary_Figure_S5.jpg",
               "TIFF": outdir / "Supplementary_Figure_S5.tif"}
    for fmt, path in outputs.items():
        if fmt == "JPEG":
            fig.savefig(path, dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
        elif fmt == "TIFF":
            fig.savefig(path, dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    for fmt, path in outputs.items():
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if fmt == "TIFF":
                rgb.save(path, dpi=(DPI, DPI), compression="tiff_lzw")
            elif fmt == "JPEG":
                rgb.save(path, dpi=(DPI, DPI), quality=95, optimize=True)
            else:
                rgb.save(path, dpi=(DPI, DPI), optimize=True)
    manifest = {
        "figure": "Supplementary Figure S5", "build_status": "reproduced_from_packaged_data",
        "interpretation": "Representational comparison among full, no-vocal, and isolated-vocal Obonuco configurations; pair links show representational displacement and are not temporal trajectories.",
        "source_data": str(source_csv.relative_to(root).as_posix()), "source_data_sha256": sha256(source_csv),
        "source_tables": {str(input_path.relative_to(root).as_posix()): sha256(input_path)},
        "external_reference_attribution": "HFC reference descriptors originate from the BovineTalk dataset (Gavojdian et al., 2024; doi:10.3389/fvets.2024.1357109). This table contains only study-derived PCA coordinates and distances, not recordings or source-level descriptors.",
        "coordinates_match_preserved_distance_table": True, "distances": "Saved full 3D PCA distances to GC and HFC reference centroids; no recalculation or PCA refit.",
        "configurations": ORDER, "production": {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "dpi": DPI, "color_mode": "RGB", "tiff_compression": "LZW", "minimum_font_pt": 8, "minimum_line_pt": LINE_PT, "panel_labels": "(A)-(B)"}, "exports": {}
    }
    for fmt, path in outputs.items():
        with Image.open(path) as image:
            manifest["exports"][path.name] = {"path": str(path.relative_to(root).as_posix()), "sha256": sha256(path), "format": image.format,
                "mode": image.mode, "pixels": list(image.size), "dpi": [float(v) if v is not None else None for v in image.info.get("dpi", (None, None))], "compression": image.info.get("compression")}
    (provdir / "FigureS5__provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exports": {k: str(v) for k, v in outputs.items()}, "source_data": str(source_csv)}, indent=2))


if __name__ == "__main__":
    main()
