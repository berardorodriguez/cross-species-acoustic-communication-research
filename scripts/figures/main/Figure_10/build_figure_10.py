"""Build final  Figure 10 exports from preserved PCA-coordinate tables.

No feature extraction, PCA fit, coordinate modification, or scientific analysis
is performed here. The packaged numeric coordinates are plotted verbatim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "figures" / "main" / "Figure_10"
PROV = ROOT / "02_PROVENANCE"
SOURCE = REPO_ROOT / "data" / "derived" / "figure_10"
FINAL = ROOT / "00_FINAL"
DPI = 300
WIDTH_MM = 180
HEIGHT_MM = 160
PIXELS = (round(WIDTH_MM / 25.4 * DPI), round(HEIGHT_MM / 25.4 * DPI))

BLACK = "#202020"
GREEN = "#218c52"
RED = "#d93636"
GRAY = "#777777"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_and_check() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_xy = pd.read_csv(SOURCE / "macro_global_coordinates.csv")
    local_xy = pd.read_csv(SOURCE / "micro_local_coordinates.csv")
    local_var = pd.read_csv(SOURCE / "local_pca_variance.csv")
    expected = {
        "O1": {"O1", "OP1a", "OP1b", "ON1a", "ON1b"},
        "O2": {"O2", "OP2a", "OP2b", "ON2a", "ON2b"},
    }
    for table, xcol, ycol in [
        (global_xy, "PC1", "PC2"),
        (local_xy, "PC1_l", "PC2_l"),
    ]:
        assert set(table.family) == set(expected)
        for family, versions in expected.items():
            subset = table[table.family == family]
            assert set(subset.variant) == versions
            assert subset.variant.is_unique
        assert np.isfinite(table[[xcol, ycol]].to_numpy()).all()
    assert set(local_var.family) == set(expected)
    assert local_var.family.is_unique
    assert np.isfinite(local_var[["PC1_var", "PC2_var"]].to_numpy()).all()
    return global_xy, local_xy, local_var


def annotate(ax, label, position, color=BLACK, weight="normal"):
    ax.text(position[0], position[1], label, fontsize=8, color=color,
            fontweight=weight, ha="left", va="center", clip_on=False)


GLOBAL_OFFSETS = {
    "O1": {"O1": (-9.0, -4.6), "OP1a": (-7.15, -6.25), "OP1b": (-8.7, -9.25),
           "ON1a": (-3.8, -4.3), "ON1b": (-4.4, 13.7)},
    "O2": {"O2": (16.0, 1.5), "OP2a": (16.0, -0.9), "OP2b": (16.0, -2.5),
           "ON2a": (-2.5, 2.5), "ON2b": (-6.1, 13.0)},
}
LOCAL_OFFSETS = {
    "O1": {"O1": (2.2, 0.7), "OP1a": (0.5, 3.7), "OP1b": (3.2, 3.7),
           "ON1a": (-2.8, 3.4), "ON1b": (-6.8, -9.4)},
    "O2": {"O2": (-6.7, -1.2), "OP2a": (-7.15, 0.8), "OP2b": (-7.15, 2.5),
           "ON2a": (6.8, 3.0), "ON2b": (11.0, -5.5)},
}


def draw_family(ax, family, table, xcol, ycol, label_positions, title, xlab, ylab):
    subset = table[table.family == family].set_index("variant")
    base_id = family
    base = subset.loc[base_id]
    for branch, color, marker in [("OP", GREEN, "s"), ("ON", RED, "D")]:
        variants = sorted(v for v in subset.index if v.startswith(branch))
        for variant in variants:
            row = subset.loc[variant]
            # Each line directly relates a named variant to Base. Suffixes a/b
            # do not encode a plotted sequence or a temporal trajectory.
            ax.plot([base[xcol], row[xcol]], [base[ycol], row[ycol]],
                    color=GRAY, linewidth=2, alpha=0.75, zorder=1)

    ax.scatter(base[xcol], base[ycol], s=66, c=BLACK, marker="o", linewidths=2.0, zorder=3)
    annotate(ax, "Base", label_positions[base_id], weight="bold")
    for branch, color, marker in [("OP", GREEN, "s"), ("ON", RED, "D")]:
        for variant in sorted(v for v in subset.index if v.startswith(branch)):
            row = subset.loc[variant]
            ax.scatter(row[xcol], row[ycol], s=55, c=color, marker=marker, linewidths=2.0, zorder=3)
            annotate(ax, variant, label_positions[variant])

    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(axis="both", labelsize=8, width=2.0, length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.margins(x=0.18, y=0.22)


def main() -> None:
    global_xy, local_xy, local_var = read_and_check()
    fig, axes = plt.subplots(2, 2, figsize=(PIXELS[0] / DPI, PIXELS[1] / DPI))
    fig.patch.set_facecolor("white")

    global_variance = (0.487, 0.247)
    for i, family in enumerate(["O1", "O2"]):
        ax = axes[0, i]
        draw_family(
            ax, family, global_xy, "PC1", "PC2", GLOBAL_OFFSETS[family],
            f"{family} — global bovine-referenced projection",
            f"PC1 ({global_variance[0] * 100:.1f}%)",
            f"PC2 ({global_variance[1] * 100:.1f}%)",
        )
        fig.text(0.025 + i * 0.48, 0.952, f"({chr(65 + i)})",
                 fontsize=11, fontweight="bold", ha="left", va="top")

        var_row = local_var[local_var.family == family].iloc[0]
        ax = axes[1, i]
        draw_family(
            ax, family, local_xy, "PC1_l", "PC2_l", LOCAL_OFFSETS[family],
            f"{family} — independently fitted within-family PCA",
            f"PC1 ({var_row.PC1_var * 100:.1f}%)",
            f"PC2 ({var_row.PC2_var * 100:.1f}%)",
        )
        fig.text(0.025 + i * 0.48, 0.485, f"({chr(67 + i)})",
                 fontsize=11, fontweight="bold", ha="left", va="top")

    # Panels A and B depict the same global PCA space and share both limits.
    gx = global_xy.PC1.to_numpy()
    gy = global_xy.PC2.to_numpy()
    xpad = (gx.max() - gx.min()) * 0.11
    ypad = (gy.max() - gy.min()) * 0.12
    for ax in axes[0, :]:
        ax.set_xlim(gx.min() - xpad, gx.max() + xpad)
        ax.set_ylim(gy.min() - ypad, gy.max() + ypad)

    legend = [
        Line2D([], [], marker="o", linestyle="none", color=BLACK, markersize=7, markeredgewidth=2.0, label="Base"),
        Line2D([], [], marker="s", linestyle="none", color=GREEN, markersize=7, markeredgewidth=2.0, label="OP variant"),
        Line2D([], [], marker="D", linestyle="none", color=RED, markersize=7, markeredgewidth=2.0, label="ON variant"),
        Line2D([], [], color=GRAY, linewidth=2, label="Base-to-variant relation"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.12,
                        wspace=0.34, hspace=0.48)
    fig.canvas.draw()
    for ax in axes.flat:
        for spine in ax.spines.values():
            if spine.get_visible():
                assert spine.get_linewidth() >= 2.0
        for tickline in ax.get_xticklines() + ax.get_yticklines():
            if tickline.get_visible():
                assert tickline.get_markeredgewidth() >= 2.0
        for line in ax.lines:
            assert line.get_linewidth() >= 2.0
        for collection in ax.collections:
            assert np.all(np.asarray(collection.get_linewidths()) >= 2.0)
    assert all(max(line.get_linewidth(), line.get_markeredgewidth()) >= 2.0 for line in legend)
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text().strip():
            assert item.get_fontsize() >= 8, (item.get_text(), item.get_fontsize())
            box = item.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= PIXELS[0] and box.y1 <= PIXELS[1], (item.get_text(), box)

    exports = []
    for ext in ["png", "jpg", "tif"]:
        path = FINAL / f"Figure_10.{ext}"
        buffer = BytesIO()
        fig.savefig(buffer, dpi=DPI, facecolor="white", format=ext)
        buffer.seek(0)
        with Image.open(buffer) as im:
            rgb = im.convert("RGB")
            assert rgb.size == PIXELS
        image_format = {"png": "PNG", "jpg": "JPEG", "tif": "TIFF"}[ext]
        save_options = {"dpi": (DPI, DPI)}
        if ext == "jpg":
            save_options.update(quality=95, subsampling=0)
        rgb.save(path, format=image_format, **save_options)
        with Image.open(path) as im:
            assert im.size == PIXELS and im.mode == "RGB"
            assert max(abs(float(value) - DPI) for value in im.info["dpi"]) < 0.01
        exports.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    plt.close(fig)

    inputs = [
        "data/derived/figure_10/macro_global_coordinates.csv",
        "data/derived/figure_10/micro_local_coordinates.csv",
        "data/derived/figure_10/local_pca_variance.csv",
    ]
    metadata = {
        "figure": 10,
        "status": "GRAPHICAL PRODUCTION CLOSED / FRONTIERS FORMAT CHECK PASSED 2026-10-04",
        "new_scientific_analysis": False,
        "new_scientific_analysis": False,
        "coordinate_policy": "Plots preserved global and local PCA coordinates without refitting or point displacement.",
        "global_PC1_PC2_variance_percent": [48.7, 24.7],
        "global_variance_source": "Printed axis labels in the source figure.",
        "canvas_px": list(PIXELS),
        "nominal_mm": [WIDTH_MM, HEIGHT_MM],
        "dpi": DPI,
        "mode": "RGB",
        "minimum_text_pt": 8,
        "visible_connection_width_pt": 2,
        "visible_axes_spines_ticks_and_marker_edges_width_pt": 2,
        "format_adjustment_2026_10_04": "Axes, ticks, spines, and marker edges were raised to 2 pt; coordinates, labels, and figure content were unchanged.",
        "variant_link_policy": "Each variant is linked directly to its family Base; no a-to-b sequence or temporal trajectory is drawn.",
        "source_files": [{"file": p, "sha256": sha256(REPO_ROOT / p)} for p in inputs],
        "build_script_sha256": sha256(Path(__file__)),
        "exports": exports,
    }
    (PROV / "Figure_10_build_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Final Figure 10 exports built from preserved coordinate tables; no PCA was refit.")


if __name__ == "__main__":
    main()


