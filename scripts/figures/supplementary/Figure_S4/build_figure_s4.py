"""Build Supplementary Figure S4 from packaged OAAS plot coordinates."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

DPI = 300
WIDTH_MM = 180.0
HEIGHT_MM = 158.0
LINE_PT = 2.0
COLORS = {"fragment": "#2F9E69", "base": "#397BA5", "op": "#2457A6", "on": "#C64036", "link": "#9AA3AE"}


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
    package = root / "figures" / "supplementary" / "Figure_S4"
    outdir = package / ("00_FINAL" if args.release else "00_REBUILD")
    provdir = package / "02_PROVENANCE"
    outdir.mkdir(parents=True, exist_ok=True)
    public_data_dir = root / "data" / "derived" / "supplementary_figure_s4"
    public_data_dir.mkdir(parents=True, exist_ok=True)

    source_path = root / "data" / "derived" / "figure_09" / "Figure09_OAAS_plot_coordinates.csv"
    source = pd.read_csv(source_path)
    if len(source) != 40 or source.record_id.nunique() != 40:
        raise ValueError("Expected 36 configurations/fragments and four reference centroids")
    fragment_rows = source.loc[source.role.eq("fragment")]
    nodes = source.loc[source.role.ne("reference") & source.role.ne("fragment")].copy()
    nodes = nodes.rename(columns={"record_id": "id"})
    centroids = []
    for family in ("O1", "O2"):
        fragments = fragment_rows.loc[fragment_rows.family.eq(family)]
        if len(fragments) != 13:
            raise ValueError(f"Expected 13 fragments for {family}")
        mean = fragments[["PC1", "PC2"]].mean()
        centroids.append({"id": f"{family}_fragments_centroid", "family": family,
                          "role": "fragment_centroid", "branch": "fragment centroid", "variant": "",
                          "PC1": mean.PC1, "PC2": mean.PC2, "nearest_bovine_class": "GC"})
    nodes = pd.concat([pd.DataFrame(centroids), nodes], ignore_index=True)
    nodes["function"] = nodes.role
    nodes["source_record"] = "Figure09_OAAS_plot_coordinates.csv"
    source_csv = public_data_dir / "FigureS4_exact_node_coordinates.csv"
    nodes[["id", "family", "role", "branch", "variant", "PC1", "PC2", "nearest_bovine_class", "source_record"]].to_csv(source_csv, index=False, float_format="%.17g")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    xlim = (-3450, 450)
    ylim = (-3500, 400)
    for ax, family, panel in zip(axes[0], ("O1", "O2"), ("(A)", "(B)")):
        sub = nodes.loc[nodes.family.eq(family)]
        centroid = sub.loc[sub.role.eq("fragment_centroid")].iloc[0]
        long_nodes = sub.loc[sub.role.ne("fragment_centroid")]
        # Undirected dashed links compare analytical representations across scales.
        for _, node in long_nodes.iterrows():
            ax.plot([centroid.PC1, node.PC1], [centroid.PC2, node.PC2], color=COLORS["link"],
                    linestyle=(0, (4, 3)), linewidth=LINE_PT, zorder=1)
        ax.scatter(centroid.PC1, centroid.PC2, s=85, marker="o", color=COLORS["fragment"],
                   edgecolors="white", linewidths=LINE_PT, zorder=4)
        draw = (("positive", "op", "s", 90), ("negative", "on", "D", 90))
        for role, color_key, marker, size in draw:
            part = long_nodes.loc[long_nodes.role.eq(role)]
            for _, node in part.iterrows():
                face = COLORS[color_key] if node.variant == "a" else "white"
                ax.scatter([node.PC1], [node.PC2], s=size, marker=marker, facecolors=face,
                           edgecolors=COLORS[color_key], linewidths=LINE_PT, zorder=3)
        # Base and OP configurations can nearly coincide; the open circle preserves
        # the exact Base point while allowing the underlying OP marker to remain visible.
        base = long_nodes.loc[long_nodes.role.eq("base")]
        ax.scatter(base.PC1, base.PC2, s=98, marker="o", facecolors="white",
                   edgecolors=COLORS["base"], linewidths=LINE_PT, zorder=4)

        ax.set_title(f"{panel} {family} FASE family", fontsize=9, pad=7)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xticks([-3000, -2000, -1000, 0])
        ax.set_yticks([-3000, -2000, -1000, 0])
        ax.tick_params(width=LINE_PT, length=4, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(LINE_PT)
        ax.grid(False)

    for ax, family, panel in zip(axes[1], ("O1", "O2"), ("(C)", "(D)")):
        long_nodes = nodes.loc[nodes.family.eq(family) & nodes.role.ne("fragment_centroid")]
        draw = (("positive", "op", "s", 90), ("negative", "on", "D", 90))
        for role, color_key, marker, size in draw:
            part = long_nodes.loc[long_nodes.role.eq(role)]
            for _, node in part.iterrows():
                face = COLORS[color_key] if node.variant == "a" else "white"
                ax.scatter([node.PC1], [node.PC2], s=size, marker=marker, facecolors=face,
                           edgecolors=COLORS[color_key], linewidths=LINE_PT, zorder=3)
        base = long_nodes.loc[long_nodes.role.eq("base")]
        ax.scatter(base.PC1, base.PC2, s=98, marker="o", facecolors="white",
                   edgecolors=COLORS["base"], linewidths=LINE_PT, zorder=4)
        if family == "O1":
            ax.set_xlim(-3310, -2550)
            ax.set_ylim(-3340, -2600)
        else:
            ax.set_xlim(-2660, -2617)
            ax.set_ylim(-2686, -2666)
        ax.set_title(f"{panel} {family} long-duration detail (family-specific zoom)", fontsize=8, pad=6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(width=LINE_PT, length=3, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(LINE_PT)

    legend = [
        Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["fragment"], markeredgecolor="white", markeredgewidth=LINE_PT, markersize=8, label="Fragment-level centroid"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["base"], markeredgewidth=LINE_PT, markersize=8, label="Base"),
        Line2D([], [], marker="s", color="none", markerfacecolor=COLORS["op"], markeredgecolor=COLORS["op"], markeredgewidth=LINE_PT, markersize=8, label="OP design branch"),
        Line2D([], [], marker="D", color="none", markerfacecolor=COLORS["on"], markeredgecolor=COLORS["on"], markeredgewidth=LINE_PT, markersize=8, label="ON design branch"),
        Line2D([], [], color=COLORS["link"], linewidth=LINE_PT, linestyle=(0, (4, 3)), label="Cross-scale representational link"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.015), columnspacing=1.5, handlelength=2.4)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.15, wspace=0.18, hspace=0.42)

    output_files = {
        "PNG": outdir / "Supplementary_Figure_S4.png",
        "JPEG": outdir / "Supplementary_Figure_S4.jpg",
        "TIFF": outdir / "Supplementary_Figure_S4.tif",
    }
    fig.savefig(output_files["PNG"], dpi=DPI, facecolor="white")
    fig.savefig(output_files["JPEG"], dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    fig.savefig(output_files["TIFF"], dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    for fmt, path in output_files.items():
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if fmt == "TIFF":
                rgb.save(path, dpi=(DPI, DPI), compression="tiff_lzw")
            elif fmt == "JPEG":
                rgb.save(path, dpi=(DPI, DPI), quality=95, optimize=True)
            else:
                rgb.save(path, dpi=(DPI, DPI), optimize=True)

    manifest = {
        "figure": "Supplementary Figure S4",
        "build_status": "reproduced_from_packaged_data",
        "coordinate_source": "Compact Figure 9 plot-coordinate dataset; no audio files or larger external tables are required.",
        "interpretation": "Cross-scale representational links among fragment centroids, Base, and controlled OP/ON design branches; lines are undirected and not trajectories.",
        "source_coordinates": str(source_csv.relative_to(root).as_posix()),
        "source_coordinates_sha256": sha256(source_csv),
        "coordinate_sources": {str(source_path.relative_to(root).as_posix()): sha256(source_path)},
        "node_count": int(len(nodes)),
        "families": {f: {"fragment_centroids": 1, "base": 1, "op_variants": 2, "on_variants": 2} for f in ("O1", "O2")},
        "no_coordinate_jitter": True,
        "cross_scale_links_per_family": 5,
        "production": {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "dpi": DPI, "color_mode": "RGB", "tiff_compression": "LZW", "minimum_font_pt": 8, "minimum_line_pt": LINE_PT, "panel_labels": "(A)-(D)"},
        "exports": {},
    }
    for fmt, path in output_files.items():
        with Image.open(path) as image:
            manifest["exports"][path.name] = {"path": str(path.relative_to(root).as_posix()), "sha256": sha256(path), "format": image.format,
                "mode": image.mode, "pixels": list(image.size), "dpi": [float(v) if v is not None else None for v in image.info.get("dpi", (None, None))],
                "compression": image.info.get("compression")}
    (provdir / "FigureS4__provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exports": {k: str(v) for k, v in output_files.items()}, "source_data": str(source_csv)}, indent=2))


if __name__ == "__main__":
    main()
