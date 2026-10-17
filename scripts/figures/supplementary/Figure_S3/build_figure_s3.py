"""Build the  window-level acoustic-information profile figure (S3)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DPI = 300
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 150.0
SOLID_LINE_PT = 2.0
METRICS = [
    ("ACI", "ACI-derived spectral–temporal variability", "ACI-derived value (unitless)"),
    ("NDSI", "Normalized spectral-band difference\n(0–2 vs 2–8 kHz)", "Normalized difference (unitless)"),
    ("BI", "2–8 kHz spectral magnitude", "Summed STFT magnitude (a.u.)"),
    ("AEI", "Gini-based spectral-energy inequality", "Gini coefficient (unitless)"),
]

REPO_ROOT = Path(__file__).resolve().parents[4]


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()
COLORS = {"O1": "#D29528", "O2": "#397BA5"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--release", action="store_true", help="Write exports to the final-output directory")
    args = parser.parse_args()

    data = pd.read_csv(args.windows_csv)
    required = {"work", "file", "window_start_s", "window_end_s", *(m[0] for m in METRICS)}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    counts = data.groupby("work").size().to_dict()
    if counts != {"O1": 276, "O2": 276}:
        raise ValueError(f"Unexpected analytical-window counts: {counts}")
    if not np.allclose(data["window_end_s"] - data["window_start_s"], 5.0):
        raise ValueError("All observations must be complete 5-s windows")
    manifest_path = args.source_manifest.resolve()
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest["total_windows"] != 552 or source_manifest["windows_per_work"] != counts:
        raise ValueError("Source manifest does not match the window-level data")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4))
    axes = axes.flatten()
    rng = np.random.default_rng(20261006)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    for index, ((metric, title, ylabel), ax) in enumerate(zip(METRICS, axes)):
        for xpos, work in enumerate(("O1", "O2"), start=1):
            values = data.loc[data.work.eq(work), metric].to_numpy(dtype=float)
            jitter = rng.uniform(-0.11, 0.11, size=len(values))
            ax.scatter(
                xpos + jitter, values, s=5.5, alpha=0.32,
                color=COLORS[work], edgecolors="none", rasterized=True,
                zorder=2,
            )
            box = ax.boxplot(
                values, positions=[xpos], widths=0.34, whis=1.5,
                patch_artist=True, showfliers=False, manage_ticks=False,
                medianprops={"color": "#202020", "linewidth": SOLID_LINE_PT},
                whiskerprops={"color": "#333333", "linewidth": SOLID_LINE_PT},
                capprops={"color": "#333333", "linewidth": SOLID_LINE_PT},
                boxprops={"edgecolor": "#333333", "linewidth": SOLID_LINE_PT},
            )
            box["boxes"][0].set_facecolor(COLORS[work])
            box["boxes"][0].set_alpha(0.24)
        ax.set_title(f"({chr(65 + index)}) {title}", fontsize=8.1, pad=5)
        ax.set_xlim(0.55, 2.45)
        ax.set_xticks([1, 2], labels=["O1", "O2"])
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax.tick_params(axis="both", labelsize=8, width=SOLID_LINE_PT, length=3)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(SOLID_LINE_PT)
        ax.spines["bottom"].set_linewidth(SOLID_LINE_PT)
        ax.set_xlabel("FASE realization", fontsize=8, labelpad=2)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLORS[w], markersize=5.5, label=w)
        for w in ("O1", "O2")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=8, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.13, hspace=0.47, wspace=0.24)

    output_files = {
        "PNG": args.outdir / "Supplementary_Figure_S3.png",
        "JPEG": args.outdir / "Supplementary_Figure_S3.jpg",
        "TIFF": args.outdir / "Supplementary_Figure_S3.tif",
    }
    fig.savefig(output_files["PNG"], dpi=DPI, facecolor="white")
    fig.savefig(output_files["JPEG"], dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    fig.savefig(output_files["TIFF"], dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    for fmt, path in output_files.items():
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            if fmt == "TIFF":
                rgb.save(path, dpi=(DPI, DPI), compression="tiff_lzw")
            elif fmt == "JPEG":
                rgb.save(path, dpi=(DPI, DPI), quality=95, optimize=True)
            else:
                rgb.save(path, dpi=(DPI, DPI), optimize=True)

    statistics = {}
    for metric, _, _ in METRICS:
        statistics[metric] = {
            work: {
                "n_windows": int(data.work.eq(work).sum()),
                "median": float(data.loc[data.work.eq(work), metric].median()),
                "q1": float(data.loc[data.work.eq(work), metric].quantile(0.25)),
                "q3": float(data.loc[data.work.eq(work), metric].quantile(0.75)),
            }
            for work in ("O1", "O2")
        }
    manifest = {
        "figure": "Supplementary Figure S3",
        "build_status": "reproduced_from_packaged_data",
        "interpretation": "Descriptive window-level acoustic-information profiles; windows are not independent biological replicates.",
        "window_data": relative_to_repo(args.windows_csv),
        "window_data_sha256": sha256(args.windows_csv),
        "window_data_manifest": relative_to_repo(manifest_path),
        "window_data_manifest_sha256": sha256(manifest_path),
        "source_rebuild_script": relative_to_repo(Path(__file__).resolve().parent / "build_figure_s3_window_data.py"),
        "source_rebuild_script_sha256": sha256(Path(__file__).resolve().parent / "build_figure_s3_window_data.py"),
        "figure_builder": relative_to_repo(Path(__file__).resolve()),
        "figure_builder_sha256": sha256(Path(__file__).resolve()),
        "metric_panel_order": [metric for metric, _, _ in METRICS],
        "windows_per_work": counts,
        "window_duration_seconds": 5,
        "n_source_fragments_per_work": 13,
        "summary_statistics": statistics,
        "production": {
            "width_mm": FIGURE_WIDTH_MM,
            "height_mm": FIGURE_HEIGHT_MM,
            "dpi": DPI,
            "color_mode": "RGB",
            "tiff_compression": "LZW",
            "minimum_font_pt": 8,
            "minimum_solid_line_pt": SOLID_LINE_PT,
            "panel_labels": "(A)-(D)",
        },
        "exports": {},
    }
    for fmt, path in output_files.items():
        with Image.open(path) as image:
            manifest["exports"][path.name] = {
                "path": relative_to_repo(path),
                "sha256": sha256(path),
                "format": image.format,
                "mode": image.mode,
                "pixels": list(image.size),
                "dpi": [float(value) if value is not None else None for value in image.info.get("dpi", (None, None))],
                "compression": image.info.get("compression"),
            }
    provenance_path = args.outdir.parent / "02_PROVENANCE" / "FigureS3__provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exports": {fmt: str(path) for fmt, path in output_files.items()}, "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
