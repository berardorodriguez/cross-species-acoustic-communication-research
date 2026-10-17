"""Build Figure 11 from the minimal, package-local plotting data.

Panels A-C summarize 30 repeated group/session scans per condition. Panel D
plots the retained 0-22 RMSSD mean and sample SD by arm and time index. The
figure is descriptive; the script performs no inferential tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]
DATA = ROOT / "data/derived/figure_11"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/figure_11_rebuild",
        help="Directory for generated TIFF, JPEG, and PNG files.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ethogram = pd.read_csv(DATA / "Figure11_panelsA_B_scan_level_source.csv")
    locomotion = pd.read_csv(DATA / "Figure11_panelC_scan_level_source.csv")
    hrv_summary = pd.read_csv(DATA / "Figure11_panelD_mean_sd_by_arm_index.csv")
    ethogram["near_source_count_d1_d2"] = ethogram["d1"] + ethogram["d2"]

    condition_order = ["Control", "Pink noise", "Obonuco"]
    expected_n = {condition: 30 for condition in condition_order}
    for name, frame in (("ethogram", ethogram), ("locomotion", locomotion)):
        actual = frame.groupby("condition").size().to_dict()
        if actual != expected_n:
            raise ValueError(f"Unexpected {name} repeated-scan counts: {actual}")

    observed_animals = hrv_summary.groupby("arm")["n"].unique().to_dict()
    expected_animals = {"Control": np.array([6]), "Musica": np.array([7]), "Rosa": np.array([6])}
    if set(observed_animals) != set(expected_animals) or any(
        not np.array_equal(observed_animals[k], expected_animals[k]) for k in expected_animals
    ):
        raise ValueError(f"Unexpected HRV animal counts: {observed_animals}")
    if hrv_summary.groupby("arm")["relative_index_0based"].nunique().to_dict() != {
        "Control": 23,
        "Musica": 23,
        "Rosa": 23,
    }:
        raise ValueError("Expected 23 plotted records per HRV arm.")

    colors = {"Control": "#555555", "Pink noise": "#2E70B5", "Obonuco": "#7D3FA0"}
    styles = {"Control": "-", "Pink noise": "--", "Obonuco": ":"}
    fig = plt.figure(figsize=(180 / 25.4, 125 / 25.4), dpi=300, facecolor="white")
    bar_axes = [
        fig.add_axes([0.105, 0.67, 0.235, 0.22]),
        fig.add_axes([0.405, 0.67, 0.235, 0.22]),
        fig.add_axes([0.705, 0.67, 0.235, 0.22]),
    ]
    panel_specs = [
        ("A", "Near-source cattle\n(d1 + d2; 0-5 m)", "near_source_count_d1_d2", 22, [0, 5, 10, 15, 20], "Cattle count per scan"),
        ("B", "Fully oriented cattle\n(p1)", "p1", 4.5, [0, 1, 2, 3, 4], "Cattle count per scan"),
        ("C", "Walking cattle\n(30-s sample)", "Conteo_CAM", 9, [0, 2, 4, 6, 8], "Cattle count per sample"),
    ]
    for ax, (panel, title, variable, ymax, yticks, ylabel) in zip(bar_axes, panel_specs):
        heights, errors = [], []
        for condition in condition_order:
            source = locomotion if panel == "C" else ethogram
            values = source.loc[source["condition"] == condition, variable]
            heights.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)))
        x = np.arange(3)
        ax.bar(
            x,
            heights,
            yerr=errors,
            color=[colors[c] for c in condition_order],
            edgecolor="none",
            linewidth=0,
            width=0.68,
            error_kw={"elinewidth": 2.0, "capsize": 2.3, "capthick": 2.0, "ecolor": "#222222"},
            zorder=3,
        )
        ax.set_title(f"({panel}) {title}", loc="center", fontsize=8.2, fontweight="bold", pad=5, linespacing=1.05)
        ax.set_ylabel(ylabel, fontsize=8.0, labelpad=3)
        ax.set_xticks([])
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=8.0, width=2.0, length=2.5, pad=2)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(2.0)

    handles = [Line2D([0], [0], color=colors[c], linestyle=styles[c], lw=2.0, label=c) for c in condition_order]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 0.625), ncol=3, frameon=False, fontsize=8.5, handlelength=2.3, columnspacing=2.0)

    axd = fig.add_axes([0.12, 0.145, 0.85, 0.36])
    hrv_map = {"Control": "Control", "Rosa": "Pink noise", "Musica": "Obonuco"}
    for arm, condition in hrv_map.items():
        group = hrv_summary.loc[hrv_summary["arm"] == arm].sort_values("relative_index_0based")
        x = group["relative_index_0based"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        sd = group["sd"].to_numpy(dtype=float)
        axd.fill_between(x, np.maximum(0, mean - sd), mean + sd, color=colors[condition], alpha=0.16, linewidth=0)
        axd.plot(x, mean, color=colors[condition], linestyle=styles[condition], lw=2.0)
    axd.set_title("(D) RMSSD trajectories during the initial ~23-min exposure window", loc="left", fontsize=8.5, fontweight="bold", pad=5)
    axd.set_xlim(0, 23)
    axd.set_ylim(0, 45)
    axd.set_xticks([0, 5, 10, 15, 20])
    axd.set_yticks([0, 10, 20, 30, 40])
    axd.set_xlabel("Time (min)", fontsize=8.5, labelpad=3)
    axd.set_ylabel("RMSSD (ms)", fontsize=8.5, labelpad=3)
    axd.tick_params(axis="both", labelsize=8.0, width=2.0, length=3)
    for side in ("top", "right"):
        axd.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axd.spines[side].set_linewidth(2.0)

    fig.subplots_adjust(0, 0, 1, 1)
    outputs = {
        "tiff": args.output_dir / "Figure11__REBUILT.tif",
        "jpg": args.output_dir / "Figure11__REBUILT.jpg",
        "png": args.output_dir / "Figure11__REBUILT.png",
    }
    fig.savefig(outputs["tiff"], dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outputs["jpg"], dpi=300, format="jpg", pil_kwargs={"quality": 95, "subsampling": 0})
    fig.savefig(outputs["png"], dpi=300, format="png")
    plt.close(fig)

    for path in outputs.values():
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            dpi = image.info.get("dpi", (300, 300))
        if path.suffix.lower() == ".tif":
            rgb.save(path, format="TIFF", dpi=dpi, compression="tiff_lzw")
        elif path.suffix.lower() == ".jpg":
            rgb.save(path, format="JPEG", dpi=dpi, quality=95, subsampling=0)
        else:
            rgb.save(path, format="PNG", dpi=dpi, optimize=True)
    with Image.open(outputs["tiff"]) as image:
        if image.mode != "RGB" or image.size != (2125, 1476):
            raise ValueError(f"Unexpected export: {image.mode} {image.size}")
    print("Rebuilt Figure 11 from package-local plotting data:")
    for path in outputs.values():
        print(f"{path.name}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
