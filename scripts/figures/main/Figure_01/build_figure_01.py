"""Rebuild Figure 1 panels A–C from the packaged derived plotting data.

No source audio is opened or required by this renderer. Representative
spectrogram coordinates are stored in data/derived/figure_01/*.npz; panel C
uses the packaged four-category summary table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


MM_PER_INCH = 25.4
CATEGORIES = ["GC", "LFC", "PSC", "HFC"]
CATEGORY_TITLES = {
    "GC": "Grazing call (GC)",
    "LFC": "Low-frequency call (LFC)",
    "PSC": "Pre-slaughter call (PSC)",
    "HFC": "High-frequency call (HFC)",
}
BANDS = ["Low (0–1 kHz)", "Mid (1–3 kHz)", "High (3–8 kHz)"]
BAND_COLORS = {
    BANDS[0]: "#1f77b4",
    BANDS[1]: "#2ca02c",
    BANDS[2]: "#ff7f0e",
}


def read_profiles(path: Path) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    profiles: dict[str, dict[str, list[tuple[float, float]]]] = {
        category: {band: [] for band in BANDS} for category in CATEGORIES
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            category, band = row["category"], row["band"]
            if category not in profiles or band not in profiles[category]:
                raise ValueError(f"Unexpected category or band in {path}: {row}")
            profiles[category][band].append(
                (float(row["normalized_time"]), float(row["relative_energy"]))
            )
    result = {}
    for category, bands in profiles.items():
        result[category] = {}
        for band, values in bands.items():
            if not values:
                raise ValueError(f"Missing panel C values: {category}, {band}")
            values.sort(key=lambda item: item[0])
            result[category][band] = (
                np.asarray([item[0] for item in values]),
                np.asarray([item[1] for item in values]),
            )
    return result


def style_axes(ax, spine_width: float = 2.0) -> None:
    ax.tick_params(axis="both", labelsize=8.5, width=2.0, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)


def build(data_dir: Path, output_dir: Path) -> None:
    data_dir = data_dir.resolve()
    profiles_path = data_dir / "Figure01_all_categories_band_profiles.csv"
    spectra_path = data_dir / "Figure01_representative_spectrograms.npz"
    for required in (profiles_path, spectra_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required derived-data file not found: {required}")
    profiles = read_profiles(profiles_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(spectra_path, allow_pickle=False) as stored:
        spectra = {category: stored[f"spec_{category}"].copy() for category in CATEGORIES}
        times = {category: stored[f"time_{category}"].copy() for category in CATEGORIES}
        frequencies = stored["frequency_hz"].copy()
    if any(spectra[category].shape != (len(frequencies), len(times[category])) for category in CATEGORIES):
        raise ValueError("Representative-spectrogram array dimensions do not match the stored axes")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(180.0 / MM_PER_INCH, 150.0 / MM_PER_INCH))
    outer = fig.add_gridspec(
        3, 1, left=0.075, right=0.925, top=0.985, bottom=0.155,
        height_ratios=[1.0, 1.45, 1.0], hspace=0.46,
    )
    global_vmin = min(float(spectra[c].min()) for c in CATEGORIES)
    global_vmax = max(float(spectra[c].max()) for c in CATEGORIES)

    # Panel A: representative spectrograms, in the approved horizontal order.
    grid_a = outer[0].subgridspec(2, 4, height_ratios=[0.30, 1], hspace=0.38, wspace=0.38)
    title_a = fig.add_subplot(grid_a[0, :])
    title_a.axis("off")
    title_a.text(0.0, 0.55, "(A)", fontsize=15, fontweight="bold", va="center")
    title_a.text(0.085, 0.55, "Time–frequency spectrograms", fontsize=9, fontweight="bold", va="center")
    axes_a = [fig.add_subplot(grid_a[1, index]) for index in range(4)]
    for index, (ax, category) in enumerate(zip(axes_a, CATEGORIES)):
        ax.pcolormesh(times[category], frequencies, spectra[category], shading="auto",
                      cmap="magma", vmin=global_vmin, vmax=global_vmax)
        ax.set_ylim(0, 3500)
        ax.set_title(CATEGORY_TITLES[category], pad=3, fontsize=8.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_yticks([0, 1000, 2000, 3000])
        ax.set_xlabel("Time (s)", fontsize=8.0, labelpad=1)
        ax.set_ylabel("Frequency (Hz)" if index == 0 else "", fontsize=8.0, labelpad=1)
        style_axes(ax)

    # Panel B: 3-D surfaces use the same derived matrix and level scale.
    grid_b = outer[1].subgridspec(
        2, 5, height_ratios=[0.27, 1], width_ratios=[1, 1, 1, 1, 0.07],
        hspace=0.34, wspace=0.30,
    )
    title_b = fig.add_subplot(grid_b[0, :])
    title_b.axis("off")
    title_b.text(0.0, 0.55, "(B)", fontsize=15, fontweight="bold", va="center")
    title_b.text(0.085, 0.55, "Three-dimensional time–frequency sound level", fontsize=9,
                 fontweight="bold", va="center")
    axes_b = [fig.add_subplot(grid_b[1, index], projection="3d") for index in range(4)]
    cax = fig.add_subplot(grid_b[1, 4])
    surf = None
    for index, (ax, category) in enumerate(zip(axes_b, CATEGORIES)):
        spec = spectra[category]
        f = frequencies[::3]
        t = times[category][::3]
        z = spec[::3, ::3]
        t_grid, f_grid = np.meshgrid(t, f)
        surf = ax.plot_surface(t_grid, f_grid, z, rstride=1, cstride=1, cmap="magma",
                               linewidth=0, antialiased=False, vmin=global_vmin, vmax=global_vmax)
        ax.set_title(category, pad=7, fontsize=8.5)
        ax.set_xlabel("Time (s)" if index in (1, 2) else "", labelpad=0, fontsize=8)
        ax.set_ylabel("Frequency (Hz)" if index == 0 else "", labelpad=0, fontsize=8)
        if index == 0:
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel("Level (dB)", labelpad=0, fontsize=8, rotation=90)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 3500)
        ax.set_zlim(global_vmin, global_vmax)
        ax.set_xticks([0, 0.75, 1.5])
        ax.set_yticks([0, 3000])
        ax.set_zticks([-80, 0])
        ax.view_init(elev=35, azim=-135)
        ax.set_box_aspect((1.4, 1.4, 1.0))
        ax.dist = 6.8
        ax.tick_params(labelsize=8.0, pad=0, width=2.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_linewidth(2.0)
            axis._axinfo["grid"]["linewidth"] = 2.0
    if surf is not None:
        colorbar = fig.colorbar(surf, cax=cax)
        colorbar.set_label("Sound level (dB)", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.outline.set_linewidth(2.0)

    # Panel C: the stored 300-frame summary values, without source reprocessing.
    grid_c = outer[2].subgridspec(2, 4, height_ratios=[0.30, 1], hspace=0.38, wspace=0.38)
    title_c = fig.add_subplot(grid_c[0, :])
    title_c.axis("off")
    title_c.text(0.0, 0.55, "(C)", fontsize=15, fontweight="bold", va="center")
    title_c.text(0.085, 0.55, "Relative energy profiles across normalized time", fontsize=9,
                 fontweight="bold", va="center")
    axes_c = [fig.add_subplot(grid_c[1, index]) for index in range(4)]
    for index, (ax, category) in enumerate(zip(axes_c, CATEGORIES)):
        for band in BANDS:
            x, y = profiles[category][band]
            ax.plot(x, y, color=BAND_COLORS[band], linewidth=2.0, label=band)
        ax.set_title(CATEGORY_TITLES[category], pad=3, fontsize=8.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("")
        ax.set_ylabel("Relative energy" if index == 0 else "", fontsize=8.0, labelpad=1)
        ax.grid(False)
        style_axes(ax)
    handles, labels = axes_c[0].get_legend_handles_labels()
    fig.text(0.50, 0.080, "Normalized time", ha="center", va="center", fontsize=8.5)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.015), ncol=3, frameon=False)

    png_path = output_dir / "Figure_01.png"
    tif_path = output_dir / "Figure_01.tif"
    jpg_path = output_dir / "Figure_01.jpg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(tif_path, dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(jpg_path, dpi=300,
                pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    rgb_path = output_dir / "Figure_01.rgb.tmp.tif"
    with Image.open(tif_path) as image:
        image.convert("RGB").save(rgb_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    rgb_path.replace(tif_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/derived/figure_01"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/figure_01"))
    args = parser.parse_args()
    build(args.data_dir, args.outdir)


if __name__ == "__main__":
    main()
