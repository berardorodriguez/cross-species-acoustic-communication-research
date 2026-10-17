from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "figures" / "main" / "Figure_12"
SOURCE = ROOT / "data" / "derived" / "figure_12" / "Figure12_animal_phase_means.csv"
OUTPUT = PACKAGE
FINAL = PACKAGE / "00_FINAL"
PROVENANCE = PACKAGE / "02_PROVENANCE"
SOURCE_COPY = PROVENANCE / "Figure12_animal_phase_means_source_verified_copy.csv"

SOURCE_COLUMNS = ["PRE60", "O1", "INTERMEDIATE", "O2"]
DISPLAY_LABELS = ["PRE", "O11", "Intervening\ninterval", "O12"]
EXPECTED_MEANS = np.array([2.1143, 2.1344, 2.1722, 2.2430])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    FINAL.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE)
    required = {"cow", *SOURCE_COLUMNS}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(data.columns))}")
    data = data[["cow", *SOURCE_COLUMNS]].copy()
    if len(data) != 7 or data["cow"].nunique() != 7:
        raise ValueError("Expected seven unique animal-level rows.")
    if data[SOURCE_COLUMNS].isna().any().any():
        raise ValueError("Missing animal phase means are not permitted.")
    if not np.isfinite(data[SOURCE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite animal phase means are not permitted.")

    means = data[SOURCE_COLUMNS].mean(axis=0).to_numpy(dtype=float)
    if not np.allclose(means, EXPECTED_MEANS, atol=0.00005, rtol=0):
        raise ValueError(f"Phase means disagree with the manuscript record: {means}")

    data.to_csv(SOURCE_COPY, index=False, float_format="%.12g")
    x = np.arange(len(SOURCE_COLUMNS))
    values = data[SOURCE_COLUMNS].to_numpy(dtype=float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 2.0,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(2126 / 300, 1476 / 300), dpi=300)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.27, top=0.96)

    for row in values:
        ax.plot(x, row, color="#8A9299", linewidth=2.0, alpha=0.62, zorder=1)
        ax.scatter(x, row, s=22, color="#8A9299", edgecolor="#8A9299", linewidth=2.0, zorder=2)

    ax.plot(x, means, color="#176B87", linewidth=3.0, marker="o", markersize=5.2,
            markerfacecolor="#176B87", markeredgecolor="#176B87", markeredgewidth=2.0,
            label="Group mean", zorder=4)
    ax.set_xticks(x, DISPLAY_LABELS)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(1.0, 3.2)
    ax.set_xlabel("Experimental phase", labelpad=7)
    ax.set_ylabel("Phase-mean lnRMSSD", labelpad=7)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=4, width=2.0, color="#50575C")
    fig.legend(handles=ax.get_legend_handles_labels()[0], labels=ax.get_legend_handles_labels()[1],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               handlelength=1.5, ncol=1)

    paths = {
        "png": FINAL / "Figure_12.png",
        "jpg": FINAL / "Figure_12.jpg",
        "tif": FINAL / "Figure_12.tif",
    }
    fig.savefig(paths["png"], dpi=300, facecolor="white")
    fig.savefig(paths["jpg"], dpi=300, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
    fig.savefig(paths["tif"], dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    image_checks = []
    for extension, path in paths.items():
        with Image.open(path) as image:
            if image.mode != "RGB":
                rgb = image.convert("RGB")
                if extension == "jpg":
                    rgb.save(path, format="JPEG", dpi=(300, 300), quality=95, subsampling=0)
                elif extension == "tif":
                    rgb.save(path, format="TIFF", dpi=(300, 300), compression="tiff_lzw")
                else:
                    rgb.save(path, format="PNG", dpi=(300, 300))
        with Image.open(path) as image:
            image_checks.append((extension, image.size, image.mode, image.info.get("dpi")))
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB output, received {image.mode} for {path.name}")
            if image.size != (2126, 1476):
                raise ValueError(f"Unexpected canvas {image.size} for {path.name}")

    report = [
        "# Figure 12 production and quality check",
        "",
        "**Build:** generated from the packaged descriptive dataset; no new analysis.",
        "",
        f"**Source:** `{SOURCE.relative_to(ROOT).as_posix()}`",
        f"**Source SHA-256:** `{sha256(SOURCE)}`",
        f"**Source copy SHA-256:** `{sha256(SOURCE_COPY)}`",
        "**Biological units:** seven cows; one phase mean per cow for each of four phases.",
        "**Column mapping:** PRE60 → PRE; O1 → O11; INTERMEDIATE → Intervening interval; O2 → O12.",
        "**Endpoint:** animal-level phase-mean lnRMSSD. Lines connect repeated observations from the same cow. The emphasized blue line shows the arithmetic group mean at each phase.",
        "**Group means (PRE, O11, intervening interval, O12):** " + ", ".join(f"{v:.4f}" for v in means),
        "**Validation:** seven unique animals; no missing/non-finite values; group means match the manuscript-reported values within 0.00005; PNG/JPG/TIF are RGB at 300 dpi and 2126 × 1476 pixels (180 × 125 mm nominal); plotted traces, axes, and ticks are at least 2 pt wide; no fine grid lines are used; the group-mean legend is centered below the plotting area, consistent with Figures 10 and 11.",
        "**Interpretive limits:** descriptive phase summaries only. Fixed sequence does not support an O12-specific causal effect; O11/O12 are not independent replications; the intervening interval is not identified as washout, rest, or silence.",
        "",
        "## Exports",
        "",
    ]
    for extension, path in paths.items():
        check = next(item for item in image_checks if item[0] == extension)
        report.append(f"- `{path.name}` — {check[1][0]} × {check[1][1]} px, {check[2]}, SHA-256 `{sha256(path)}`")
    (PROVENANCE / "Figure12_production_and_QC.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Figure 12 exports created and passed data/canvas/color checks.")
    for extension, size, mode, dpi in image_checks:
        print(f"{extension}: {size[0]}x{size[1]} px, {mode}, {dpi}")


if __name__ == "__main__":
    main()
