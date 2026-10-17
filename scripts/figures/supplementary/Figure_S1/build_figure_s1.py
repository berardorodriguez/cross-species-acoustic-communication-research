"""Build  Supplementary Figure S1 from exact-ID MICRO/MACRO matches.

The figure describes nearest-bovine-reference class correspondence for the
acoustic-control subsets represented as synthetic, mechanical/environmental,
and high-distortion heavy-metal music. The source subdomain key remains
human_extreme. The figure does not depict a temporal or causal transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

CLASSES = ["GC", "LFC", "HFC", "PSC"]
FAMILIES = {
    "Synthetic sounds": {"synthetic"},
    "Mechanical/environmental sounds": {
        "airplane", "chainsaw", "engine", "hand_saw", "helicopter", "train", "washing_machine"
    },
    "High-distortion heavy-metal music": {"human_extreme"},
}
EXPECTED_ROWS = {
    "Synthetic sounds": 3,
    "Mechanical/environmental sounds": 22,
    "High-distortion heavy-metal music": 4,
}
FINAL_WIDTH_MM = 180.0
DPI = 300
LINE_PT = 2.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_class(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper()
            .str.replace("DIST_TO_", "", regex=False)
            .str.replace("NEAREST_", "", regex=False)
            .str.replace("CLASS_", "", regex=False))


def exact_control_pairs(plot_data_path: Path) -> tuple[pd.DataFrame, dict]:
    needed = {"object_id", "family", "subdomain", "class_micro", "class_macro"}
    pairs = pd.read_csv(plot_data_path)
    missing = needed - set(pairs.columns)
    if missing:
        raise ValueError(f"Compact S1 plot data is missing: {sorted(missing)}")
    if pairs.object_id.duplicated().any():
        raise ValueError("Duplicate pseudonymous control IDs in compact S1 plot data")
    pairs["subdomain_micro"] = pairs["subdomain"].astype(str).str.strip().str.lower()
    pairs["class_micro"] = clean_class(pairs["class_micro"])
    pairs["class_macro"] = clean_class(pairs["class_macro"])
    if not pairs.class_micro.isin(CLASSES).all() or not pairs.class_macro.isin(CLASSES).all():
        raise ValueError("Unexpected nearest-bovine-reference class")

    ordered = []
    for family in FAMILIES:
        block = pairs.loc[pairs.family.eq(family)].copy()
        if len(block) != EXPECTED_ROWS[family]:
            raise ValueError(f"{family}: expected {EXPECTED_ROWS[family]} exact-ID rows, found {len(block)}")
        ordered.append(block)
    result = pd.concat(ordered, ignore_index=True)
    if result.object_id.duplicated().any() or len(result) != 29:
        raise ValueError("Expected 29 unique matched control objects")
    audit = {
        "matched_control_objects": int(len(pairs)),
        "family_counts": {k: int(v) for k, v in result.family.value_counts().to_dict().items()},
    }
    return result, audit


def counts_and_proportions(block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(block.class_micro, block.class_macro).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    row_n = counts.sum(axis=1)
    props = counts.div(row_n.replace(0, np.nan), axis=0) * 100
    return counts, props


def draw_panel(ax, block: pd.DataFrame, panel: str, title: str):
    counts, props = counts_and_proportions(block)
    row_n = counts.sum(axis=1)
    cmap = LinearSegmentedColormap.from_list(
        "viridis_uniform_black_text", plt.get_cmap("viridis")(np.linspace(0.50, 1.00, 256))
    )
    cmap.set_bad("0.88")
    ax.imshow(np.ma.masked_invalid(props.to_numpy(float)), vmin=0, vmax=100,
              aspect="equal", cmap=cmap)
    ax.set_xticks(range(4), CLASSES, fontsize=8)
    ax.set_yticks(range(4), [f"{c} (n={int(row_n.loc[c])})" for c in CLASSES], fontsize=8)
    ax.set_xlabel("MACRO nearest bovine reference", fontsize=8, labelpad=3)
    ax.set_ylabel("MICRO nearest bovine reference", fontsize=8, labelpad=3)
    ax.set_title(f"{title}\nn = {len(block)} matched units", fontsize=9, pad=6)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", linewidth=LINE_PT, color="0.72")
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_PT)
    for i, cls_micro in enumerate(CLASSES):
        if int(row_n.loc[cls_micro]) == 0:
            continue
        for j, cls_macro in enumerate(CLASSES):
            n = int(counts.loc[cls_micro, cls_macro])
            if n:
                p = float(props.loc[cls_micro, cls_macro])
                pct = "100%" if np.isclose(p, 100) else f"{p:.1f}%"
                ax.text(j, i, f"{n}\n({pct})", ha="center", va="center",
                        fontsize=8, color="black")
    ax.text(-0.25, 1.08, f"({panel})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="bottom")
    return counts, props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-data", required=True, type=Path,
                    help="Compact pseudonymous exact-ID MICRO/MACRO class assignments")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.outdir
    final = root / "00_REBUILD"
    source_dir = root / "02_PROVENANCE" / "SOURCE_DATA"
    final.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    pairs, audit = exact_control_pairs(args.plot_data.resolve())
    table = source_dir / "FigureS1__acoustic_control_exact_id_matches.csv"
    pairs.to_csv(table, index=False)
    counts_manifest = {}

    fig = plt.figure(figsize=(180 / 25.4, 7.0), dpi=DPI)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.10],
                          left=0.105, right=0.97, top=0.91, bottom=0.10,
                          wspace=0.52, hspace=0.68)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0])]
    note_ax = fig.add_subplot(gs[1, 1])
    note_ax.axis("off")
    cbar_ax = fig.add_subplot(gs[2, :])
    for ax, (panel, (family, _)) in zip(axes, zip("ABC", FAMILIES.items())):
        block = pairs.loc[pairs.family.eq(family)]
        counts, _ = draw_panel(ax, block, panel, family)
        counts_manifest[family] = {r: {c: int(counts.loc[r, c]) for c in CLASSES} for r in CLASSES}

    cmap = LinearSegmentedColormap.from_list(
        "viridis_uniform_black_text", plt.get_cmap("viridis")(np.linspace(0.50, 1.00, 256))
    )
    sm = ScalarMappable(norm=Normalize(0, 100), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Within-MICRO-row proportion (%)", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=8, width=LINE_PT)
    cbar.outline.set_linewidth(LINE_PT)
    note_ax.text(0.5, 0.62, "Gray = no MICRO observations;\nrow percentage is undefined.",
                 ha="center", va="center", fontsize=9)
    fig.suptitle("Cross-scale nearest-reference correspondence in acoustic controls",
                 fontsize=10, y=0.995)
    png = final / "Supplementary_Figure_S1.png"
    jpg = final / "Supplementary_Figure_S1.jpg"
    tif = final / "Supplementary_Figure_S1.tif"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(jpg, dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    fig.savefig(tif, dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    for path in (png, tif):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if path.suffix.lower() == ".tif":
                rgb.save(path, compression="tiff_lzw", dpi=(DPI, DPI))
            else:
                rgb.save(path, dpi=(DPI, DPI))

    manifest = {
        "figure": "Supplementary Figure S1",
        "build_status": "reproduced_from_packaged_data",
        "interpretation": "Descriptive exact-ID nearest-reference correspondence in independently fitted MICRO and MACRO representations; not a physical or causal transition.",
        "groups": audit["family_counts"],
        "subdomain_counts": {
            family: {str(k): int(v) for k, v in pairs.loc[pairs.family.eq(family), "subdomain_micro"].value_counts().to_dict().items()}
            for family in FAMILIES
        },
        "matched_acoustic_control_objects": audit["matched_control_objects"],
        "counts_by_micro_to_macro_class": counts_manifest,
        "sources": {
            "compact_cross_scale_assignments": {
                "path": "data/derived/supplementary_figure_s1/FigureS1_acoustic_control_cross_scale_classes.csv",
                "sha256": sha256(args.plot_data.resolve()),
            },
            "exact_id_matching": "Validated upstream from the released MICRO/MACRO feature tables; source IDs are replaced by pseudonymous object labels in the compact plot table.",
        },
        "exports": {},
        "production": {"width_mm": 180, "height_mm": 177.8, "dpi": DPI, "color_mode": "RGB", "panel_labels": "(A)-(C)"},
    }
    for path in (png, jpg, tif):
        with Image.open(path) as im:
            manifest["exports"][path.name] = {
                "path": path.relative_to(root).as_posix(), "sha256": sha256(path), "format": im.format,
                "mode": im.mode, "pixels": list(im.size),
                "dpi": [float(x) if x is not None else None for x in im.info.get("dpi", (None, None))],
                "compression": im.info.get("compression"),
            }
    (root / "02_PROVENANCE" / "FigureS1__provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": audit, "counts": counts_manifest, "output": str(png)}, indent=2))


if __name__ == "__main__":
    main()
