from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[2]
BEHAVIOR = PROJECT / "data" / "derived" / "figure_05" / "Figure05_behavior_hrv_minute_level.csv"
OAAS = PROJECT / "data" / "derived" / "figure_05" / "Figure05A_OAAS_plot_data.csv"
OUT = PROJECT / "figures" / "supplementary"
PROV = PROJECT / "data" / "derived" / "supplementary_figure_s2"
SOURCE_OUT = PROV
for directory in (OUT, PROV):
    directory.mkdir(parents=True, exist_ok=True)

DPI = 300
WIDTH_MM = 180
HEIGHT_MM = 128
ORDER = [
    "BASELINE", "RUIDO_ROSA", "SILBOS", "PASTOREO", "CANTO_ORDENO",
    "KULNING", "GRITO_MONTE", "VAQUERIA", "FERIA",
]
LABELS = {
    "BASELINE": "Baseline/rest",
    "RUIDO_ROSA": "Pink noise",
    "SILBOS": "Herding whistles",
    "PASTOREO": "Grazing calls (bovine)",
    "CANTO_ORDENO": "Milking songs",
    "KULNING": "Kulning",
    "GRITO_MONTE": "Range shouts",
    "VAQUERIA": "Cattle-herding calls",
    "FERIA": "Pre-slaughter calls (bovine)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


behavior = pd.read_csv(BEHAVIOR)
required_behavior = {"canonical", "mirar_prop", "p1_mean", "p2_mean", "p3_mean", "herd_est"}
missing = required_behavior - set(behavior.columns)
if missing:
    raise ValueError(f"Missing behavior fields: {sorted(missing)}")
if len(behavior) != 90:
    raise ValueError(f"Expected 90 retained group-level intervals, found {len(behavior)}")
behavior["source_orientation_s_min_equivalent"] = (
    (behavior["p1_mean"] + behavior["p2_mean"] + behavior["p3_mean"])
    / behavior["herd_est"] * 60.0
)
formula_error = float(
    np.max(np.abs(behavior["source_orientation_s_min_equivalent"] - behavior["mirar_prop"] * 60.0))
)
if formula_error > 1e-8:
    raise ValueError(f"Reconstructed endpoint differs from retained mirar_prop by {formula_error}")

behavior_summary = (
    behavior.groupby("canonical", sort=False)
    .agg(
        n_intervals=("canonical", "size"),
        orientation_mean=("source_orientation_s_min_equivalent", "mean"),
        orientation_sd=("source_orientation_s_min_equivalent", "std"),
    )
    .reindex(ORDER)
)
if behavior_summary.isna().any().any():
    raise ValueError("An expected category is absent from the behavior source")

# Use the same category-wise median LNU1/LNU2 centroids as Table S1 and Figure 5A.
coordinates = pd.read_csv(OAAS)
required_coordinates = {"canonical", "x", "y"}
if required_coordinates - set(coordinates.columns):
    raise ValueError(f"Missing Figure 5A coordinate fields: {sorted(required_coordinates - set(coordinates.columns))}")
centroids = coordinates.groupby("canonical")[ ["x", "y"] ].median()
if "PASTOREO" not in centroids.index:
    raise ValueError("Grazing-call bovine reference (PASTOREO) is absent from the Figure 5A coordinates")
reference = centroids.loc["PASTOREO"]
distance = np.sqrt(((centroids.loc[ORDER] - reference) ** 2).sum(axis=1))

data = pd.DataFrame({
    "category_order": np.arange(1, len(ORDER) + 1),
    "canonical": ORDER,
    "category_label": [LABELS[c] for c in ORDER],
    "n_group_level_one_minute_intervals": behavior_summary["n_intervals"].astype(int).to_numpy(),
    "distance_from_grazing_call_reference_LNU1_LNU2": distance.to_numpy(),
    "mean_source_directed_orientation_s_min_equivalent": behavior_summary["orientation_mean"].to_numpy(),
    "sd_source_directed_orientation_s_min_equivalent": behavior_summary["orientation_sd"].to_numpy(),
})
stat = spearmanr(data["distance_from_grazing_call_reference_LNU1_LNU2"],
                 data["mean_source_directed_orientation_s_min_equivalent"])
rho, p_value = float(stat.statistic), float(stat.pvalue)
if not np.isfinite(rho) or len(data) != 9:
    raise ValueError("The category-level association could not be calculated")

csv_path = SOURCE_OUT / "category_distance_orientation.csv"
data.to_csv(csv_path, index=False, float_format="%.10g")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.facecolor": "white",
})
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=DPI)
ax = fig.add_axes([0.105, 0.20, 0.56, 0.72])
palette = ["#496A81", "#758C9B", "#8B6F9A", "#2B7A78", "#B47B39", "#557C55", "#9C5D5D", "#3D6B8E", "#7A664A"]

for row, color in zip(data.itertuples(index=False), palette):
    ax.scatter(
        row.distance_from_grazing_call_reference_LNU1_LNU2,
        row.mean_source_directed_orientation_s_min_equivalent,
        s=90, color=color, edgecolor="white", linewidth=2.0, zorder=3,
    )
    ax.text(
        row.distance_from_grazing_call_reference_LNU1_LNU2,
        row.mean_source_directed_orientation_s_min_equivalent,
        str(row.category_order), ha="center", va="center", color="white",
        fontsize=8, fontweight="bold", zorder=4,
    )

ax.set_xlabel("Distance from grazing-call reference (LNU1–LNU2)", labelpad=6)
ax.set_ylabel("Mean source-directed orientation\n(s/min-equivalent)", labelpad=6)
ax.set_xlim(-0.20, 4.45)
ax.set_ylim(7.5, 20.0)
ax.set_xticks(np.arange(0, 4.6, 1.0))
ax.set_yticks(np.arange(8, 20, 2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)
ax.tick_params(width=2.0, length=4)
ax.grid(False)
ax.text(
    0.04, 0.96,
    f"Spearman ρ = {rho:.2f}; p = {p_value:.4f}\n9 category summaries",
    transform=ax.transAxes, ha="left", va="top", fontsize=8,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "none", "linewidth": 0},
)

handles = []
for row, color in zip(data.itertuples(index=False), palette):
    handles.append(plt.Line2D([], [], marker="o", linestyle="None", markersize=4.5,
                              markerfacecolor=color, markeredgecolor="white",
                              label=f"{row.category_order}. {row.category_label} (n={row.n_group_level_one_minute_intervals})"))
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.69, 0.51),
           frameon=False, labelspacing=1.0, handletextpad=0.55, borderaxespad=0.0)

outputs = {
    "png": OUT / "Supplementary_Figure_S2.png",
    "jpg": OUT / "Supplementary_Figure_S2.jpg",
    "tif": OUT / "Supplementary_Figure_S2.tif",
}
intermediates = {
    path: Path(tempfile.gettempdir()) / (path.stem + ".matplotlib" + path.suffix)
    for path in outputs.values()
}
for path, temp_path in intermediates.items():
    fig.savefig(temp_path, dpi=DPI, bbox_inches=None)
plt.close(fig)

for path, temp_path in intermediates.items():
    with Image.open(temp_path) as image:
        rgb = image.convert("RGB")
        if path.suffix == ".tif":
            rgb.save(path, dpi=(DPI, DPI), compression="tiff_lzw")
        elif path.suffix == ".jpg":
            rgb.save(path, dpi=(DPI, DPI), quality=95, optimize=True)
        else:
            rgb.save(path, dpi=(DPI, DPI), optimize=True)
    with Image.open(path) as image:
        image.verify()

caption = (
    "Supplementary Figure S2. Category-level association between acoustic distance from the bovine grazing-call reference "
    "and source-directed orientation during playback. Each point represents one of nine playback-category summaries. "
    "Horizontal position is the Euclidean distance between the category-wise median centroid and the grazing-call reference "
    "in the LNU1–LNU2 representation underlying Figure 5A; vertical position is mean group-level source-directed orientation "
    "across repeated one-minute intervals, expressed on an s/min-equivalent scale. Across the nine category summaries, "
    f"distance was negatively associated with mean source-directed orientation (Spearman ρ = {rho:.2f}, p = {p_value:.4f}; "
    "n = 9). The association is descriptive at the category level. Numbers in the key give the number of repeated one-minute "
    "group-level intervals per category."
)
(PROV / "caption.md").write_text(
    "# Supplementary Figure S2\n\n" + caption +
    "\n",
    encoding="utf-8",
)

provenance = {
    "figure": "Supplementary Figure S2",
    "status": "reproducible_build_from_packaged_inputs",
    "scientific_scope": "Descriptive association across nine category summaries; not independent validation and not an individual-animal prediction.",
    "acoustic_distance": "Euclidean distance between category-wise median centroids in LNU1-LNU2; reference is the PASTOREO (bovine grazing-call) centroid; matches the coordinate source and distance construction used by Supplementary Table S1.",
    "behavior_endpoint": "((p1_mean + p2_mean + p3_mean) / herd_est) * 60; source-directed group orientation expressed as s/min-equivalent.",
    "endpoint_formula_max_abs_difference_from_mirar_prop_x60": formula_error,
    "category_summaries": 9,
    "group_level_one_minute_intervals": int(len(behavior)),
    "spearman_rho": rho,
    "spearman_p_value": p_value,
    "no_linear_fit": True,
    "sources": {
        str(path.relative_to(PROJECT).as_posix()): {
            "path": path.relative_to(PROJECT).as_posix(), "sha256": sha256(path)
        }
        for path in (BEHAVIOR, OAAS)
    },
    "derived_source_table": {"path": csv_path.relative_to(PROJECT).as_posix(), "sha256": sha256(csv_path)},
    "exports": {},
    "production": {
        "width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "dpi": DPI,
        "color_mode": "RGB", "tiff_compression": "LZW",
        "minimum_text_pt": 8, "minimum_solid_line_pt": 2,
        "layout": "single panel; one-page height",
    },
}
for fmt, path in outputs.items():
    with Image.open(path) as image:
        provenance["exports"][path.name] = {
            "path": path.relative_to(PROJECT).as_posix(), "sha256": sha256(path), "format": image.format,
            "mode": image.mode, "pixels": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", ())],
            "compression": image.info.get("compression"),
        }
(PROV / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

print(json.dumps({"rho": rho, "p": p_value, "n_categories": 9, "n_intervals": len(behavior), "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
