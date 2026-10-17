# Figure 1 reproduction

Figure 1 is rebuilt from the derived data packaged with this repository. Its renderer does not require or process source audio.

From the repository root, run:

```bash
python scripts/figures/main/Figure_01/build_figure_01.py
```

The script reads `data/derived/figure_01/Figure01_representative_spectrograms.npz` for the four representative spectrograms and `data/derived/figure_01/Figure01_all_categories_band_profiles.csv` for the category-level profiles in Panel C. The plotted derived arrays are included; the source recordings are not.

The LFC/HFC example values derive from BovineTalk and are attributed to Gavojdian et al. (2024). The GC/PSC values derive from author-collected Colombian recordings. The CC BY 4.0 scope for the derived data is listed in `docs/DATA_LICENSE_FILE_MANIFEST.csv`; no source audio is distributed.

By default, the script writes rebuilt files to `outputs/figure_01/` so the packaged publication exports under `figures/main/` remain unchanged. The repository includes those publication exports as `Figure_01.png` and `Figure_01.tif`.

## Source citation

Gavojdian et al. (2024), *BovineTalk: Machine Learning for Vocalization Analysis of Dairy Cattle under the Negative Affective State of Isolation*. https://doi.org/10.3389/fvets.2024.1357109.
