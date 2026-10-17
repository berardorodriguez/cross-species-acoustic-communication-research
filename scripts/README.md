# Scripts and reproducibility boundaries

The Python scripts are licensed under MIT. Install the route-specific packages from the matching file in `environment/`; run commands from the repository root. Generated build products are ignored by Git.

## Routes using packaged inputs

- Figure builders for Figures 2–13 use packaged plotting tables, arrays, and figure assets. Figure 2 takes explicit `--micro`, `--macro`, `--obonuco-exact-id`, and `--outdir` arguments. Figure 7 reads the six spectrogram arrays and acoustic curve data under `data/derived/figure_07/`. Figure 13 uses `figures/support/figure_13/icon_source.tif` for the pictogram source crops.
- Figure 1 is rebuilt by `scripts/figures/main/Figure_01/build_figure_01.py` from the packaged representative spectrogram arrays and four-category profile table. Install `environment/requirements-figure01-derived-data.txt`; the renderer does not need or process source audio. The data-generation provenance records the four representative files; their WAV recordings are not distributed.
- Supplementary Figure S2 is rebuilt by `scripts/figures/supplementary/Figure_S2/build_figure_s2.py` from the packaged Figure 5 behavior and OAAS coordinate tables. The route writes its category summary under `data/derived/supplementary_figure_s2/` and the PNG/JPEG/TIFF figure exports under `figures/supplementary/`.
- Supplementary Figure S3 is rebuilt from its packaged window table and count manifest. From the repository root, run `python scripts/figures/supplementary/Figure_S3/build_figure_s3.py --windows-csv data/derived/supplementary_figure_s3/FigureS3_5s_acoustic_information_windows.csv --source-manifest data/derived/supplementary_figure_s3/source_manifest.json --outdir outputs/supplementary_figure_s3`. The included `build_figure_s3_window_data.py` documents the upstream route, which requires source audio fragments not distributed here.
- `scripts/oaas/fit_oaas_model.py` and `scripts/oaas/compute_oaas_distances.py` rebuild the MICRO/MACRO fits and distances from the packaged feature tables. `extract_oaas_features.py` and `make_bovine_ensembles_180s.py` document optional upstream workflows for separately held audio; those recordings and their end-to-end processing are outside this repository's reproduction route.
- `scripts/acoustic_scores/verify_saved_scores.py` checks the retained O1 A_adj rows against their packaged descriptors and reference-IQR tables. It does not process audio.
- `scripts/hrv_alignment/fit_gee_models.py` runs the prespecified primary and sensitivity models from the packaged 126-row input. Install `environment/requirements-hrv-model.txt` first; this route uses statsmodels. It does not process audio. The preceding alignment and section-weighting scripts are included but require individual-level Kubios physiology input that is not distributed here, so their upstream outputs cannot be regenerated from this package alone.
- The three scripts under `scripts/supplementary/` rebuild the separate Methods, Figures, and Tables PDFs from their bundled Markdown and figure files.

## External source files

`scripts/metadata/generate_technical_audio_metadata.py` requires the separately held 384-file comparative corpus; the corresponding audio files are not distributed. Full audio feature extraction, 180-second ensemble generation, and upstream alignment from raw acoustic/HRV records are not claimed as package-local reproduction routes.
