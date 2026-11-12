# Reproducibility guide

This guide describes the analysis routes supported by the files in this repository. It distinguishes rebuilds from packaged derived values from workflows that require source recordings or feature tables not distributed here.

## Software environments

Install the packages listed in the environment file for the route being run. The repository pins selected figure, supplementary-document, OAAS-core, and HRV-analysis dependencies separately. Run commands from the repository root. Keep route-specific software environments separate where their pinned package versions differ.

## Figures and supplementary materials

- Figures 2–13 can be rebuilt using their figure scripts and packaged plot inputs. Figure-specific instructions and dependencies are stored with each figure.
- Figure 7 can be rebuilt with `scripts/figures/main/build_figure_07.py`; it reads six compressed spectrogram arrays, acoustic curves, and the score excerpt from the packaged folders. This uses saved derived arrays and does not process source audio.
- Figure 13 can be rebuilt with `figures/main/Figure_13/01_CODE/build_figure_13.py` and the supplied icon source at `figures/support/figure_13/icon_source.tif`.
- Figure 1 can be rebuilt from the packaged `data/derived/figure_01/Figure01_representative_spectrograms.npz` arrays and `Figure01_all_categories_band_profiles.csv` table with `scripts/figures/main/Figure_01/build_figure_01.py`. The renderer reads derived values only; it does not need or process source audio. BovineTalk-derived LFC/HFC values are attributed in the data metadata and `LICENSE_DATA`.
- Supplementary Figure S2 can be rebuilt from the packaged Figure 5 behavior and OAAS coordinate tables with `scripts/figures/supplementary/Figure_S2/build_figure_s2.py`. It regenerates the nine category summaries and the PNG/JPEG/TIFF exports without source audio.
- Supplementary Figure S3 can be rebuilt from its packaged window table and count manifest. From the repository root, run:

  ```bash
  python scripts/figures/supplementary/Figure_S3/build_figure_s3.py --windows-csv data/derived/supplementary_figure_s3/FigureS3_5s_acoustic_information_windows.csv --source-manifest data/derived/supplementary_figure_s3/source_manifest.json --outdir outputs/supplementary_figure_s3
  ```

  Its upstream window-generation script is included, but reproducing those windows requires source audio fragments that are not distributed.
- Supplementary Methods, Supplementary Figures, and Supplementary Tables can be rebuilt from their respective Markdown files using the three supplementary PDF builders.

These rebuilds reproduce the packaged figures and reader-facing documents. They do not imply that every upstream analysis or audio-processing step is available.

## OAAS fits and acoustic distances

The OAAS fitting and distance scripts operate on the 208-feature MICRO and MACRO tables. The path-sanitized canonical tables are included under `data/derived/oaas_core/` and are covered by the file-scoped CC BY 4.0 terms. From the repository root, fit each scale and then calculate its centroid distances:

```bash
python scripts/oaas/fit_oaas_model.py --features_csv data/derived/oaas_core/features_MICRO_derived.csv --output_dir outputs/OAAS_MICRO
python scripts/oaas/compute_oaas_distances.py --oaas_dir outputs/OAAS_MICRO
python scripts/oaas/fit_oaas_model.py --features_csv data/derived/oaas_core/features_MACRO_derived.csv --output_dir outputs/OAAS_MACRO
python scripts/oaas/compute_oaas_distances.py --oaas_dir outputs/OAAS_MACRO
```

The scripts reproduce model fits, projections, bovine-centroid distances, and nearest-class assignments from an existing feature table. They do not extract features from audio or regenerate the 180-s ensembles.

## MICRO–MACRO random-assembly Null

The released `data/derived/assembly_null/null_iteration_results.csv` contains the 500 valid planned realizations (base seeds 10000–10499), and `c2_frozen_micro_hfc_obonuco_ids.csv` records the 127 Obonuco fragments selected as HFC-nearest in MICRO. Recompute the descriptive summary from the released results with:

```bash
python scripts/oaas/summarize_assembly_null.py \
  --results-csv data/derived/assembly_null/null_iteration_results.csv \
  --output-csv outputs/assembly_null/null_summary.csv
```

The Monte Carlo upper-tail values use the plus-one estimate `(count + 1) / (N + 1)`. For C2, ties count in the upper tail; the observed value is reproduced exactly in 146 of 500 realizations, with no exceedances.

To rerun the primary 500-realization simulation, install `environment/requirements-oaas-core-validated.txt` and provide authorized local access to the source bovine recordings. The audio root must contain `analysis_ready_clean/BOVINE_CORE/`; the recordings are not included in this repository. From the repository root, run:

```bash
python scripts/oaas/run_assembly_null.py \
  --audio-corpus-root /path/to/authorized_audio_corpus \
  --output-dir outputs/assembly_null \
  --stage primary
```

The sanitized corpus metadata, frozen MACRO feature table, and C2 ID list default to the copies packaged here. The script regenerates 20 category-specific 180-s ensembles per realization, extracts the 208 descriptors, refits the bovine-centered MACRO OAAS, projects the comparison set, and records C1/C2. Temporary generated audio and intermediate feature files are removed after each iteration. This full rerun route requires external recordings and is not executable from the public repository alone; the released results table permits independent recomputation of the reported summary.

## Acoustic-to-HRV analysis

The GEE script runs from the packaged 126-row model input under `data/derived/hrv_acoustic_alignment/`. Install `environment/requirements-hrv-model.txt` and run `python scripts/hrv_alignment/fit_gee_models.py --output-dir outputs/hrv_models` from the repository root to reproduce the primary and sensitivity models. The separate alignment and section-weighting scripts are included, but require the individual-level Kubios physiology input, which is not distributed; therefore those upstream tables cannot be regenerated from this repository alone. The packaged model route does not process audio and does not estimate causal effects or physiological latency.

## Audio scope

The single Obonuco O1/FASE example is for listening context only and is not used by the rebuild scripts. The larger source-audio corpus and third-party recordings are not distributed. Audio-use terms are in `LICENSE_AUDIO`.

## Reproduction limits

Rebuilding a figure from packaged plot data is not the same as repeating the full experiment or regenerating every upstream dataset. This repository does not provide the recordings needed to redo complete source-audio feature extraction or ensemble generation. The OAAS core route uses the packaged path-sanitized feature tables; raw recordings remain external.

