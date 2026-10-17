# Data files

This directory contains metadata and selected derived data supporting the associated article. The complete list of data files covered by CC BY 4.0 is `../docs/DATA_LICENSE_FILE_MANIFEST.csv`; consult `../LICENSE_DATA` for attribution and scope.

## Metadata

- `metadata/Supplementary_Dataset_D1_Corpus_Metadata.csv` describes 384 comparative-corpus records. It contains metadata and relative source identifiers, not audio. Those identifiers do not provide access to recordings or permission to redistribute them.
- `metadata/Supplementary_Dataset_D2_Technical_Audio_Metadata.csv` provides technical properties read from headers for the same 384 source files. It contains no filenames or source paths and does not report measured acoustic bandwidth.
- `metadata/Obonuco_FASE_Analytical_Object_Metadata.csv` inventories the 145 analytical objects used in the Obonuco FASE work: 130 fragments, 10 long units, and five stems.

## Derived data

`derived/` contains compact values used by supported figure and analysis routes, including the 208-feature MICRO and MACRO OAAS tables, acoustic-to-HRV model inputs and outputs, and figure plotting data. Figure 1 can be rebuilt from its packaged four-category profile table and derived spectrogram arrays. The separate `figure_01/local_colombian/` tables provide additional derived values for author-collected GC and PSC recordings. `figure_07/` contains the acoustic-curve data and six compressed spectrogram arrays needed to rebuild Figure 7; these are derived from the retained O1/O2 production recordings, whose source WAV files are not included.

`derived/supplementary_figure_s2/` contains the nine category-level values plotted in Supplementary Figure S2, its caption, and build provenance. The table is derived from the packaged Figure 5 behavior and OAAS coordinate tables and can be regenerated with `scripts/figures/supplementary/Figure_S2/build_figure_s2.py`.

`derived/supplementary_figure_s3/` contains 552 five-second acoustic-information windows and a small count manifest used to validate Supplementary Figure S3. The plotting script runs from these packaged inputs. The upstream window-generation script is included for transparency but requires the source audio fragments, which are not distributed.

The source-audio corpus is not included. Reproducing the complete audio-to-feature processing requires authorized access to the corresponding recordings and is outside the included package.
