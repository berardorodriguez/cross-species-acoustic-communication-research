# Research materials for cross-species acoustic communication

This repository provides selected data, scripts, figures, and supplementary materials associated with:

**Affective Information Transmission in Cross-Species Acoustic Communication: Evidence for Multiscale Acoustic–Affective Organization**

The materials support inspection and selected reproduction routes for the analyses and figures described in the article. The full source-audio corpus is not included.

## Contents

- Main-article figures and the data or scripts needed for supported rebuilds.
- Supplementary Methods, Supplementary Figures, and Supplementary Tables, each provided as PDF and Markdown.
- Metadata datasets and selected derived acoustic and physiological-analysis tables.
- Scripts and pinned environment files for figure production, selected OAAS fitting and distance calculations, and the reported acoustic-to-HRV analyses.
- One short Obonuco O1/FASE audio example for listening context, subject to the separate terms in `LICENSE_AUDIO`.

## Reproduction scope

The approved figure files and three supplementary PDFs are included. Figure 1 can be rebuilt from packaged derived spectrogram arrays and category profiles; no source WAV files are needed by its renderer or distributed in this repository. BovineTalk-derived LFC/HFC values are attributed in the data metadata. The OAAS fitting route starts from the path-sanitized, packaged MICRO and MACRO feature tables. The acoustic-to-HRV route starts from its compact, derived model input and does not process audio.

See `REPRODUCIBILITY.md` for supported commands, software environments, and route-specific limits.

The scripts directory describes each included workflow and marks the steps that require source recordings or analysis inputs held outside this repository.

## Audio example

`audio_examples/Obonuco_O1_Representative_Excerpt.wav` is provided for listening and citation in the context of the associated publication. It is not an analysis input and does not represent the full composition. The author authorizes this limited access; other reuse, redistribution, or adaptation requires separate permission. See `LICENSE_AUDIO`. No third-party source recordings or broader audio corpus are included.

## Figure-production disclosure

The Figure 13 pictograms were generated using ChatGPT (OpenAI), including GPT-5.5. OpenAI Codex assisted with the Python script used to compose and render the editable vector figure with Matplotlib. OpenAI ImageGen was used for exploratory visual candidates; those raster outputs are not embedded in the final vector. The authors reviewed the figure and are responsible for its content. The manuscript contains the complete AI-use disclosure.

## Citation and reuse

Please cite the associated article and this repository when using these materials. Code is released under the MIT License in `LICENSE_CODE`, including OAAS-derived adaptations; see `THIRD_PARTY_NOTICES.md` for the preserved upstream notice. The 60 files listed in `docs/DATA_LICENSE_FILE_MANIFEST.csv` are released under CC BY 4.0. Figures, supplementary documents, and audio have separate rights and licensing conditions. Audio terms are file-scoped in `LICENSE_AUDIO` and do not grant reuse rights for other content.

## File verification

`docs/PUBLIC_PACKAGE_INVENTORY.csv` lists package files, their sizes, and SHA-256 hashes. `docs/SHA256SUMS.txt` provides a standard checksum list. Each list excludes both inventory outputs to avoid self-referential checksums. After changing package contents, regenerate both from the repository root with `python scripts/repository/build_package_inventory.py`.

