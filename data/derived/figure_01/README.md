# Figure 1 derived plotting data

Figure 1 is reproducible from the data files in this folder. `Figure01_representative_spectrograms.npz` stores the plotted time-frequency values and axes for the four representative examples used in panels A and B. `Figure01_all_categories_band_profiles.csv` stores the 300 normalized-time values for each of the three frequency bands across GC, LFC, PSC, and HFC in panel C. The renderer reads these files directly and does not need, open, or process source audio.

The spectrogram arrays were derived from one representative recording per category using a 48-kHz sample rate, 1,024-point Hann-window FFT, 256-sample hop, 0–3.5-kHz display range, and an 80-dB display floor. The profile table contains category means from GC 11, LFC 59, PSC 20, and HFC 108 source files, processed with the parameters recorded in the manuscript figure provenance. Source recordings are not included.

The local Colombian GC/PSC tables under `local_colombian/` provide additional derived values and processing metadata. All files listed in `docs/DATA_LICENSE_FILE_MANIFEST.csv` are covered by the repository's file-scoped CC BY 4.0 data notice. BovineTalk-derived LFC/HFC values are attributed to Gavojdian et al. (2024), with the author's stated working assumption about reuse terms; no BovineTalk WAV audio is redistributed.
