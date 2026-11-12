# MICRO–MACRO random-assembly Null: public reproduction notes

The primary comparison contains 500 valid random 180-s bovine-ensemble realizations generated with planned base seeds 10000–10499. The runner generates five ensembles for each bovine reference class (GC, LFC, HFC, PSC), extracts the frozen 208 acoustic descriptors, refits the bovine-centered MACRO OAAS, and projects the comparison materials.

`data/derived/assembly_null/null_iteration_results.csv` contains one row per valid planned realization. `c2_frozen_micro_hfc_obonuco_ids.csv` gives the frozen set of 127 Obonuco fragments selected as HFC-nearest in MICRO. `null_summary.csv` is generated from the iteration results by `scripts/oaas/summarize_assembly_null.py`.

The observed C1 (0.836438) fell within the random-assembly distribution (mean 0.835106; SD 0.006037; 2.5th–97.5th percentiles 0.823473–0.847287). The observed C2 (117/127) was the Null maximum, occurred as an exact tie in 146/500 realizations, and was never exceeded. These results do not support additional stabilization in C1/C2 beyond this random-assembly baseline. They do not test receiver perception and do not establish that natural vocal assemblies are artifacts.

The repository does not redistribute the source-audio corpus. The released results permit recomputation of the summary. Rerunning the 500 realizations requires authorized local access to the bovine recordings and a local corpus root with `analysis_ready_clean/BOVINE_CORE/`; see `REPRODUCIBILITY.md` for the command. The runner removes temporary generated audio and intermediate feature files after each iteration.
