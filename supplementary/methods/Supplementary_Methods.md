# Supplementary Methods

**Article title:** Affective Information Transmission in Cross-Species Acoustic Communication: Evidence for Multiscale Acoustic–Affective Organization

These methods follow the sequence of the Materials and Methods section and provide the operational definitions, analytical procedures, and model specifications summarized in the main text.

## Research-creation framework and multiscale workflow

The study combined a bovine-referenced acoustic comparison, composition and controlled sound production, and receiver-response evaluation. MICRO and MACRO are independently fitted representations of short-duration vocalizations and temporally integrated acoustic material; their coordinates are not treated as a physical progression from one scale to another. O1 and O2 provide the compositional bases for the FASE analyses, while pre-compositional playback and post-compositional field evaluation are separate tests of receiver responses.



## Acoustic preprocessing, feature extraction, and comparative corpus

The comparative OAAS input manifest and feature table each contain 384 source files: 198 bovine, 42 pastoral-human, 56 non-pastoral-human, 23 human-uncertain, 34 animal-comparison, 29 acoustic-control, and 2 pastoral-variant files. All 384 feature-table rows report the standardized 22,050-Hz analysis rate, and no row-level exclusion flags are present. Supplementary Dataset D2 reports source-file duration, encoding, channel count, sample rate, bit depth, size, and nominal Nyquist limit, extracted from the 384 audio headers. The nominal Nyquist limit is a digital sampling ceiling, not measured acoustic bandwidth. SNR and performer/recordist identity were not consistently available. The feature-extraction pipeline loads mono audio, subtracts the signal mean, trims low-level leading/trailing regions using a 30-dB `librosa` threshold, applies pre-emphasis (coefficient 0.97), and RMS-normalizes to 0.08 with a 0.99 peak limit. These are digital preparation steps for the comparative feature table, not a biological call-quality screen; FASE whole-work, fragment, and transformation analyses use separate analysis scripts.

The bovine subset comprises 11 GC, 59 LFC, 108 HFC, and 20 PSC files. The LFC and HFC subsets (167 files) were selected from the open-access BovineTalk archive of 1,146 dairy-cattle vocalizations; the 11 GC and 20 PSC files are project recordings. Bovine vocalizations were manually reviewed and segmented in Audacity 3.7.8. A call was retained when its full acoustic envelope was present and no contaminating background noise was audible. This was a qualitative listening criterion, without a numeric signal-to-noise cutoff. The available manifest does not preserve an itemized log of excluded files.

### Descriptor coverage and limits

The OAAS feature manifest contains 208 model descriptors. They comprise 10 global duration/amplitude, onset, entropy, and harmonic/percussive fields; seven fundamental-frequency and voicing fields; 120 summaries from 20 MFCCs and their first- and second-order deltas; 14 summaries from seven spectral-contrast bands; 24 summaries from 12 chroma bins; 12 summaries from six tonnetz dimensions; 12 mean/standard-deviation summaries for spectral centroid, bandwidth, 0.85 roll-off, flatness, zero-crossing rate, and framewise RMS; three envelope-modulation fields; and six band-power fields. Band-power values represent mean framewise summed STFT power in 0–1, 1–3, and 3–8 kHz, plus each band's proportion of their combined power. The 22,050-Hz `sample_rate` column is retained as metadata and is not part of the 208-feature vector. Within each coefficient, band, or bin series, frame summaries are the mean and population standard deviation (ddof = 0).

For feature extraction, the FFT length was the largest power of two no greater than 1,024 and no greater than the signal length (minimum 32); hop length was one quarter of the FFT length (minimum 16 samples). pYIN estimates were bounded from C2 to C7. Spectral entropy is the natural-log entropy of the normalized mean power spectrum. The envelope-modulation rate proxy counts sign changes in successive framewise RMS differences and divides by signal duration. The `jitter_proxy` is the mean absolute difference between consecutive finite positive pYIN estimates divided by their mean; it is not a cycle-to-cycle jitter measure and may bridge unvoiced gaps. The extractor uses the installed `librosa` defaults for windowing, centering, and arguments not explicitly set in the script.

This set does not systematically quantify subharmonics, deterministic chaos, frequency jumps, shimmer, or full fundamental-frequency contour shape. Loading contributions indicate distributed contributions across PC1–PC3, but no descriptor-ablation experiment was conducted. The analysis therefore cannot exclude the possibility that some observed correspondences reflect timbral resemblance. The four feature-specific HRV predictors are a separate, prespecified subset and should not be read as a test of the full 208-feature architecture.

## HRV acquisition and quality control

The validated Polar H10 field-monitoring and HRV quality-control workflow was implemented in both the pre-compositional and post-compositional experiments, following the team's harness-standardization protocol (Betancourth-Chaves et al., 2026). RR intervals were acquired at 1,000 Hz. Processing used 5-min segments, a threshold-based artifact filter setting of 0.3, smoothness-priors detrending (lambda = 1,000), manual review of ECG stability, removal of the first and last 5 min, and retention of clean segments with more than 95% valid RR intervals. PRE records were processed in Kubios HRV Scientific Premium v3.5; POST records were processed in Kubios HRV Scientific v4.1. The available derived HRV workbooks do not contain beat-level correction counts or a per-animal exclusion log, so corrected-beat totals and excluded-data amounts cannot be reconstructed for the full cohorts.

## Animals and general setting

The pre-compositional playback study involved 70 lactating Holstein cows. A randomly selected subsample of 20 cows was assigned to four parallel HRV groups of five. The post-compositional field behavior study involved 75 Holstein cows. The physiological analyses used 19 cows: seven in the FASE/Obonuco condition, six in Control, and six in Pink noise. Behavioral observations are group-level records; individual animals were not tracked.

## Bovine reference categories and OAAS construction

For the MACRO bovine reference, five derived 180-s ensembles were generated separately for each category (GC, LFC, HFC, PSC) from the corresponding class-specific source pool. Source recordings were loaded as mono audio at 22,050 Hz and low-level leading and trailing segments were trimmed using a 30-dB threshold. Within each category, source order was shuffled with deterministic class- and ensemble-specific seeds. Recordings were concatenated until the target duration was reached; when a recording exceeded the remaining duration, a randomly selected contiguous segment filled the remainder. The source pool was reshuffled and reused as needed. Each completed ensemble was peak-normalized to 0.99 and written as 16-bit PCM WAV. These are derived acoustic representations from the source recordings, not additional independent biological recordings. The 208 acoustic descriptors were extracted anew from each completed 180-s ensemble; MACRO feature values were not calculated by summing MICRO feature values. The MICRO and MACRO representations were standardized and fitted independently. For feature i, standardization was calculated as follows, with mean_i and standard deviation SD_i estimated from the corresponding bovine reference data:

$$
z_i = \frac{x_i - mean_i}{SD_i}
$$

PCA was fitted to the bovine reference and comparison materials were projected using that scale-specific transformation. Euclidean distances to bovine category centroids were calculated within the applicable representation; quantitative nearest-reference analyses retained PC1–PC10. For a projected point p and category centroid c_k, Euclidean distance across the ten retained components was calculated as follows:

$$
d_k = \sqrt{\sum_{j=1}^{10}(p_j - c_{k,j})^2}
$$

Figure 5A's display uses its separate LNU1–LNU2 coordinates.

Relative positioning between two bovine reference centroids A and B was summarized using the Reference Alignment Index (RAI), where d_A and d_B are the distances from the tested configuration to the respective centroids:

$$
RAI_{A,B} = \frac{d_B - d_A}{d_B + d_A}
$$

Thus, the LFC–HFC Balance is `(d_HFC − d_LFC)/(d_HFC + d_LFC)`, and the GC–PSC Balance is `(d_PSC − d_GC)/(d_PSC + d_GC)`. Each bounded contrast ranges from −1 to 1; positive values indicate relative proximity to LFC or GC, respectively, and negative values indicate relative proximity to HFC or PSC.

### Transcultural, cross-species, and instrumental comparisons

Pastoral and non-pastoral vocal materials, porcine vocalizations, and acoustic controls were compared with the bovine reference categories using the corresponding scale-specific feature-processing and projection procedures. Pastoral classification followed documented cultural context and primary function in livestock coordination or management. Candidate musical-instrument materials were processed through the same applicable pipeline, and Euclidean distances to bovine reference centroids were used as an exploratory aid for source-material selection.

## Behavioral and physiological responses to acoustic stimuli

### Pre-compositional playback experiment

The pre-compositional playback experiment involved 70 lactating Holstein cows. Twenty cows were randomly selected for HRV monitoring and assigned to four parallel groups of five. RR intervals were recorded with Polar H10-based monitoring at 1,000 Hz; HRV and behavioral records were aligned at one-minute resolution. The four HRV groups corresponded to positive (T1), negative (T2), mixed (T3), and control conditions. The T3 master combined stimuli from the positive and negative contextual sets. The 30-min T1–T3 masters were played through a centrally positioned JBL Boombox 4; the PRE methods note reports a mean level of approximately 65 ± 3 dB measured at 10 m and a 24-h interval between treatment exposures. The T1/T2/T3 HRV records were processed in Kubios HRV Scientific Premium v3.5. The physical playback maps establish the stimulus sequence within each assembled master. The available records do not establish counterbalancing of category order across groups or blinded behavioral coding. Minute-level PRE HRV observations were summarized at the treatment level and are not used for stimulus-specific confirmatory inference.

The associated herd-level behavioral summary contains 90 one-minute interval records, with 30 intervals per assembled master (T1–T3). The physical playback maps identify baseline periods as “Descanso / silencio”; the analysis labels them “Baseline (rest, normalized noise floor),” distinct from the separate pink-noise stimulus. Baseline blocks vary in duration and do not constitute a fixed-duration post-stimulus response window. Drone video covered the paddock, and one-minute intervals were classified for body orientation and distance to the loudspeaker using an ethogram and ImageJ-based angle/distance measurements. The source-oriented proportion was computed as follows:

$$
P_{source} = \frac{p1_{mean} + p2_{mean} + p3_{mean}}{herd_{est}}
$$

In the behavioral dataset, `herd_est` equals `d1_mean + d2_mean + d3_mean + d4_mean` in all 90 intervals; the scan-specific denominator ranges from 68 to 72. For each interval, `herd_est` is the estimated number of cattle contributing to the observation, calculated as the sum of counts across d1–d4; it is a scan-specific group denominator, not a fixed cohort count. For Figure 5B, this proportion was multiplied by 60 and displayed on a seconds-per-minute-equivalent scale. These records are group-level interval observations, not counts of discrete glances, individual gaze durations, or event-level measures of post-stimulus latency; individual cow identifiers are not available. PRE and the later Obonuco experiment used the same p and d categories. Source-oriented configurations comprised p1 (body and head aligned toward the source; 0°), p2 (body toward the source; head/neck rotated 30°–90° relative to the body), and p3 (body parallel to the source; 90°, with the head directed toward it). Distance zones were d1 (0–1 m), d2 (1–5 m), d3 (5–12 m), and d4 (>12 m), measured relative to the loudspeaker. PRE observations were recorded at one-minute intervals; the later Obonuco study’s two-minute cadence is specific to that experiment.

### Post-compositional field experiment: behavioral observations

The field cohort comprised 75 Holstein cows. Each control, pink-noise, or Obonuco session lasted 60 min, and sessions were separated by one week. A JBL Boombox 4 was positioned centrally in the paddock; playback level was approximately 65 ± 3 dB, measured with a sound-level meter at 12 m. The speaker remained in place but switched off during Control, and a 30-min adaptation period preceded each session. Behavioral observations were collected in 30 repeated group scans per condition at two-minute intervals. Drone video was recorded at approximately 90 m and 4K/30 fps; ImageJ v1.54f was used for spatial/postural scoring, with a 30-s video window at each scan for locomotion coded in BORIS. The Figure 11 behavioral records contain scan-level counts: cattle in near-source zones d1+d2 (0–5 m), cattle fully oriented toward the source (p1), and cattle observed walking. They do not retain individual identities or coder-level event records; the plotted results are group/session summaries, and repeated scans are not independent animal replicates. The observer repeated coding for a subset, but it does not specify the retested sample, interval, agreement statistic, or acceptance threshold; no quantified reliability estimate is available.

## Functional Affective Sound Environments: composition and derived versions

O1 and O2 are two produced realizations of the same underlying composition. The composition integrates instrumental, environmental, and bovine vocal material; the same processed bovine Grazing Call stem was used in both realizations. Twelve derived versions were produced from the O1 and O2 branches, yielding 14 configurations including the two Base realizations. The three acoustic design branches were OP (GC-directed), ON (PSC-like), and a separate HFC-directed branch. The branch labels refer to their acoustic design targets; receiver responses were evaluated separately in the field experiment.

Musical production was carried out in Ableton Live 12. An Ableton Push 3 controller was used to enter melodic and harmonic lines; the VST plug-ins used were native to Ableton Live 12. Sibelius Ultimate was used to prepare the score.

The environmental layer shared by O1 and O2 originated from a field recording made in a tropical high-Andean pasture while Holstein cows were grazing. It was recorded with a Zoom H6 and a 360-degree microphone fitted with a windscreen. The recorder was mounted on a tripod and operated with a control to minimize handling and equipment-contact noise. The recording included wind, insects, distant water flow, and low-level ruminant activity. During integration, spectrally stable, low-transient material was favored, while prominent or potentially disruptive acoustic events were attenuated. The environmental layer was kept separate for comparison with the complete compositions and other isolated components.

The variant-generation scripts apply loudness normalization and combinations of filtering, spectral equalization, and dynamic compression. Selected high-intensity variants also apply vibrato and tremolo. The variant renders used 44.1-kHz, 16-bit PCM output. Variant-specific input lineage and parameter settings are documented with the analysis scripts and audio files; those signal-processing operations are distinct from the 208-descriptor OAAS analysis. The field-exposed O11 and O12 realizations derive from the O1 and O2 branches, respectively. The analyses of complete works, proportional fragments, and transformed versions are separate representations; fragment durations are not assumed to be equal.

### Whole-work, fragment, transformation, and stem analyses

The complete O1 and O2 works were analyzed from their source WAVs with the 208-descriptor OAAS extractor at a target rate of 22,050 Hz, then projected through the MACRO scaler, PCA, and bovine centroids without refitting. Complete-work analyses were conducted separately from fragment and transformed-version analyses.

Each O1 and O2 Base work was represented by thirteen structurally defined temporal fragments whose durations vary; they are not equal-duration segments. Long transformed versions were analyzed separately from the complete works and fragments.

The component-level comparison uses a separate 41-descriptor workflow. Descriptors were extracted from mono audio resampled to 44,100 Hz, averaged over frames for each stem, and projected through a StandardScaler/PCA model fitted to 195 eligible bovine rows. This stem workflow is distinct from the 208-descriptor MACRO representation used for full works, fragments, and transformed versions.

#### Window-level acoustic-information descriptors for O1 and O2

Each of the 13 source fragments comprising O1 and each of the 13 comprising O2 was divided into complete, non-overlapping 5-s windows. This yielded 276 windows per realization (23 min of analyzed signal each); remaining fragment tails shorter than 5 s were excluded. A short-time Fourier transform (STFT; 2,048-sample segment, 1,024-sample overlap) was applied to each window. Four descriptors were calculated from the resulting magnitude spectrum: (1) ACI-derived spectral–temporal variability, the sum of absolute frame-to-frame magnitude differences divided by total magnitude; (2) Gini-based spectral-energy inequality, the Gini coefficient of magnitude summed across 20 linearly partitioned frequency-index bands; (3) normalized spectral-band difference (NBS), calculated from the summed STFT magnitude (M) in each stated range:

$$
NBS = \frac{M_{2-8\,kHz} - M_{0-2\,kHz}}{M_{2-8\,kHz} + M_{0-2\,kHz}}
$$

The fourth descriptor was 2–8 kHz spectral magnitude, the summed STFT magnitude from 2 kHz to below 8 kHz. The latter three names describe their operational implementations and are not algebraically identical to canonical ecological indices. Window distributions characterize the two composed FASE realizations. Repeated windows were not treated as independent biological replicates in inferential tests.

### Trajectory metrics for Table 2

For each of the 14 compositional configurations in Table 2, the 13 temporally ordered proportional fragments were represented in the MACRO PC1–PC3 coordinates. Trajectory length (TL) is the cumulative Euclidean displacement across consecutive fragments:

$$
TL = \sum_{k=1}^{12} \sqrt{\sum_{j=1}^{3}(x_{k+1,j} - x_{k,j})^2}
$$

For each configuration, let x_{k,j} denote the coordinate of fragment k (k = 1,…,13) on principal component j (j = 1, 2, 3), and let mean_j denote the mean coordinate on component j across the 13 fragments. Trajectory dispersion (TD) is the sum of the three population variances (ddof = 0):

$$
TD = \sum_{j=1}^{3} \left[\frac{1}{13}\sum_{k=1}^{13}(x_{k,j} - mean_j)^2\right]
$$

$$
mean_j = \frac{1}{13}\sum_{k=1}^{13}x_{k,j}
$$

TL and TD summarize the organization of the ordered fragment representation; they are distinct from centroid-distance balances.

## Statistical and computational analyses

No observations were removed from the retained inferential analyses under a statistical outlier rule. Analysis-specific sample sizes reflect the stated cohort, temporal-support, and predictor-matching criteria; these analytic eligibility criteria are distinct from the source-recording quality screen described above.

### Post-compositional HRV: confirmatory condition trajectories

Polar H10-based telemetry was used in the 19-cow physiological cohort (Obonuco *n* = 7, Control *n* = 6, Pink noise *n* = 6). Records were processed in Kubios HRV Scientific v4.1. For Figure 11D, raw RMSSD was estimated from overlapping 300-s windows advanced in 60-s increments. These adjacent windows overlap by 80%; the Figure 11D curves and their across-animal mean ± SD are descriptive.

Confirmatory inference over the common 0–60-min interval used lnRMSSD, with 61 repeated records per cow (1,159 records total). We fitted a Gaussian generalized estimating equation (GEE) with condition, a centered natural cubic spline for time (4 degrees of freedom), and the Condition × Time interaction. Cow was the clustering unit; the working correlation was AR(1), with bias-reduced covariance. The global interaction was assessed jointly with 8 degrees of freedom. Three prespecified pairwise global trajectory contrasts (Obonuco–Control, Pink noise–Control, and Obonuco–Pink noise) were adjusted using Holm's method.

The global interaction was supported (Wald chi-squared(8) = 54.5695, *p* = 5.352 × 10^(-9)); the Obonuco–Pink noise contrast remained supported (Holm *p* = 0.000563), whereas Obonuco–Control and Pink noise–Control did not (Holm *p* = 0.977591 for each).

### Post-compositional HRV: phase transitions and inter-individual coordination

Within the seven-cow FASE/Obonuco cohort, the fixed sequence comprised PRE, O11, an intervening interval, and O12. The prespecified phase-transition family compared the cow-level mean lnRMSSD for the intervening interval with O11 and O12 with the intervening interval. Each contrast used an exact two-sided paired sign-flip test over all 128 possible sign patterns; Holm correction covered the two contrasts. For O12–intervening, the mean paired difference was 0.07082 (SD 0.06312; six cows increased and one decreased; raw *p* = 0.03125; Holm *p* = 0.0625). For intervening–O11, the mean paired difference was 0.03775 (SD 0.07617; five cows increased and two decreased; raw and Holm *p* = 0.328125). The O12–O11 comparison is descriptive, and the fixed order does not isolate an O12-specific effect.

Inter-individual coordination during O11 was evaluated for RMSSD and Stress Index in seven cows (21 animal pairs; 23 records per cow). For each metric, pairwise Spearman correlations were evaluated at lags −2 to +2 on the 300-s HRV windows advanced by 60 s (80% overlap); the primary statistic was the median pairwise maximum positive correlation across that lag range. The null distribution used 10,000 independent circular shifts of animal time series, with the full lag-optimization procedure repeated for each randomization and one animal fixed as the rotation anchor. The observed lag-optimized median correlations were 0.223 for RMSSD and 0.173 for Stress Index; each fell within its empirical null 95% interval (0.079–0.484 and 0.032–0.499, respectively). Upper-tail probabilities were adjusted by Holm across the two metrics. Neither metric exceeded the null expectation (RMSSD Holm *p* = 1.000; Stress Index Holm *p* = 1.000). This analysis does not establish shared affect, a causal mechanism, or physiological latency.

### Post-compositional HRV: acoustic-feature associations

Four prespecified feature-specific analyses examined lnRMSSD in relation to mean spectral flatness (stored as `roughness` in the source table), the adjusted acoustic score A_adj, BPM, and onset density during the Obonuco O1 exposure. The matched analysis set comprised seven cows and 18 repeated records per cow (126 rows). Each predictor was evaluated in a Gaussian GEE clustered by cow, with AR(1) working correlation, bias-reduced covariance, and centered linear time. Holm adjustment was applied across the four primary predictor tests. No lag search or best-lag selection was performed. Since consecutive 300-s HRV windows advance by 60 s, the estimates overlap by 80%; these models evaluate contemporaneous association at that support and do not estimate physiological latency or causal effects.

No feature association was supported. Per 1-SD increase in each predictor, the lnRMSSD coefficient (95% CI) was −0.00658 (−0.02198 to 0.00881) for mean spectral flatness (source-table column `roughness`), 0.01644 (−0.01649 to 0.04937) for A_adj, 0.00310 (−0.01972 to 0.02593) for BPM, and −0.00247 (−0.02062 to 0.01567) for onset density. The corresponding raw *p* values were 0.40198, 0.32782, 0.78991, and 0.78940; all four Holm-adjusted *p* values were 1.000. These are distinct from the descriptive Figure 11D raw-RMSSD curves and from the full-cohort lnRMSSD condition-trajectory model.

### Operational definitions of acoustic predictors

In the O1 predictor table, the column named `roughness` is the mean spectral-flatness value, computed from the magnitude short-time Fourier transform (STFT) using the acoustic-scoring pipeline (22,050-Hz mono audio; 4,096-sample FFT; 512-sample hop; spectral flatness averaged over frames). It is a descriptor of the distribution of spectral energy, not the peak-pair dissonance curve shown in Figure 7D. Figure 7D uses a separate framewise measure: for 50-ms Hann-windowed STFT frames advanced by 25 ms, the 20 strongest spectral peaks are retained and their amplitudes normalized to the strongest peak. For each pair of distinct peaks m and n, let a_m and a_n be their amplitudes normalized to the strongest peak in the frame, and let f_m and f_n be their frequencies in Hz. The pairwise contribution and frequency-separation term are:

$$
D_{mn} = a_m a_n (e^{-3.5x_{mn}} - e^{-5.75x_{mn}})
$$

$$
x_{mn} = \frac{0.24|f_m-f_n|}{0.021\min(f_m,f_n)+19}
$$

Framewise roughness is the sum across the 190 distinct pairs among the 20 retained peaks, with negative contributions truncated at zero:

$$
R_t = \sum_{m<n}\max(D_{mn}, 0)
$$

For the display, the O1 and O2 series are jointly min–max normalized and smoothed with successive moving-average kernels of 151 and 31 samples. The two measures have different definitions and scales and should not be interpreted as interchangeable measurements of one feature.

The predictor A_adj is a constructed acoustic score, not an observation of bovine arousal or affect. The scoring script first transforms each available descriptor into a direction-adjusted deviation from the local pastoral reference median, divided by that reference IQR and clipped to 0–3. For features designated “higher is worse,” the deviation is `(value − local median)/local IQR`; for features designated “lower is worse,” it is `(local median − value)/local IQR`; negative deviations are set to zero. The base score is the logistic transform of a weighted sum of nine direction-adjusted descriptor scores. Here, logistic(u) = 1/(1 + e^(−u)):

$$
A_{base} = \operatorname{logistic}(0.20s_{RMS} + 0.18s_{onset} + 0.15s_{BPM} + 0.12s_{centroid} + 0.10s_{ZCR} + 0.08s_{bandwidth} + 0.07s_{HF/LF} + 0.05s_{tilt} + 0.05s_{SNR})
$$

The adjusted score is clipped to the unit interval [0, 1]:

$$
A_{adj} = \operatorname{clip}(A_{base} + 0.08P_{HF} - 0.06R_{LF}, 0, 1)
$$

For each nonmissing descriptor, `P_HF` increments when the value falls beyond the high-frequency reference IQR in its adverse direction (above Q3 when “higher is worse,” below Q1 when “lower is worse”); `R_LF` increments when the value falls within the low-frequency reference IQR. Both sums are divided by the number of nonmissing descriptors. Predictor values were subsequently time-weighted to the acoustic portions overlapping each HRV support interval. A_adj is reported as an analysis predictor only; it does not validate the affective meaning of the score or establish a receiver state.
