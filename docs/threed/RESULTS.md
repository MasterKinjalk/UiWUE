# Challenge 3: validation (D3) and stress results (D4)

Plan: `plan/Kinjalk-Work-Plan.md`, section 7. Code: `src/uiwue/threed/{stats,envdata,experiments}.py`.
Pipeline and trait definitions: `docs/threed/PIPELINE.md`.

**Status, 2026-11-15:** all numbers below are **real**. They come from
`data/derived/threed/traits_runs.parquet`:

- 5,970 runs of 28 plants (W1: 11 control, W2: 17 deficit), 2025-08-12 to 2025-09-19, all five
  slots;
- 2,615 of the runs pass QC;
- frame table with 79,667 rows; parity table with 140 runs.

Every number is also in `data/derived/threed/results/summary.md`, which the CLI rewrites on every
run.

## 1. Headline

| Check | Result |
|---|---|
| E0a paper parity | **PASS.** Control +4.30 mm/day (paper +4.30), interaction −13.08 mm/day (paper −13.08); difference 0.02 %. |
| E0b QC agreement (8 Tier-2 plants, 1,680 runs) | `qc_dark`: precision 0.96, recall 0.79 (meets 0.7). `qc_not_centred`: precision 0.37, recall 0.11 (fails). `qc_no_plant`: not testable (1 labelled run). |
| E1 reliable traits (SNR ≥ 3) | 11 of 12. Only `inclination_sd` fails (SNR 2.4). |
| E2 bed check | W2 heights grew 1.1-1.6 mm/day slower than W1 before the cutoff (raw p 0.008-0.019; Holm over traits 0.08-0.15). Height and `n_points` results carry this warning. |
| E3 first traits (plan design, 15 h, Holm) | `n_points` and `hull_volume` on day 6 (2025-09-09), then `leaf_area_est` and `projected_area` on day 8. The heights reach significance only without the Holm correction (day 6-7). |
| E3 sensitivity (all daytime slots) | `n_points` and `leaf_area_est` on day 4 (2025-09-07), one day before the first written wilting note. Heights follow on day 7-8. |
| E4 daily rhythm | No growing midday height dip. With dark runs excluded, the midday dip of `projected_area` deepens by 1.06 cm²/day (p = 0.018) under deficit, and that of `leaf_area_est` by 2.54 cm²/day (p = 0.004). |
| E5 structural proxy for turgor | **`leaf_area_est`, lag +1 day: pooled W2 r = −0.59** (n = 396, p < 1e-16); 8 of 9 W2 clip plants have rho −0.39 to −0.78. After each plant's time trend is removed, `height_p50` tracks turgor best (r = −0.40, lag +1). |

## 2. Rerun

```bash
.venv/Scripts/python.exe -m uiwue.threed.experiments --exp E0a E0b E1 E2 E3 E4 E5
```

- Inputs: `--runs`, `--frames` and `--parity` default to the three tables in
  `data/derived/threed/`. The environmental files and ADA labels are read from their fixed paths.
- Run time: about 25 s on 4 cores.
- Outputs go to `data/derived/threed/results/`:
  - the `schema.RESULT_FILES` CSVs, each with a `source` column set to `real`;
  - `e3_onset.csv`, which also has the `unit` and `onset_day_unadjusted` columns;
  - the secondary tables `e1_noise_cyl.csv`, `e3_onset_daytime.csv`, `e3_onset_cyl.csv` (each
    with its daily tests), `e4_models.csv`, `e4_models_nodark.csv`,
    `e0b_qc_agreement_variants.csv` and `e5_turgor_variants.csv`;
  - the figures (section 6) and `summary.md`.
- `--synthetic` writes a demonstration on synthetic tables to `results/synthetic/` and never
  touches the real results.
- Tests: `.venv/Scripts/python.exe -m pytest -q tests/threed`. All 77 pass; 33 of them are the
  statistics and environmental-data tests. The synthetic tests show that each method recovers
  effects built in on purpose.

## 3. Design facts every result depends on

1. **Treatment is confounded with bed.** W1 is the only control bed, so a bed effect (light,
   position, pot batch) cannot be told apart from the treatment.
   - The guards are each plant's own baseline (mean of 2025-09-01 and 2025-09-02) and E2.
   - E2 finds that W2 heights already grew more slowly before the cutoff (section 5). Part of the
     height difference may be a bed effect.
   - This limit stays however clean the numbers look.
2. **Frames.** The 15 frames of a run are a 3 x 5 grid of robot positions, not repeat views.
   - The pipeline found and fixed a sign error in ADA's robot-to-depth mapping
     (`docs/threed/PIPELINE.md`). With the fix, frames from different grid columns agree as well
     as frames from one column.
   - The pipeline aggregates over the frames that see the whole plant cylinder (median 6 per run).
   - E1 measures the spread over exactly those frames, so **E1's noise is close to a real
     frame-to-frame noise floor**. It is not inflated by viewpoint.
   - The same statistic within one grid column is only 16-28 % smaller (for example `height_p75`:
     2.08 mm against 1.74 mm).
3. **QC thins the daily samples.** In the treatment window only 53-64 % of runs pass QC. The main
   reasons are `qc_overlap` (closed canopy, 22-29 %) and `qc_high_spread`.
   - E3 also needs the plant's own baseline on 2025-09-01/02. In the plan design (15 h only) this
     leaves 9 control and 9 deficit plants, and 2-9 plants per group on a given day. On
     2025-09-14 only 2 control plants are left.
   - The daytime sensitivity analysis (section 5, E3) uses the 09, 12, 15 and 18 h slots. It keeps
     11 control and 13 deficit plants, 9-12 per group per day.
4. **Dark runs are measured by height alone.**
   - The pipeline treats a run as dark when the median brightness of its used frames is below 10.
     Its vegetation mask then uses height alone (above 15 mm) instead of ExG plus height.
   - In the treatment window every 06 h run is dark, and 42 % of 18 h runs and 19 % of 09 h runs
     are dark. Area and point counts measured by the two rules are not exactly comparable.
   - This matters for E4 (overnight recovery compares 06 h with 18 h) and partly for E5. Both
     have a "dark runs excluded" sensitivity check.

## 4. D3: validation

### E0a: paper parity (PASS)

**Method.** The ADA model (`analysis/plant_height/step_04_fit_lmm.py`) is replicated exactly:

- Data: W1/W2 Tier-1 plants, folder hour 15, frame 7, 2025-09-03 to 2025-09-14, `n_veg` ≥ 1000.
  120 runs of 10 plants; none fall below the 1000-point threshold.
- Height = `0.60 − z_p25`, in metres. Day counted from the first retained date.
- Model: `height ~ day * treatment + (1 | plant)`, fitted by ML with L-BFGS.
- Pass rule: both slopes within 20 % of the paper.

| Quantity | Ours | 95 % CI | Paper | Rel. diff | Pass |
|---|---|---|---|---|---|
| control slope (mm/day) | 4.300 | 3.18 to 5.42 | 4.30 | +0.02 % | yes |
| interaction W2 − W1 (mm/day) | −13.083 | −14.7 to −11.5 | −13.08 | −0.02 % | yes |
| same model on our `height_p75` above soil, qc_pass runs (87 of 120) | 5.34 / −11.0 | 3.74 to 6.94 / −13.2 to −8.74 | | +24 % / −16 % | not scored |

- The p-values (4.61e-14 and 3.45e-59) equal those in the ADA README, so the parity path
  reproduces the published analysis to rounding.
- Our soil-relative height gives a 24 % steeper control slope and a 16 % smaller deficit
  effect. Neither difference is an error:
  - our height is measured from the fitted soil plane, not from a fixed 0.60 m;
  - only the isolated centre plant counts, so neighbours are excluded;
  - only qc_pass runs enter.

### E0b: QC agreement with the ADA hand labels

- Data: 1,680 runs of the 8 Tier-2 plants appear in both our table and the ADA labels. E0b
  covers only these plants.
- The ADA labels are per frame and are reduced to one label per run.
  - Main rules: "Dark" on at least 8 of 15 RGB frames; "Not Centered" on any depth, intensity,
    RGB or 3D frame; "no plant" on any frame.
  - Other rules are in `e0b_qc_agreement_variants.csv`.

| Our flag | ADA label | Precision | Recall | ADA positives | Our flags | Target recall 0.7 |
|---|---|---|---|---|---|---|
| `qc_dark` | Dark (≥ 8 of 15 RGB frames) | **0.96** | **0.79** | 472 | 387 | met |
| `qc_not_centred` | Not Centered (any frame) | 0.37 | 0.11 | 714 | 214 | not met |
| `qc_no_plant` | no plant | 0 | 0 | 1 | 224 | not testable |

- **Dark.** The flags agree well.
  - Counting "Medium Dark" too gives precision 1.00 but recall 0.58. Our threshold (brightness
    < 10) matches "Dark", not "Medium Dark".
- **Not centred.** The two flags answer different questions.
  - ADA marks 42 % of the labelled runs, mostly because single RGB frames at the edge of the 3 x 5
    grid are off-centre. That is a property of each image's framing.
  - Our flag (12.7 % of these runs) fires when the isolated plant's centroid lies more than 3 cm
    from the plant centre. That is a property of the segmentation.
  - Restricting ADA to depth frames leaves only 14 positive runs (recall 0.07).
- **No plant.** ADA used "no plant" once (W2_G2, 2025-09-16, 09 h, thermal frames only).
  - 62 % of our 224 `qc_no_plant` runs fall on or before 2025-08-20. Those are seedlings that the
    reviewers did not label.

### E1: noise floor and reliable traits

- **Noise**: the median over qc_pass runs of the frame-to-frame MAD (unscaled), on the frames the
  pipeline used (`used` and `ok`). It is computed over 2,597 runs with a median of 6 frames each.
- **Signal**: |control − deficit| of each plant's change from its own baseline, at 15 h on
  2025-09-14 (2 control and 7 deficit plants that day).
- **Rule**: SNR < 3 means "not reliable at this sensor resolution".

| Trait | Unit | Noise | Noise within column | Signal | SNR | Reliable |
|---|---|---|---|---|---|---|
| height_p50 | mm | 1.88 | 1.53 | 95.9 | 51 | yes |
| height_p75 | mm | 2.08 | 1.74 | 97.3 | 47 | yes |
| height_p95 | mm | 2.65 | 2.18 | 105 | 39 | yes |
| projected_area | cm² | 3.08 | 2.45 | 96.7 | 31 | yes |
| leaf_area_est | cm² | 5.48 | 4.38 | 250 | 46 | yes |
| hull_volume | cm³ | 42.7 | 34.7 | 3,000 | 71 | yes |
| canopy_radius | mm | 0.41 | 0.33 | 4.8 | 12 | yes |
| inclination_mean | deg | 1.26 | 1.02 | 10.0 | 7.9 | yes |
| inclination_sd | deg | 0.63 | 0.52 | 1.5 | **2.4** | **no** |
| droop_index | ratio | 0.019 | 0.016 | 0.19 | 10 | yes |
| compactness | ratio | 0.022 | 0.018 | 0.39 | 18 | yes |
| n_points | count | 92 | 67 | 5,340 | 58 | yes |

- **cyl_* traits** (all vegetation in the plant cylinder, without clustering;
  `e1_noise_cyl.csv`) have about the same noise as the clustered traits (`cyl_height_p75` 2.03 mm,
  `cyl_projected_area` 4.2 cm²). All five are reliable.
- `canopy_radius` passes, but its signal is only 4.8 mm. The 8.5 cm isolation cylinder caps it,
  so it saturates on large plants.

## 5. D4: stress results

### E2: were W1 and W2 alike before the cutoff?

`trait ~ day * bed + (1 | plant)` on 15 h qc_pass runs, 2025-08-20 to 2025-09-02, 28 plants.

| Trait | W2 − W1 slope | 95 % CI | p (raw) | p (Holm over 11 traits) | W1 slope | Warning |
|---|---|---|---|---|---|---|
| height_p50 | −1.07 mm/day | −1.97 to −0.18 | 0.019 | 0.15 | 5.0 mm/day | yes |
| height_p75 | −1.36 mm/day | −2.35 to −0.36 | 0.008 | 0.08 | 6.1 mm/day | yes |
| height_p95 | −1.56 mm/day | −2.70 to −0.42 | 0.008 | 0.08 | 7.3 mm/day | yes |
| n_points | −43.7 /day | −76.6 to −10.7 | 0.009 | 0.08 | 277 /day | yes |
| areas, volume, radius, inclination, droop, compactness | | | 0.13-0.81 | | | no |

- Before the cutoff, W2 heights grew about 20 % more slowly than W1 heights.
- Baseline normalisation removes size differences between plants, not differences in growth
  rate. So the height results in E3 contain up to about 1.4 mm/day of pre-existing bed
  difference. The deficit effect after the cutoff is several times larger (W2 heights fall from
  2025-09-07 on).
- The areas and the volume show no pre-cutoff difference. That is one reason the
  area-family onsets are the more trustworthy result.

### E3: which trait shows stress first?

- Values: % change from the plant's baseline (2025-09-01/02).
- Trend: `value ~ day * treatment + (1 | plant)` over 2025-09-01 to 2025-09-14, reported as the
  deficit − control slope in % of baseline per day.
- Onset: two-sided Mann-Whitney test per day, Holm-corrected over the 12 treatment days. The
  onset is the first of two consecutive significant days. Holm is reachable with these group
  sizes.
- Lead time = onset minus the first written symptom of the analysed deficit plants:
  - W2_A8, "wilting", 2025-09-08, day 5;
  - the protocol of 2025-09-08 says many W2 plants were wilting that day;
  - symptoms were only scored on 2, 6, 8, 10 and 12 September.

**Plan design: 15 h runs, 9 control vs 9 deficit plants** (`e3_onset.csv`)

| Rank | Trait | Onset (Holm) | Onset (unadjusted) | Lead vs symptom, days (Holm / unadj.) | Slope deficit − control, %/day (95 % CI) |
|---|---|---|---|---|---|
| 1 | n_points | day 6 (09-09) | day 5 | +1 / 0 | −11.1 (−12.9 to −9.3) |
| 2 | hull_volume | day 6 (09-09) | day 5 | +1 / 0 | −12.7 (−15.0 to −10.4) |
| 3 | leaf_area_est | day 8 (09-11) | day 5 | +3 / 0 | −9.2 (−10.9 to −7.6) |
| 4 | projected_area | day 8 (09-11) | day 5 | +3 / 0 | −5.8 (−7.1 to −4.4) |
| 5 | height_p50 | none | day 6 | − / +1 | −7.4 (−9.3 to −5.5) |
| 6 | height_p75 | none | day 6 | − / +1 | −6.6 (−8.3 to −4.8) |
| 7 | height_p95 | none | day 7 | − / +2 | −6.9 (−8.6 to −5.2) |
| 8 | compactness | none | day 8 | − / +3 | −3.6 (−4.6 to −2.5) |
| 9 | inclination_mean | none | none | − | −2.2 (−2.9 to −1.5) |
| 10 | canopy_radius | none | none | − | −0.7 (−1.1 to −0.4) |
| 11 | droop_index | none | none | − | −0.7 (−1.7 to 0.2), p = 0.14 |

**Sensitivity: daytime slots 09-18 h, plant-day means, 11 control vs 13 deficit plants**
(`e3_onset_daytime.csv`)

| Rank | Trait | Onset (Holm) | Onset (unadjusted) | Lead vs symptom (Holm) | Slope, %/day (95 % CI) |
|---|---|---|---|---|---|
| 1 | n_points | **day 4 (09-07)** | day 4 | **−1** | −9.5 (−10.5 to −8.6) |
| 2 | leaf_area_est | **day 4 (09-07)** | day 4 | **−1** | −7.9 (−8.8 to −7.1) |
| 3 | hull_volume | day 5 (09-08) | day 5 | 0 | −10.7 (−11.7 to −9.7) |
| 4 | projected_area | day 6 (09-09) | day 4 | +1 | −5.3 (−6.0 to −4.6) |
| 5 | height_p95 | day 7 | day 6 | +2 | −6.8 (−7.6 to −6.1) |
| 6 | height_p50 | day 7 | day 7 | +2 | −7.2 (−8.1 to −6.4) |
| 7 | height_p75 | day 8 | day 6 | +3 | −6.6 (−7.4 to −5.8) |
| 8 | compactness | day 10 | day 5 | +5 | −3.4 (−3.9 to −2.9) |
| 9 | inclination_mean | none | day 9 | − | −1.4 (−1.8 to −1.1) |
| 10-11 | canopy_radius, droop_index | none | none | − | −0.5, −0.8 |

**Secondary: cyl_* traits, 15 h** (`e3_onset_cyl.csv`). `cyl_n_points` and `cyl_projected_area`
reach onset on day 6. `cyl_height_p50` reaches it only without the correction (day 6), and
`cyl_height_p75`/`p95` show no onset. The cluster-free traits do not detect stress earlier than
the clustered ones.

**Reading.**

- The order is the same in all three analyses. Point count, leaf area and hull volume
  (how much canopy the camera sees) separate first. The heights follow 2-3 days later, after the
  wilting of 2025-09-07/08.
- `droop_index` and the inclination traits do not separate the groups, although E1 rates them
  reliable. At 2-3 mm per pixel, leaf angle carries no early signal here.
- With every daytime run used, the first traits lead the first written symptom by one day.
  With 15 h runs alone, QC leaves too few plants per day, and the onset comes 1-3 days after
  the symptom.
- All slopes are negative and their CIs exclude 0 (except `droop_index`). A significant
  treatment effect over the window does not depend on the onset rule.
- Lead times against teammates' NDVI and leaf-air temperature onsets (plan E3, step 3b) are not
  computed: those tables were not available.

### E4: daily rhythm

- Midday depression = value(15 h) − value(09 h). Overnight recovery = value(06 h, next day) −
  value(18 h).
- Data: qc_pass runs, 2025-08-28 to 2025-09-18, 28 plants.
- Model: `response ~ days_since_cutoff * treatment + VPD + (1 | plant)`.
  - VPD is the 09-15 h local mean of the W1 air sensor, used for both beds (plan).
  - A PAR-adjusted fit changes the significant slopes by less than 2 %. It moves only
    non-significant slopes by more.

| Trait | Response | Deficit − control slope (per day) | 95 % CI | p |
|---|---|---|---|---|
| height_p50/75/95 | midday depression | +0.26 / +0.37 / +0.50 mm | span 0 | 0.21-0.33 |
| projected_area | midday depression, all runs | −0.60 cm² | −1.40 to 0.20 | 0.14 |
| projected_area | **midday depression, dark runs excluded** | **−1.06 cm²** | −1.93 to −0.19 | **0.018** |
| leaf_area_est | **midday depression, dark runs excluded** | **−2.54 cm²** | −4.26 to −0.82 | **0.004** |
| projected_area | overnight recovery | −1.05 cm² | −1.90 to −0.21 | 0.015 |
| n_points | overnight recovery | −50.8 | −83.3 to −18.3 | 0.002 (Holm onset day 9) |

**Finding.**

- The plan's hypothesis was a growing midday depression in **height**. It is not supported: the
  height slopes are positive and not significant.
- What does deepen under deficit is the midday loss of **visible leaf area** (leaves fold or hang
  at midday). This shows only once dark 09 h runs, which use the height-only rule, are left out.
  No daily onset passes the Holm rule.
- The overnight-recovery effects for area and points are significant. They compare a 06 h run
  that is always dark with a 18 h run that is often dark, so the change in vegetation rule may
  drive them.
- The heatmaps (`e4_heatmap_*.png`) show this artefact. After 2025-09-09, W1 area at 18 h sits
  15-20 % below the plant's daily mean, and 40 % of these 18 h runs are dark. Treat the recovery
  results as unconfirmed.

### E5: structural proxy for leaf turgor

- Clip plants, all processed: W2 R7, O6, M2, L7, K2, H5, G8, D3, A6 and W1 Q6, J5, B3.
- Turgor per run: the median clip reading within ±15 min of the run time, 2025-09-02 to
  2025-09-19.
- Lags −2..+2 days; a positive lag means the turgor reading comes before the trait.
- The clip value rises as turgor falls, so **r < 0 means the trait falls as the leaf loses
  turgor**.

**Best structural proxy (plan rule: largest significant pooled |r|): `leaf_area_est` at lag +1 day.**

- Pooled W2 repeated-measures r = **−0.59** (p < 1e-16, n = 396 runs, 9 plants).
- Per W2 plant, Spearman rho = −0.39 to −0.78 in 8 of 9 plants, all p ≤ 0.011. W2_D3 has
  rho = +0.05 from 18 runs.
- The three W1 control clips give +0.36 (B3, p = 0.006), +0.16 and +0.06: none shows the
  negative coupling of the deficit plants.
- `projected_area`, `n_points` and `hull_volume` come close (r = −0.52 to −0.56 at lag +1). The
  heights do not (r = −0.16 to −0.22 at lag +1).

Sensitivity (pooled W2 r at lag +1, `e5_turgor_variants.csv`):

| Trait | All runs (plan) | Dark runs excluded | 15 h only | Dark excluded, detrended |
|---|---|---|---|---|
| leaf_area_est | −0.59 | −0.65 | −0.63 | −0.16 |
| projected_area | −0.56 | −0.61 | −0.61 | −0.12 |
| n_points | −0.54 | −0.60 | −0.62 | −0.21 |
| hull_volume | −0.52 | −0.57 | −0.57 | −0.16 |
| height_p50 | −0.16 | −0.17 | −0.23 | **−0.40** (p = 1e-10, n = 244) |
| height_p75 | −0.17 | −0.15 | −0.20 | −0.37 |

**Reading.**

- The leaf-area coupling holds in every variant without detrending. It is mainly the shared
  drought trend: area shrinks as turgor falls over days.
- Once each plant's linear time trend is removed, the area correlation drops to about −0.15. The
  **median height** becomes the best tracker of turgor *fluctuations*, at lag +1: low turgor is
  followed a day later by a lower canopy.
- So the area traits are the structural proxy for the *level* of water deficit, and
  `height_p50` is the proxy for its day-to-day *changes*.
- The linear detrend does not fully remove the rise-and-recovery shape around re-watering, so
  the detrended values are a sensitivity check, not a primary result.

## 6. Figures

| Figure | File (in `data/derived/threed/results/`) |
|---|---|
| E0a parity trajectories | `e0a_parity.png` |
| E1 signal-to-noise per trait | `e1_snr.png` |
| E3 trajectories, mean ± 1 SD per group, plants as thin lines, cutoff / re-water / onset marked | `e3_trajectories.png`, `e3_trajectories_daytime.png` |
| E3 ranked onset and interaction slope with CI | `e3_onset.png` |
| E4 day x hour heatmaps per bed | `e4_heatmap_height_p75.png`, `e4_heatmap_droop_index.png`, `e4_heatmap_projected_area.png` |
| E5 best proxy against turgor per clip plant; pooled r by trait and lag | `e5_turgor_best.png`, `e5_turgor_lags.png` |

## 7. Environmental data: checked facts

- **Air sensor 1 is W3 and sensor 2 is W1**, as the Sensor Guide says.
  - Until 2025-09-02, sensor 1 reads 0.4-1.7 °C cooler and 0.3-2.4 %RH more humid at midday.
  - From 2025-09-03 to 2025-09-15 it reads 1.7-4.4 °C warmer and 1.9-12.0 %RH drier. The gap
    closes after re-watering.
  - VPD at W3 is 0.15-0.70 kPa higher than at W1 after the cutoff. W2 has no sensor, so the W1
    VPD used for W2 in E4 underestimates it.
- **The PAR logger clock is CEST, not UTC.** The protocol says the sensor was moved at 15:30 on
  2025-09-24, and the values drop between 15:32 and 15:47.
- **Leaf clips: sensors 1-9 are in W3, 10-18 in W2 and 19-21 in W1.** 99 of 294,247 readings
  are drop-outs or spikes and are removed. From 2025-09-03 to 2025-09-14 the W2 clips rise by
  +5.9 to +26.5 units and the W1 clips by +1.7 to +2.5.
- **Written drought symptoms:** 41 plants have one. 39 are in W3, 2 in W2 (A6 on 09-06, A8 on
  09-08) and none in W1.

## 8. Caveats

- **One control bed.** Every treatment contrast is also a bed contrast (section 3.1). E2 shows
  W2 heights already grew more slowly before the cutoff.
- **E0b covers only the 8 Tier-2 plants.** The not-centred comparison is weak because the ADA
  label and our flag measure different things.
- **QC thins the daily tests** (section 3.3). The plan's 15 h design has 2-9 plants per group
  per day and loses plants without a 15 h baseline. The daytime sensitivity analysis is the
  better-powered onset estimate.
- **Dark runs use a different vegetation rule** (section 3.4). This affects E4 overnight
  recovery and to a smaller degree E5.
- **`canopy_radius` is capped by the 8.5 cm cylinder.** `qc_overlap` excludes 22-29 % of the
  treatment-window runs: 21 % of W1 and 28 % of W2 runs at 15 h.
- **`inclination_mean` jumps by about +20 % in both beds on 2025-09-03** (`e3_trajectories.png`).
  The cause is not known; it has been reported to the pipeline owner. The trait gives no stress
  signal in E3.
- **Re-watering after 2025-09-15 is not modelled.** W2 heights rebound by 2025-09-17.

## 9. Deviations from the plan, with reasons

| Plan | Done | Why |
|---|---|---|
| E1 noise from 15 repeat views | MAD over the frames that cover the plant (median 6), plus `noise_within_column` | the frames are a 3 x 5 grid; with the corrected robot mapping the two statistics differ by 16-28 % |
| E1 signal = control − treatment on 2025-09-14 | difference of baseline changes (trait units); raw difference kept as `signal_raw` | removes plant-size offsets that are not treatment |
| E3 onset, Holm only | Holm onset (`onset_day`) plus `onset_day_unadjusted` and `holm_reachable` | QC leaves few plants on some days |
| E3 15 h only | 15 h as planned, plus a daytime-slot sensitivity analysis and `cyl_*` traits | 15 h alone keeps 9 + 9 plants after the baseline requirement |
| E3 trend scale not fixed | % of baseline per day | slopes comparable across traits |
| E3 beds | W1 against W2 | W3 lost leaves to RWC sampling |
| E3 symptom reference | earliest written symptom among the *analysed* deficit plants, else the bed-wide note | "the first written symptom for those plants" |
| E4 covariates | VPD as planned; PAR as a sensitivity fit; dark runs excluded as a second sensitivity check | the plan's model has only VPD; dark runs change the vegetation rule |
| E5 | as planned (all qc_pass runs), plus no-dark, 15 h and detrended variants | separates the shared drought trend from tracking |
| E0a | ADA window 2025-09-03..14, its 1000-point filter, no qc_pass filter | replicate the published model exactly |
| E0b | all runs, not only qc_pass | the flags themselves are under test |
