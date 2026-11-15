# Challenge 3: validation (D3) and stress results (D4)

Plan: `plan/Kinjalk-Work-Plan.md`, section 7. Code: `src/uiwue/threed/{stats,envdata,experiments}.py`.
Pipeline and trait definitions: `docs/threed/PIPELINE.md`.

**Status, 2026-11-15 (all 216 plants):** all numbers below are **real**. They come from
`data/derived/threed/traits_runs.parquet`:

- 46,296 runs of all 216 plants (72 per bed), 2025-08-12 to 2025-09-19, all five slots;
- 19,963 of the runs pass QC;
- frame table with 650,388 rows; parity table with 140 runs.

The main analyses compare W1 (control) with W2 (deficit), as the plan says: 144 plants. A
sensitivity run adds W3 to the deficit group (`results/beds_w123/`). The earlier 28-plant results
are kept in `data/derived/threed/results_28plants/` for comparison.

Every number is also in `data/derived/threed/results/summary.md`, which the CLI rewrites on every
run.

## 1. Headline

| Check | Result |
|---|---|
| E0a paper parity | **PASS.** Control +4.30 mm/day (paper +4.30), interaction −13.08 mm/day (paper −13.08); difference 0.02 %. |
| E0b QC agreement (8 Tier-2 plants, 1,680 runs) | `qc_dark`: precision 0.97, recall 0.79 (meets 0.7). `qc_not_centred`: precision 0.37, recall 0.11 (fails). `qc_no_plant`: not testable (1 labelled run). |
| E1 reliable traits (SNR ≥ 3) | **All 12.** `inclination_sd` now passes (SNR 3.6; 2.4 with 28 plants). |
| E2 bed check | Before the cutoff, W2 grew more slowly than W1 in every size trait: heights by 20-24 % of the W1 rate (Holm p < 1e-8), hull volume 19 %, `n_points` 27 %, leaf area 9 % (Holm 0.009) and projected area 8 % (Holm 0.048). Shape traits show no difference. |
| E3 first traits (plan design, 15 h, Holm) | **`n_points`, `leaf_area_est`, `hull_volume` and `projected_area` on day 4 (2025-09-07)**, one day before the first written symptom of the analysed plants. `height_p95` follows on day 5, the other heights on day 7. 57 control and 49 deficit plants. |
| E3 sensitivity (all daytime slots) | The same traits and all three heights on day 4. 70 control and 66 deficit plants. |
| E3 step 3b, lead against teammates | On the same 30 plants, leaf minus air temperature reaches onset on day 4 and the first 3D traits on day 5: **thermal leads 3D by one day**. RGB colour (day 7) comes before any 3D onset on its 18 plants. |
| E4 daily rhythm | No growing midday height dip. With dark runs excluded, the midday dip of `leaf_area_est` deepens by 1.21 cm²/day under deficit (p = 0.005). Deficit plants regain 0.42 mm more median height per day overnight (p = 4e-4, Holm onset day 4). |
| E5 structural proxy for turgor | **`leaf_area_est`, lag +1 day: pooled W2 r = −0.58** (n = 402, p < 1e-16). 8 of 9 W2 and 9 of 9 W3 clip plants have rho −0.24 to −0.86. After each plant's time trend is removed, `height_p50` tracks turgor best (r = −0.41, lag +1). |

**What changed with 216 plants instead of 28.**

- The area-family onset moved from day 6-8 to day 4 in the plan design. With 29-46 plants per
  group per day, the Holm rule is no longer starved of plants.
- E2 now finds a pre-cutoff bed difference in the areas too (8-9 % of the W1 growth rate). With
  28 plants it was invisible. No size trait is free of the bed effect; the areas carry the
  smallest one.
- E4's midday area dip is half or less of the 28-plant estimate. The height overnight
  recovery effect is new.
- E5 is unchanged for W2 and now also covers the 9 W3 clip plants. All 9 show the same negative
  coupling.

## 2. Rerun

```bash
.venv/Scripts/python.exe -m uiwue.threed.experiments --exp E0a E0b E1 E2 E3 E4 E5
.venv/Scripts/python.exe -m uiwue.threed.experiments --exp E1 E2 E3 E4 E5 --beds W1 W2 W3 --out data/derived/threed/results/beds_w123
```

- Inputs: `--runs`, `--frames` and `--parity` default to the three tables in
  `data/derived/threed/`. The environmental files and ADA labels are read from their fixed paths.
- Run time: about 70 s each, on 4 cores.
- The trait table itself comes from `python -m uiwue.threed.run --data-dir data --jobs 12` (about
  31 min for the 176 plants added on 2026-11-15; runs already in the table are skipped).
- Outputs go to `data/derived/threed/results/`:
  - the `schema.RESULT_FILES` CSVs, each with a `source` column set to `real`;
  - `e3_onset.csv`, which also has the `unit` and `onset_day_unadjusted` columns;
  - the secondary tables `e1_noise_cyl.csv`, `e3_onset_daytime.csv`, `e3_onset_cyl.csv` (each
    with its daily tests), `e3_team_leads.csv` (lead time against the thermal and RGB team
    tables; real runs only), `e4_models.csv`, `e4_models_nodark.csv`,
    `e0b_qc_agreement_variants.csv` and `e5_turgor_variants.csv`;
  - the figures (section 6) and `summary.md`.
- `--synthetic` writes a demonstration on synthetic tables to `results/synthetic/` and never
  touches the real results.
- Tests: `.venv/Scripts/python.exe -m pytest -q tests/threed`. All pass. The synthetic tests
  show that each method recovers effects built in on purpose.

## 3. Design facts every result depends on

1. **Treatment is confounded with bed.** W1 is the only control bed, so a bed effect (light,
   position, pot batch) cannot be told apart from the treatment.
   - The guards are each plant's own baseline (mean of 2025-09-01 and 2025-09-02) and E2.
   - E2 finds that W2 already grew more slowly than W1 before the cutoff, in heights, volume,
     point count and, to a smaller degree, area (section 5). Part of every size difference may
     be a bed effect.
   - This limit stays however clean the numbers look.
2. **Frames.** The 15 frames of a run are a 3 x 5 grid of robot positions, not repeat views.
   - The pipeline found and fixed a sign error in ADA's robot-to-depth mapping
     (`docs/threed/PIPELINE.md`). With the fix, frames from different grid columns agree as well
     as frames from one column.
   - The pipeline aggregates over the frames that see the whole plant cylinder (median 7 per run).
   - E1 measures the spread over exactly those frames, so **E1's noise is close to a real
     frame-to-frame noise floor**. It is not inflated by viewpoint.
   - The same statistic within one grid column is 19-28 % smaller (for example `height_p75`:
     2.10 mm against 1.65 mm).
3. **QC thins the daily samples.** In the treatment window 68 % of W1 runs and 63 % of W2 runs
   pass QC (48 % in W3). The main reasons are `qc_overlap` (closed canopy, 16-20 %; 30 % in W3)
   and `qc_high_spread` (17-19 %).
   - E3 also needs the plant's own baseline on 2025-09-01/02. In the plan design (15 h only) this
     leaves 57 control and 49 deficit plants, and 29-46 plants per group on a given day.
   - The daytime sensitivity analysis (section 5, E3) uses the 09, 12, 15 and 18 h slots. It keeps
     70 control and 66 deficit plants, 55-70 per group per day.
4. **Dark runs are measured by height alone.**
   - The pipeline treats a run as dark when the median brightness of its used frames is below 10.
     Its vegetation mask then uses height alone (above 15 mm) instead of ExG plus height.
   - In the treatment window 99 % of 06 h runs are dark, and 44 % of 18 h runs, 15 % of 09 h runs
     and 9 % of 15 h runs. Area and point counts measured by the two rules are not exactly
     comparable.
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
| same model on our `height_p75` above soil, qc_pass runs (87 of 120) | 5.94 / −12.3 | 4.26 to 7.62 / −14.8 to −9.94 | | +38 % / −6 % | not scored |

- The p-values (4.61e-14 and 3.45e-59) equal those in the ADA README, so the parity path
  reproduces the published analysis to rounding.
- Our soil-relative height gives a steeper control slope and a 6 % smaller deficit effect.
  Neither difference is an error:
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
| `qc_dark` | Dark (≥ 8 of 15 RGB frames) | **0.97** | **0.79** | 472 | 384 | met |
| `qc_not_centred` | Not Centered (any frame) | 0.37 | 0.11 | 714 | 211 | not met |
| `qc_no_plant` | no plant | 0 | 0 | 1 | 218 | not testable |

- **Dark.** The flags agree well.
  - Counting "Medium Dark" too gives precision 1.00 but recall 0.58. Our threshold (brightness
    < 10) matches "Dark", not "Medium Dark".
- **Not centred.** The two flags answer different questions.
  - ADA marks 42 % of the labelled runs, mostly because single RGB frames at the edge of the 3 x 5
    grid are off-centre. That is a property of each image's framing.
  - Our flag (12.6 % of these runs) fires when the isolated plant's centroid lies more than 3 cm
    from the plant centre. That is a property of the segmentation.
  - Restricting ADA to depth frames leaves only 14 positive runs (recall 0.14).
- **No plant.** ADA used "no plant" once (W2_G2, 2025-09-16, 09 h, thermal frames only).
  - Across all plants, 95 % of `qc_no_plant` runs fall in August (60 % on or before
    2025-08-20). Those are seedlings that the reviewers did not label.

### E1: noise floor and reliable traits

- **Noise**: the median over qc_pass runs of the frame-to-frame MAD (unscaled), on the frames the
  pipeline used (`used` and `ok`). It is computed over 19,823 runs with a median of 7 frames each.
- **Signal**: |control − deficit| of each plant's change from its own baseline, at 15 h on
  2025-09-14.
- **Rule**: SNR < 3 means "not reliable at this sensor resolution".

| Trait | Unit | Noise | Noise within column | Signal | SNR | Reliable |
|---|---|---|---|---|---|---|
| height_p50 | mm | 1.99 | 1.57 | 83.8 | 42 | yes |
| height_p75 | mm | 2.10 | 1.65 | 90.9 | 43 | yes |
| height_p95 | mm | 2.61 | 2.05 | 101 | 39 | yes |
| projected_area | cm² | 3.41 | 2.61 | 59.5 | 17 | yes |
| leaf_area_est | cm² | 5.97 | 4.58 | 155 | 26 | yes |
| hull_volume | cm³ | 44.5 | 33.7 | 2,140 | 48 | yes |
| canopy_radius | mm | 0.39 | 0.32 | 2.8 | 7.1 | yes |
| inclination_mean | deg | 1.30 | 1.05 | 7.4 | 5.7 | yes |
| inclination_sd | deg | 0.66 | 0.53 | 2.4 | **3.6** | yes (borderline) |
| droop_index | ratio | 0.020 | 0.016 | 0.16 | 8.0 | yes |
| compactness | ratio | 0.024 | 0.019 | 0.26 | 11 | yes |
| n_points | count | 101 | 73 | 3,020 | 30 | yes |

- **cyl_* traits** (all vegetation in the plant cylinder, without clustering;
  `e1_noise_cyl.csv`) have about the same noise as the clustered traits (`cyl_height_p75` 2.16 mm,
  `cyl_projected_area` 4.6 cm²). All five are reliable.
- `canopy_radius` passes, but its signal is only 2.8 mm. The 8.5 cm isolation cylinder caps it,
  so it saturates on large plants.
- The signals are smaller than with 28 plants (for example projected area 59.5 against 96.7 cm²).
  The 28-plant set had only 2 control plants left on 2025-09-14; the larger groups give a less
  extreme difference.

## 5. D4: stress results

### E2: were W1 and W2 alike before the cutoff?

`trait ~ day * bed + (1 | plant)` on 15 h qc_pass runs, 2025-08-20 to 2025-09-02, 144 plants.

| Trait | W2 − W1 slope | 95 % CI | p (raw) | p (Holm over 12 traits) | W1 slope | Share of W1 rate |
|---|---|---|---|---|---|---|
| height_p50 | −1.32 mm/day | −1.74 to −0.91 | 4e-10 | 4e-9 | 5.5 mm/day | 24 % |
| height_p75 | −1.45 mm/day | −1.91 to −0.99 | 6e-10 | 5e-9 | 6.6 mm/day | 22 % |
| height_p95 | −1.87 mm/day | −2.43 to −1.31 | 7e-11 | 7e-10 | 8.0 mm/day | 23 % |
| n_points | −88 /day | −105 to −71 | 5e-24 | 6e-23 | 322 /day | 27 % |
| hull_volume | −38 cm³/day | −50 to −27 | 1e-11 | 1e-10 | 202 cm³/day | 19 % |
| leaf_area_est | −1.56 cm²/day | −2.50 to −0.61 | 0.001 | 0.009 | 16.8 cm²/day | 9 % |
| projected_area | −0.77 cm²/day | −1.33 to −0.20 | 0.008 | 0.048 | 10.0 cm²/day | 8 % |
| radius, inclination, droop, compactness | | | 0.10-0.74 | 0.48-1 | | no warning |

- Before the cutoff, W2 plants grew about a fifth more slowly in height and volume, and about a
  tenth more slowly in area.
- Baseline normalisation removes size differences between plants, not differences in growth
  rate. So every size trait in E3 contains some pre-existing bed difference. The deficit effect
  after the cutoff is several times larger: the E3 slopes are −5 to −10 % of baseline per day.
- The areas carry the smallest pre-existing difference relative to their growth. That is one
  reason the area-family onsets are the more trustworthy result.

### E3: which trait shows stress first?

- Values: % change from the plant's baseline (2025-09-01/02).
- Trend: `value ~ day * treatment + (1 | plant)` over 2025-09-01 to 2025-09-14, reported as the
  deficit − control slope in % of baseline per day.
- Onset: two-sided Mann-Whitney test per day, Holm-corrected over the 12 treatment days. The
  onset is the first of two consecutive significant days.
- Lead time = onset minus the first written symptom of the analysed deficit plants:
  - W2_A8, "wilting", 2025-09-08, day 5 (1 of the 49 analysed W2 plants has a written symptom);
  - the protocol of 2025-09-08 says many W2 plants were wilting that day;
  - symptoms were only scored on 2, 6, 8, 10 and 12 September, so a lead of one day is the
    finest the scoring can resolve.

**Plan design: 15 h runs, 57 control vs 49 deficit plants** (`e3_onset.csv`)

| Rank | Trait | Onset (Holm) | Onset (unadjusted) | Lead vs symptom, days (Holm) | Slope deficit − control, %/day (95 % CI) |
|---|---|---|---|---|---|
| 1 | n_points | **day 4 (09-07)** | day 4 | **−1** | −8.8 (−9.5 to −8.0) |
| 2 | leaf_area_est | **day 4 (09-07)** | day 4 | **−1** | −6.9 (−7.6 to −6.3) |
| 3 | hull_volume | **day 4 (09-07)** | day 4 | **−1** | −9.7 (−10.6 to −8.7) |
| 4 | projected_area | **day 4 (09-07)** | day 4 | **−1** | −5.1 (−5.6 to −4.5) |
| 5 | height_p95 | day 5 (09-08) | day 4 | 0 | −5.8 (−6.4 to −5.1) |
| 6 | compactness | day 6 (09-09) | day 4 | +1 | −3.4 (−3.8 to −3.0) |
| 7 | canopy_radius | day 6 (09-09) | day 6 | +1 | −0.4 (−0.5 to −0.3) |
| 8 | height_p75 | day 7 (09-10) | day 5 | +2 | −5.7 (−6.4 to −5.0) |
| 9 | height_p50 | day 7 (09-10) | day 7 | +2 | −6.5 (−7.3 to −5.7) |
| 10 | inclination_mean | day 10 | day 5 | +5 | −1.2 (−1.5 to −0.9) |
| 11 | inclination_sd | none | day 8 | − | +0.6 (+0.4 to +0.8) |
| 12 | droop_index | none | day 10 | − | −0.8 (−1.3 to −0.5) |

**Sensitivity: daytime slots 09-18 h, plant-day means, 70 control vs 66 deficit plants**
(`e3_onset_daytime.csv`)

| Rank | Trait | Onset (Holm) | Slope, %/day (95 % CI) |
|---|---|---|---|
| − | canopy_radius | day 0 (artefact, see below) | −0.43 (−0.48 to −0.38) |
| 1-8 | height_p95, hull_volume, n_points, leaf_area_est, height_p50, height_p75, projected_area, compactness | **day 4 (09-07)** | −3.0 to −9.0 |
| 9 | inclination_sd | day 7 | +0.8 (+0.7 to +0.9) |
| 10-11 | inclination_mean, droop_index | none (unadjusted day 9-10) | −0.7, −0.9 |

- `canopy_radius` reaches the onset rule on day 0 from a shift of only −0.2 to −0.3 % of
  baseline on days 0-2 (Holm p 0.03-0.04, not significant on day 3). With 65 plants per group,
  the test detects a difference far below any stress effect. The trait is capped by the 8.5 cm
  cylinder, and the shift starts before any stress is possible, so it is not a stress onset.

**Sensitivity: W1 vs W2 + W3, 15 h, 57 control vs 93 deficit plants** (`results/beds_w123/`)

- All size traits and all three heights reach the onset on day 4 (09-07). `n_points` and
  `compactness` are already significant on day 2 without the correction.
- The symptom reference moves to 2025-09-06 (W3_K8, wilting; 27 of the 93 deficit plants have a
  written symptom), so the lead is +1 day.
- W3 lost leaves to RWC sampling, so part of its size loss is sampling, not stress. This is why
  the plan's design leaves W3 out.

**Secondary: cyl_* traits, 15 h** (`e3_onset_cyl.csv`). `cyl_n_points` and `cyl_projected_area`
reach onset on day 4, `cyl_height_p95` on day 5, `cyl_height_p75`/`p50` on day 7. The
cluster-free traits do not detect stress earlier than the clustered ones.

**Reading.**

- The order is the same in all analyses. Point count, leaf area, hull volume and projected area
  (how much canopy the camera sees) separate first, on day 4. The heights follow 1-3 days later in
  the plan design and the same day with every daytime run.
- `droop_index` and the inclination traits separate late or not at all, although E1 rates them
  reliable. At 2-3 mm per pixel, leaf angle carries no early signal here.
- The first traits lead the first written symptom of the analysed plants by one day, the finest
  lead the symptom scoring can show.
- All slopes exclude 0. A significant treatment effect over the window does not depend on the
  onset rule.
- Against the teammates' signals (next subsection), thermal comes first, then 3D, then colour.

### E3 step 3b: lead time against the teammates' signals

- Same rule as E3 (15 h, change from the plant's own baseline, Mann-Whitney per day, Holm, two
  consecutive days), applied to the thermal and RGB team tables and to the 3D traits **on the
  same plants** (`e3_team_leads.csv`). Temperatures and NGRDI use the change in their own units,
  since they cross zero.
- The thermal table covers 30 W1/W2 plants (13 control, 17 deficit), the RGB table 18 (9 + 9).
  No NDVI table is on `main` yet; RGB NGRDI stands in for the visible colour change.

| Teammate signal | Its onset (Holm) | First 3D traits on the same plants (Holm) | 3D minus teammate |
|---|---|---|---|
| `thermal_leaf_air_dt`, leaf minus air | **day 4** | `n_points`, `leaf_area_est` day 5; `hull_volume` day 7 | **+1 day** (thermal first) |
| `thermal_leaf_t`, leaf temperature | day 6 (day 0 unadjusted) | as above | −1 day |
| `thermal_leaf_ref_dt`, leaf minus control-bed leaf | day 0 | as above | not usable (see below) |
| `rgb_ngrdi`, `rgb_green_cover` | day 7 | none with Holm; heights and volume day 8 unadjusted | ≥ +1 day (colour first, or no 3D onset) |

- **Leaf minus air temperature leads the earliest 3D traits by one day.** Stomata close before
  the canopy visibly shrinks, so this order is what the physiology predicts.
- `thermal_leaf_ref_dt` reaches the rule on day 0, a few hours after the last watering. That is
  the bed-level offset of each run (the beds are imaged 10-25 min apart, and the W3 end of the
  greenhouse is warmer; `docs/fusion.md` section 8), not stress. `thermal_leaf_t` shows the same
  day-0 shift without the correction.
- On the 18 RGB plants only 4 deficit plants keep a QC-passing 15 h baseline in the depth table,
  too few for a Holm onset. The colour change (day 7) comes before any 3D onset there.
- The team's own AUC-based onsets (`derived/detect/onsets.parquet`, all deficit plants of W2
  and W3) give the same order: leaf minus air day 3 (robust rule day 4), `depth_leaf_area_est`
  day 5, `rgb_ngrdi` day 6.
- On all 144 W1/W2 plants the 3D traits reach onset on day 4 (above). The thermal table covers
  fewer plants, so the like-for-like lead is the one in this table.

### E4: daily rhythm

- Midday depression = value(15 h) − value(09 h). Overnight recovery = value(06 h, next day) −
  value(18 h).
- Data: qc_pass runs, 2025-08-28 to 2025-09-18, 144 plants.
- Model: `response ~ days_since_cutoff * treatment + VPD + (1 | plant)`.
  - VPD is the 09-15 h local mean of the W1 air sensor, used for both beds (plan).
  - A PAR-adjusted fit changes the significant slopes by less than 2 %.

| Trait | Response | Deficit − control slope (per day) | 95 % CI | p |
|---|---|---|---|---|
| height_p50/75/95 | midday depression | −0.04 / +0.03 / +0.19 mm | span 0 | 0.20-0.83 |
| projected_area | midday depression, all runs | +0.20 cm² | −0.16 to 0.56 | 0.28 |
| projected_area | midday depression, dark runs excluded | −0.39 cm² | −0.80 to 0.03 | 0.067 |
| leaf_area_est | **midday depression, dark runs excluded** | **−1.21 cm²** | −2.05 to −0.37 | **0.005** (Holm onset day 7) |
| height_p50 | **overnight recovery** | **+0.42 mm** | 0.19 to 0.65 | **4e-4** (Holm onset day 4) |
| height_p75 | overnight recovery | +0.50 mm | 0.25 to 0.74 | 7e-5 |
| projected_area | overnight recovery | −1.14 cm² | −1.51 to −0.78 | 9e-10 (Holm onset day 8) |
| n_points | overnight recovery | −42.5 | −56.5 to −28.4 | 3e-9 (Holm onset day 7) |

**Finding.**

- The plan's hypothesis was a growing midday depression in **height**. It is not supported: the
  height slopes are near 0 and not significant.
- What does deepen under deficit is the midday loss of **visible leaf area** (leaves fold or hang
  at midday). This shows only once dark 09 h runs, which use the height-only rule, are left out.
  The effect is half or less of the 28-plant estimate.
- **Height recovers more overnight under deficit**: from 18 h to 06 h the next morning, deficit
  plants regain 0.4-0.5 mm more median height per day of drought than controls. This fits
  turgor lost during the day and regained at night.
  - Height is the trait least affected by the dark-run vegetation rule, since both rules keep the
    tall points. It still compares an always-dark 06 h run with an often-dark 18 h run, so treat
    it as likely, not confirmed.
- The overnight-recovery effects for area and points are negative and significant. The
  vegetation rule changes between the two runs, so they may be artefacts (see the heatmaps,
  `e4_heatmap_*.png`). Treat the area recovery results as unconfirmed.

### E5: structural proxy for leaf turgor

- Clip plants, all processed: W2 A6, D3, G8, H5, K2, L7, M2, O6, R7; W3 B3, B7, D3, G6, I2, K4,
  O6, Q2, Q8; W1 B3, J5, Q6.
- Turgor per run: the median clip reading within ±15 min of the run time, 2025-09-02 to
  2025-09-19.
- Lags −2..+2 days; a positive lag means the turgor reading comes before the trait.
- The clip value rises as turgor falls, so **r < 0 means the trait falls as the leaf loses
  turgor**.

**Best structural proxy (plan rule: largest significant pooled |r|): `leaf_area_est` at lag +1 day.**

- Pooled W2 repeated-measures r = **−0.58** (p < 1e-16, n = 402 runs, 9 plants).
- Per W2 plant, Spearman rho = −0.48 to −0.80 in 8 of 9 plants, all p ≤ 0.002. W2_D3 has
  rho = +0.01 from 21 runs.
- **W3 clip plants** (not in the 28-plant results): rho = −0.24 to −0.86 in all 9; 8 have
  p < 0.03 (W3_Q8: −0.24, p = 0.08). The coupling repeats in the second deficit bed.
- The three W1 control clips give +0.33 (B3, p = 0.015), +0.20 and +0.13: none shows the
  negative coupling of the deficit plants.
- `projected_area`, `n_points` and `hull_volume` come close (r = −0.51 to −0.54 at lag +1). The
  heights do not (r = −0.18 to −0.25 at lag +1).

Sensitivity (pooled W2 r at lag +1, `e5_turgor_variants.csv`):

| Trait | All runs (plan) | Dark runs excluded | 15 h only | Dark excluded, detrended |
|---|---|---|---|---|
| leaf_area_est | −0.58 | −0.60 | −0.62 | −0.15 |
| projected_area | −0.54 | −0.57 | −0.60 | −0.12 |
| n_points | −0.53 | −0.55 | −0.60 | −0.20 |
| hull_volume | −0.51 | −0.53 | −0.55 | −0.14 |
| height_p50 | −0.18 | −0.15 | −0.24 | **−0.41** (p = 4e-11, n = 251) |
| height_p75 | −0.20 | −0.15 | −0.19 | −0.39 |

**Reading.**

- The leaf-area coupling holds in every variant without detrending. It is mainly the shared
  drought trend: area shrinks as turgor falls over days.
- Once each plant's linear time trend is removed, the area correlation drops to about −0.15. The
  **median height** becomes the best tracker of turgor *fluctuations*, at lag +1: low turgor is
  followed a day later by a lower canopy.
- So the area traits are the structural proxy for the *level* of water deficit, and
  `height_p50` is the proxy for its day-to-day *changes*. The E4 overnight height recovery points
  the same way.
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
  W2 already grew more slowly than W1 before the cutoff, in every size trait.
- **E0b covers only the 8 Tier-2 plants.** The not-centred comparison is weak because the ADA
  label and our flag measure different things.
- **QC thins the daily tests** (section 3.3), now to 29-46 plants per group per day in the plan
  design. The daytime sensitivity analysis keeps 55-70.
- **Large groups find tiny differences.** With 50-70 plants per group, a Holm-significant day can
  come from a shift far below any biological effect (`canopy_radius`, day 0). Read the onsets
  together with the slopes.
- **Dark runs use a different vegetation rule** (section 3.4). This affects E4 overnight
  recovery and to a smaller degree E5.
- **`canopy_radius` is capped by the 8.5 cm cylinder.** `qc_overlap` excludes 16-20 % of the
  treatment-window runs in W1/W2: 22 % of W1 and 17 % of W2 runs at 15 h.
- **`inclination_mean` jumps by about +20 % in both beds on 2025-09-03** (`e3_trajectories.png`).
  The cause is not known; it has been reported to the pipeline owner. The trait gives no early
  stress signal in E3.
- **Re-watering after 2025-09-15 is not modelled.** W2 heights rebound by 2025-09-17.

## 9. Deviations from the plan, with reasons

| Plan | Done | Why |
|---|---|---|
| E1 noise from 15 repeat views | MAD over the frames that cover the plant (median 7), plus `noise_within_column` | the frames are a 3 x 5 grid; with the corrected robot mapping the two statistics differ by 19-28 % |
| E1 signal = control − treatment on 2025-09-14 | difference of baseline changes (trait units); raw difference kept as `signal_raw` | removes plant-size offsets that are not treatment |
| E3 onset, Holm only | Holm onset (`onset_day`) plus `onset_day_unadjusted` and `holm_reachable` | QC leaves fewer plants on some days |
| E3 15 h only | 15 h as planned, plus a daytime-slot sensitivity analysis and `cyl_*` traits | the daytime slots keep more plants per day |
| E3 trend scale not fixed | % of baseline per day | slopes comparable across traits |
| E3 beds | W1 against W2, plus W1 against W2 + W3 as a sensitivity run | W3 lost leaves to RWC sampling |
| E3 symptom reference | earliest written symptom among the *analysed* deficit plants, else the bed-wide note | "the first written symptom for those plants" |
| E4 covariates | VPD as planned; PAR as a sensitivity fit; dark runs excluded as a second sensitivity check | the plan's model has only VPD; dark runs change the vegetation rule |
| E5 | as planned (all qc_pass runs), plus no-dark, 15 h and detrended variants | separates the shared drought trend from tracking |
| E0a | ADA window 2025-09-03..14, its 1000-point filter, no qc_pass filter | replicate the published model exactly |
| E0b | all runs, not only qc_pass | the flags themselves are under test |
