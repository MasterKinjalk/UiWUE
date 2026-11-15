# 3D experiments summary (real data)

Generated 2026-11-15 02:19 UTC by `python -m uiwue.threed.experiments`.


## Input

- runs: 3645 rows, 28 plants (W1_A2, W1_A8, W1_B3, W1_B5, W1_E4, W1_G6, W1_J5, W1_K6, W1_Q6, W1_R1, W1_R7, W2_A2, W2_A6, W2_A8, W2_B3, W2_D3, W2_G2, W2_G8, W2_H5, W2_I6, W2_J5, W2_K2, W2_L7, W2_M2, W2_O6, W2_Q6, W2_R1, W2_R7)
- dates 2025-08-24 .. 2025-09-19; slots [6, 9, 12, 15, 18]; qc_pass 2200 of 3645
- frame table: no; parity table: no


## E0a paper parity

| quantity | ours | paper | rel_diff | pass | ci_low | ci_high | p_value | n_rows | n_plants |
|---|---|---|---|---|---|---|---|---|---|
| control_slope_mm_per_day [ours: height_p75 above soil, qc_pass] | 5.34 | 4.3 | 0.242 |  | 3.74 | 6.94 | 6.3e-11 | 87 | 10 |
| interaction_slope_mm_per_day [ours: height_p75 above soil, qc_pass] | -11 | -13.1 | 0.161 |  | -13.2 | -8.74 | 5.12e-22 | 87 | 10 |


## E0b QC agreement

| flag | rule | precision | recall | f1 | n_labelled | n_flagged | n_runs | n_plants |
|---|---|---|---|---|---|---|---|---|
| qc_dark | dark: >= 8 of 15 RGB frames Dark | 0.954 | 0.896 | 0.924 | 347 | 326 | 1044 | 8 |
| qc_dark | dark: any RGB frame Dark | 0.96 | 0.877 | 0.917 | 357 | 326 | 1044 | 8 |
| qc_dark | dark: >= 8 RGB frames Dark or Medium Dark | 1 | 0.632 | 0.774 | 516 | 326 | 1044 | 8 |
| qc_not_centred | not centred: any depth/intensity/RGB/3D frame | 0.481 | 0.111 | 0.18 | 559 | 129 | 1044 | 8 |
| qc_not_centred | not centred: any depth frame | 0 | 0 |  | 8 | 129 | 1044 | 8 |
| qc_not_centred | not centred: >= 3 frames | 0.0853 | 0.0769 | 0.0809 | 143 | 129 | 1044 | 8 |
| qc_no_plant | no plant: any frame | 0 | 0 |  | 1 | 32 | 1044 | 8 |


## E1 noise floor

| trait | unit | noise | noise_within_column | signal | snr | reliable | signal_date | n_runs | frames_per_run |
|---|---|---|---|---|---|---|---|---|---|
| height_p50 | mm | 1.95 |  | 95.9 | 49.2 | True | 2025-09-14 | 2200 | 6 |
| height_p75 | mm | 2.14 |  | 97.3 | 45.6 | True | 2025-09-14 | 2200 | 6 |
| height_p95 | mm | 2.72 |  | 105 | 38.5 | True | 2025-09-14 | 2200 | 6 |
| projected_area | cm2 | 3.26 |  | 96.7 | 29.7 | True | 2025-09-14 | 2200 | 6 |
| leaf_area_est | cm2 | 5.96 |  | 250 | 42 | True | 2025-09-14 | 2200 | 6 |
| hull_volume | cm3 | 46.6 |  | 3e+03 | 64.5 | True | 2025-09-14 | 2200 | 6 |
| canopy_radius | mm | 0.375 |  | 4.76 | 12.7 | True | 2025-09-14 | 2200 | 6 |
| inclination_mean | deg | 1.21 |  | 10 | 8.3 | True | 2025-09-14 | 2200 | 6 |
| inclination_sd | deg | 0.598 |  | 1.48 | 2.48 | False | 2025-09-14 | 2200 | 6 |
| droop_index | ratio | 0.0186 |  | 0.193 | 10.4 | True | 2025-09-14 | 2200 | 6 |
| compactness | ratio | 0.0217 |  | 0.394 | 18.2 | True | 2025-09-14 | 2200 | 6 |
| n_points | count | 104 |  | 5.34e+03 | 51.6 | True | 2025-09-14 | 2200 | 6 |

Noise source: runs table <trait>_mad (no frame table).


## E1 secondary: cyl_* traits (all vegetation in the cylinder, no clustering)

| trait | unit | noise | noise_within_column | signal | snr | reliable | n_runs |
|---|---|---|---|---|---|---|---|
| cyl_height_p50 | mm | 1.94 |  | 95.9 | 49.4 | True | 2200 |
| cyl_height_p75 | mm | 2.09 |  | 66.3 | 31.8 | True | 2200 |
| cyl_height_p95 | mm | 2.97 |  | 63.9 | 21.5 | True | 2200 |
| cyl_projected_area | cm2 | 4.43 |  | 101 | 22.9 | True | 2200 |
| cyl_n_points | count | 112 |  | 4.56e+03 | 40.7 | True | 2200 |


## E2 bed baseline

Traits: E1-reliable traits (11).

| trait | day_bed_coef | ci_low | ci_high | p_value | p_holm | warning | w1_slope | unit | n_plants | first_date | last_date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| height_p50 | -1.69 | -2.91 | -0.475 | 0.00648 | 0.0618 | True | 7.04 | mm/day | 28 | 2025-08-24 | 2025-09-02 |
| height_p75 | -1.92 | -3.3 | -0.547 | 0.00618 | 0.0618 | True | 8.13 | mm/day | 28 | 2025-08-24 | 2025-09-02 |
| height_p95 | -2.14 | -3.72 | -0.557 | 0.00806 | 0.0644 | True | 9.47 | mm/day | 28 | 2025-08-24 | 2025-09-02 |
| projected_area | 0.126 | -1.62 | 1.87 | 0.887 | 1 | False | 9.98 | cm2/day | 28 | 2025-08-24 | 2025-09-02 |
| leaf_area_est | 0.698 | -2.04 | 3.44 | 0.618 | 1 | False | 16.1 | cm2/day | 28 | 2025-08-24 | 2025-09-02 |
| hull_volume | -19.5 | -52.1 | 13 | 0.24 | 0.961 | False | 207 | cm3/day | 28 | 2025-08-24 | 2025-09-02 |
| canopy_radius | 0.544 | 0.000245 | 1.09 | 0.0499 | 0.349 | True | 0.341 | mm/day | 28 | 2025-08-24 | 2025-09-02 |
| inclination_mean | 0.315 | -0.158 | 0.788 | 0.192 | 0.961 | False | 0.229 | deg/day | 28 | 2025-08-24 | 2025-09-02 |
| droop_index | 0.00261 | -0.0077 | 0.0129 | 0.62 | 1 | False | 0.00847 | ratio/day | 28 | 2025-08-24 | 2025-09-02 |
| compactness | -0.00697 | -0.0145 | 0.000539 | 0.0689 | 0.413 | False | 0.0334 | ratio/day | 28 | 2025-08-24 | 2025-09-02 |
| n_points | -75.7 | -123 | -28.8 | 0.00158 | 0.0174 | True | 344 | count/day | 28 | 2025-08-24 | 2025-09-02 |


## E3 stress onset (ranked)

Traits: E1-reliable traits (11).

| rank | trait | onset_day | onset_date | onset_day_unadjusted | lead_vs_symptom_days | lead_unadjusted_days | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit | holm_reachable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | n_points | 6 | 2025-09-09 | 5 | 1 | 0 | -10.8 | -12.7 | -8.9 | 1.07e-28 | 9 | 9 | True |
| 2 | hull_volume | 6 | 2025-09-09 | 5 | 1 | 0 | -12.4 | -14.8 | -9.96 | 1.28e-23 | 9 | 9 | True |
| 3 | leaf_area_est | 8 | 2025-09-11 | 5 | 3 | 0 | -8.96 | -10.7 | -7.25 | 1.43e-24 | 9 | 9 | True |
| 4 | projected_area | 8 | 2025-09-11 | 5 | 3 | 0 | -5.54 | -6.95 | -4.14 | 1.12e-14 | 9 | 9 | True |
| 5 | height_p50 |  |  | 6 |  | 1 | -7.33 | -9.26 | -5.41 | 8.27e-14 | 9 | 9 | True |
| 6 | height_p75 |  |  | 6 |  | 1 | -6.52 | -8.32 | -4.72 | 1.28e-12 | 9 | 9 | True |
| 7 | height_p95 |  |  | 7 |  | 2 | -6.82 | -8.57 | -5.07 | 2.12e-14 | 9 | 9 | True |
| 8 | compactness |  |  | 8 |  | 3 | -3.44 | -4.51 | -2.37 | 2.5e-10 | 9 | 9 | True |
| 9 | inclination_mean |  |  |  |  |  | -2.28 | -3.02 | -1.54 | 1.35e-09 | 9 | 9 | True |
| 10 | canopy_radius |  |  |  |  |  | -0.724 | -1.07 | -0.381 | 3.44e-05 | 9 | 9 | True |
| 11 | droop_index |  |  |  |  |  | -0.774 | -1.78 | 0.229 | 0.13 | 9 | 9 | True |

Symptom reference: 2025-09-08 (Raw Protocol 8.9.25: many plants in beds 2 and 3 wilting).


## E3 sensitivity: daytime slots 09-18 h, plant-day means

| rank | trait | onset_day | onset_day_unadjusted | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | n_points | 4 | 4 | -9.56 | -10.5 | -8.59 | 4.05e-84 | 11 | 13 |
| 2 | leaf_area_est | 4 | 4 | -8.01 | -8.9 | -7.11 | 8.77e-69 | 11 | 13 |
| 3 | projected_area | 5 | 4 | -5.31 | -6.04 | -4.58 | 5.07e-46 | 11 | 13 |
| 4 | hull_volume | 5 | 5 | -10.9 | -11.9 | -9.84 | 1.06e-92 | 11 | 13 |
| 5 | compactness | 5 | 5 | -3.4 | -3.93 | -2.87 | 3.81e-36 | 11 | 13 |
| 6 | height_p95 | 7 | 6 | -7.04 | -7.83 | -6.24 | 1.4e-67 | 11 | 13 |
| 7 | height_p75 | 8 | 7 | -6.84 | -7.66 | -6.02 | 2.93e-60 | 11 | 13 |
| 8 | height_p50 | 9 | 7 | -7.42 | -8.3 | -6.53 | 6.74e-61 | 11 | 13 |
| 9 | inclination_mean |  |  | -1.48 | -1.83 | -1.13 | 9.07e-17 | 11 | 13 |
| 10 | canopy_radius |  |  | -0.511 | -0.665 | -0.356 | 9.62e-11 | 11 | 13 |
| 11 | droop_index |  |  | -0.796 | -1.2 | -0.397 | 9.33e-05 | 11 | 13 |


## E3 secondary: cyl_* traits, 15 h

| rank | trait | onset_day | onset_day_unadjusted | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cyl_n_points | 6 | 4 | -9.81 | -11.4 | -8.26 | 1.36e-35 | 9 | 9 |
| 2 | cyl_projected_area | 6 | 5 | -6.31 | -7.43 | -5.19 | 2.35e-28 | 9 | 9 |
| 3 | cyl_height_p50 |  | 6 | -7.45 | -9.37 | -5.54 | 2.56e-14 | 9 | 9 |
| 4 | cyl_height_p95 |  |  | -5.79 | -7.4 | -4.18 | 1.79e-12 | 9 | 9 |
| 5 | cyl_height_p75 |  |  | -5.8 | -7.56 | -4.04 | 1.13e-10 | 9 | 9 |


## E4 daily rhythm

| trait | response | slope_interaction | ci_low | ci_high | p_value | vpd_coef | vpd_p | slope_interaction_par_adjusted | onset_day | onset_day_unadjusted | unit | slots_present |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| height_p50 | midday_depression |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| height_p50 | overnight_recovery |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| height_p75 | midday_depression |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| height_p75 | overnight_recovery |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| height_p95 | midday_depression |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| height_p95 | overnight_recovery |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| projected_area | midday_depression |  |  |  |  |  |  |  |  |  | cm2 per day | 6,9,12,15,18 |
| projected_area | overnight_recovery |  |  |  |  |  |  |  |  | 7 | cm2 per day | 6,9,12,15,18 |
| leaf_area_est | midday_depression |  |  |  |  |  |  |  |  |  | cm2 per day | 6,9,12,15,18 |
| leaf_area_est | overnight_recovery |  |  |  |  |  |  |  |  | 7 | cm2 per day | 6,9,12,15,18 |
| hull_volume | midday_depression |  |  |  |  |  |  |  |  |  | cm3 per day | 6,9,12,15,18 |
| hull_volume | overnight_recovery |  |  |  |  |  |  |  |  |  | cm3 per day | 6,9,12,15,18 |
| canopy_radius | midday_depression |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| canopy_radius | overnight_recovery |  |  |  |  |  |  |  |  |  | mm per day | 6,9,12,15,18 |
| inclination_mean | midday_depression |  |  |  |  |  |  |  |  |  | deg per day | 6,9,12,15,18 |
| inclination_mean | overnight_recovery |  |  |  |  |  |  |  |  |  | deg per day | 6,9,12,15,18 |
| droop_index | midday_depression |  |  |  |  |  |  |  |  |  | ratio per day | 6,9,12,15,18 |
| droop_index | overnight_recovery |  |  |  |  |  |  |  |  |  | ratio per day | 6,9,12,15,18 |
| compactness | midday_depression |  |  |  |  |  |  |  |  |  | ratio per day | 6,9,12,15,18 |
| compactness | overnight_recovery |  |  |  |  |  |  |  |  |  | ratio per day | 6,9,12,15,18 |
| n_points | midday_depression |  |  |  |  |  |  |  |  |  | count per day | 6,9,12,15,18 |
| n_points | overnight_recovery |  |  |  |  |  |  |  | 9 | 7 | count per day | 6,9,12,15,18 |


## E4 sensitivity: dark runs excluded (midday depression only)

| trait | slope_interaction | ci_low | ci_high | p_value | onset_day | onset_day_unadjusted | unit | n_plants |
|---|---|---|---|---|---|---|---|---|
| height_p50 |  |  |  |  |  |  | mm per day | 0 |
| height_p75 |  |  |  |  |  |  | mm per day | 0 |
| height_p95 |  |  |  |  |  |  | mm per day | 0 |
| projected_area |  |  |  |  |  |  | cm2 per day | 0 |
| leaf_area_est |  |  |  |  |  |  | cm2 per day | 0 |
| hull_volume |  |  |  |  |  |  | cm3 per day | 0 |
| canopy_radius |  |  |  |  |  |  | mm per day | 0 |
| inclination_mean |  |  |  |  |  |  | deg per day | 0 |
| droop_index |  |  |  |  |  |  | ratio per day | 0 |
| compactness |  |  |  |  |  |  | ratio per day | 0 |
| n_points |  |  |  |  |  |  | count per day | 0 |


## E5 turgor

_skipped: no clip plant in the table_
