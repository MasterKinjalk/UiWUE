# 3D experiments summary (real data)

Generated 2026-11-15 06:59 UTC by `python -m uiwue.threed.experiments`.


## Input

- runs: 28080 rows, 216 plants (W1_A2, W1_A4, W1_A6, W1_A8, W1_B1, W1_B3, W1_B5, W1_B7, W1_C2, W1_C4, W1_C6, W1_C8, W1_D1, W1_D3, W1_D5, W1_D7, W1_E2, W1_E4, W1_E6, W1_E8, W1_F1, W1_F3, W1_F5, W1_F7, W1_G2, W1_G4, W1_G6, W1_G8, W1_H1, W1_H3, W1_H5, W1_H7, W1_I2, W1_I4, W1_I6, W1_I8, W1_J1, W1_J3, W1_J5, W1_J7, W1_K2, W1_K4, W1_K6, W1_K8, W1_L1, W1_L3, W1_L5, W1_L7, W1_M2, W1_M4, W1_M6, W1_M8, W1_N1, W1_N3, W1_N5, W1_N7, W1_O2, W1_O4, W1_O6, W1_O8, W1_P1, W1_P3, W1_P5, W1_P7, W1_Q2, W1_Q4, W1_Q6, W1_Q8, W1_R1, W1_R3, W1_R5, W1_R7, W2_A2, W2_A4, W2_A6, W2_A8, W2_B1, W2_B3, W2_B5, W2_B7, W2_C2, W2_C4, W2_C6, W2_C8, W2_D1, W2_D3, W2_D5, W2_D7, W2_E2, W2_E4, W2_E6, W2_E8, W2_F1, W2_F3, W2_F5, W2_F7, W2_G2, W2_G4, W2_G6, W2_G8, W2_H1, W2_H3, W2_H5, W2_H7, W2_I2, W2_I4, W2_I6, W2_I8, W2_J1, W2_J3, W2_J5, W2_J7, W2_K2, W2_K4, W2_K6, W2_K8, W2_L1, W2_L3, W2_L5, W2_L7, W2_M2, W2_M4, W2_M6, W2_M8, W2_N1, W2_N3, W2_N5, W2_N7, W2_O2, W2_O4, W2_O6, W2_O8, W2_P1, W2_P3, W2_P5, W2_P7, W2_Q2, W2_Q4, W2_Q6, W2_Q8, W2_R1, W2_R3, W2_R5, W2_R7, W3_A2, W3_A4, W3_A6, W3_A8, W3_B1, W3_B3, W3_B5, W3_B7, W3_C2, W3_C4, W3_C6, W3_C8, W3_D1, W3_D3, W3_D5, W3_D7, W3_E2, W3_E4, W3_E6, W3_E8, W3_F1, W3_F3, W3_F5, W3_F7, W3_G2, W3_G4, W3_G6, W3_G8, W3_H1, W3_H3, W3_H5, W3_H7, W3_I2, W3_I4, W3_I6, W3_I8, W3_J1, W3_J3, W3_J5, W3_J7, W3_K2, W3_K4, W3_K6, W3_K8, W3_L1, W3_L3, W3_L5, W3_L7, W3_M2, W3_M4, W3_M6, W3_M8, W3_N1, W3_N3, W3_N5, W3_N7, W3_O2, W3_O4, W3_O6, W3_O8, W3_P1, W3_P3, W3_P5, W3_P7, W3_Q2, W3_Q4, W3_Q6, W3_Q8, W3_R1, W3_R3, W3_R5, W3_R7)
- dates 2025-08-24 .. 2025-09-19; slots [6, 9, 12, 15, 18]; qc_pass 16880 of 28080
- frame table: yes; parity table: yes


## E0a paper parity

| quantity | ours | paper | rel_diff | pass | ci_low | ci_high | p_value | n_rows | n_plants |
|---|---|---|---|---|---|---|---|---|---|
| control_slope_mm_per_day | 4.3 | 4.3 | 0.000231 | True | 3.18 | 5.42 | 4.61e-14 | 120 | 10 |
| interaction_slope_mm_per_day | -13.1 | -13.1 | -0.000245 | True | -14.7 | -11.5 | 3.45e-59 | 120 | 10 |
| control_slope_mm_per_day [ours: height_p75 above soil, qc_pass] | 5.94 | 4.3 | 0.381 |  | 4.26 | 7.62 | 4.13e-12 | 87 | 10 |
| interaction_slope_mm_per_day [ours: height_p75 above soil, qc_pass] | -12.3 | -13.1 | 0.0561 |  | -14.8 | -9.94 | 9.27e-24 | 87 | 10 |


## E0b QC agreement

| flag | rule | precision | recall | f1 | n_labelled | n_flagged | n_runs | n_plants |
|---|---|---|---|---|---|---|---|---|
| qc_dark | dark: >= 8 of 15 RGB frames Dark | 0.96 | 0.896 | 0.927 | 347 | 324 | 1044 | 8 |
| qc_dark | dark: any RGB frame Dark | 0.963 | 0.874 | 0.916 | 357 | 324 | 1044 | 8 |
| qc_dark | dark: >= 8 RGB frames Dark or Medium Dark | 1 | 0.628 | 0.771 | 516 | 324 | 1044 | 8 |
| qc_not_centred | not centred: any depth/intensity/RGB/3D frame | 0.462 | 0.109 | 0.177 | 559 | 132 | 1044 | 8 |
| qc_not_centred | not centred: any depth frame | 0.00758 | 0.125 | 0.0143 | 8 | 132 | 1044 | 8 |
| qc_not_centred | not centred: >= 3 frames | 0.0909 | 0.0839 | 0.0873 | 143 | 132 | 1044 | 8 |
| qc_no_plant | no plant: any frame | 0 | 0 |  | 1 | 29 | 1044 | 8 |


## E1 noise floor

| trait | unit | noise | noise_within_column | signal | snr | reliable | signal_date | n_runs | frames_per_run |
|---|---|---|---|---|---|---|---|---|---|
| height_p50 | mm | 2.05 | 1.62 | 83.8 | 40.8 | True | 2025-09-14 | 16746 | 6 |
| height_p75 | mm | 2.17 | 1.71 | 90.9 | 41.9 | True | 2025-09-14 | 16746 | 6 |
| height_p95 | mm | 2.68 | 2.13 | 101 | 37.6 | True | 2025-09-14 | 16746 | 6 |
| projected_area | cm2 | 3.58 | 2.75 | 59.5 | 16.6 | True | 2025-09-14 | 16746 | 6 |
| leaf_area_est | cm2 | 6.4 | 4.89 | 155 | 24.2 | True | 2025-09-14 | 16746 | 6 |
| hull_volume | cm3 | 47.7 | 36.3 | 2.14e+03 | 44.8 | True | 2025-09-14 | 16746 | 6 |
| canopy_radius | mm | 0.367 | 0.302 | 2.81 | 7.64 | True | 2025-09-14 | 16746 | 6 |
| inclination_mean | deg | 1.25 | 1.01 | 7.4 | 5.92 | True | 2025-09-14 | 16746 | 6 |
| inclination_sd | deg | 0.626 | 0.51 | 2.35 | 3.75 | True | 2025-09-14 | 16746 | 6 |
| droop_index | ratio | 0.0194 | 0.0154 | 0.158 | 8.12 | True | 2025-09-14 | 16746 | 6 |
| compactness | ratio | 0.0237 | 0.0188 | 0.26 | 11 | True | 2025-09-14 | 16746 | 6 |
| n_points | count | 110 | 80 | 3.02e+03 | 27.4 | True | 2025-09-14 | 16746 | 6 |

Noise source: frames table, frames with used and ok True.


## E1 secondary: cyl_* traits (all vegetation in the cylinder, no clustering)

| trait | unit | noise | noise_within_column | signal | snr | reliable | n_runs |
|---|---|---|---|---|---|---|---|
| cyl_height_p50 | mm | 2.06 | 1.62 | 82.2 | 39.8 | True | 16746 |
| cyl_height_p75 | mm | 2.23 | 1.76 | 87.3 | 39.1 | True | 16746 |
| cyl_height_p95 | mm | 2.86 | 2.26 | 95.1 | 33.2 | True | 16746 |
| cyl_projected_area | cm2 | 4.84 | 3.85 | 78.9 | 16.3 | True | 16746 |
| cyl_n_points | count | 118 | 85.5 | 3.19e+03 | 27 | True | 16746 |


## E2 bed baseline

Traits: E1-reliable traits (12).

| trait | day_bed_coef | ci_low | ci_high | p_value | p_holm | warning | w1_slope | unit | n_plants | first_date | last_date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| height_p50 | -1.85 | -2.4 | -1.3 | 5.04e-11 | 5.54e-10 | True | 7.34 | mm/day | 144 | 2025-08-24 | 2025-09-02 |
| height_p75 | -1.96 | -2.58 | -1.34 | 6.1e-10 | 4.88e-09 | True | 8.5 | mm/day | 144 | 2025-08-24 | 2025-09-02 |
| height_p95 | -2.47 | -3.23 | -1.71 | 2.19e-10 | 1.97e-09 | True | 10.1 | mm/day | 144 | 2025-08-24 | 2025-09-02 |
| projected_area | -0.881 | -1.7 | -0.0662 | 0.0341 | 0.204 | True | 11.3 | cm2/day | 144 | 2025-08-24 | 2025-09-02 |
| leaf_area_est | -2.01 | -3.34 | -0.694 | 0.00279 | 0.0195 | True | 19.8 | cm2/day | 144 | 2025-08-24 | 2025-09-02 |
| hull_volume | -50.1 | -65 | -35.1 | 5.26e-11 | 5.54e-10 | True | 246 | cm3/day | 144 | 2025-08-24 | 2025-09-02 |
| canopy_radius | 0.042 | -0.205 | 0.289 | 0.739 | 1 | False | 0.703 | mm/day | 144 | 2025-08-24 | 2025-09-02 |
| inclination_mean | -0.119 | -0.348 | 0.109 | 0.307 | 1 | False | 0.666 | deg/day | 144 | 2025-08-24 | 2025-09-02 |
| inclination_sd | 0.0556 | -0.0199 | 0.131 | 0.149 | 0.744 | False | -0.0921 | deg/day | 144 | 2025-08-24 | 2025-09-02 |
| droop_index | 0.00118 | -0.00355 | 0.00591 | 0.625 | 1 | False | 0.00823 | ratio/day | 144 | 2025-08-24 | 2025-09-02 |
| compactness | -0.00107 | -0.00455 | 0.00242 | 0.549 | 1 | False | 0.031 | ratio/day | 144 | 2025-08-24 | 2025-09-02 |
| n_points | -114 | -138 | -90.6 | 1.67e-21 | 2e-20 | True | 390 | count/day | 144 | 2025-08-24 | 2025-09-02 |


## E3 stress onset (ranked)

Traits: E1-reliable traits (12).

| rank | trait | onset_day | onset_date | onset_day_unadjusted | lead_vs_symptom_days | lead_unadjusted_days | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit | holm_reachable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | n_points | 4 | 2025-09-07 | 4 | -1 | -1 | -8.55 | -9.36 | -7.75 | 3.66e-96 | 57 | 49 | True |
| 2 | leaf_area_est | 4 | 2025-09-07 | 4 | -1 | -1 | -6.72 | -7.4 | -6.03 | 1.8e-82 | 57 | 49 | True |
| 3 | hull_volume | 4 | 2025-09-07 | 4 | -1 | -1 | -9.37 | -10.4 | -8.33 | 8.91e-70 | 57 | 49 | True |
| 4 | projected_area | 4 | 2025-09-07 | 4 | -1 | -1 | -4.89 | -5.46 | -4.31 | 4.12e-62 | 57 | 49 | True |
| 5 | height_p95 | 5 | 2025-09-08 | 4 | 0 | -1 | -5.75 | -6.42 | -5.09 | 8.36e-64 | 57 | 49 | True |
| 6 | compactness | 6 | 2025-09-09 | 4 | 1 | -1 | -3.35 | -3.8 | -2.91 | 1.2e-49 | 57 | 49 | True |
| 7 | canopy_radius | 6 | 2025-09-09 | 6 | 1 | 1 | -0.403 | -0.517 | -0.289 | 3.84e-12 | 57 | 49 | True |
| 8 | height_p75 | 7 | 2025-09-10 | 5 | 2 | 0 | -5.79 | -6.52 | -5.07 | 3.74e-55 | 57 | 49 | True |
| 9 | height_p50 | 7 | 2025-09-10 | 7 | 2 | 2 | -6.57 | -7.41 | -5.74 | 1.59e-53 | 57 | 49 | True |
| 10 | inclination_mean | 10 | 2025-09-13 | 5 | 5 | 0 | -1.23 | -1.53 | -0.923 | 3.37e-15 | 57 | 49 | True |
| 11 | inclination_sd |  |  | 9 |  | 4 | 0.609 | 0.409 | 0.81 | 2.6e-09 | 57 | 49 | True |
| 12 | droop_index |  |  | 10 |  | 5 | -0.903 | -1.31 | -0.492 | 1.66e-05 | 57 | 49 | True |

Symptom reference: 2025-09-08 (BBCH sheet: W2_A8 'aphids, wilting' (1 of 49 deficit plants have a written symptom)).


## E3 sensitivity: daytime slots 09-18 h, plant-day means

| rank | trait | onset_day | onset_day_unadjusted | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | canopy_radius | 0 | 0 | -0.428 | -0.478 | -0.377 | 2.25e-62 | 70 | 64 |
| 2 | n_points | 4 | 2 | -8.31 | -8.71 | -7.9 | 0 | 70 | 64 |
| 3 | hull_volume | 4 | 4 | -9.3 | -9.76 | -8.85 | 0 | 70 | 64 |
| 4 | height_p95 | 4 | 4 | -5.47 | -5.76 | -5.18 | 2.1e-301 | 70 | 64 |
| 5 | leaf_area_est | 4 | 4 | -6.57 | -6.94 | -6.2 | 7.67e-267 | 70 | 64 |
| 6 | height_p75 | 4 | 4 | -5.56 | -5.88 | -5.24 | 2.36e-255 | 70 | 64 |
| 7 | projected_area | 4 | 4 | -4.79 | -5.09 | -4.49 | 6.35e-216 | 70 | 64 |
| 8 | compactness | 4 | 4 | -3.01 | -3.24 | -2.78 | 2.95e-148 | 70 | 64 |
| 9 | height_p50 | 7 | 4 | -6.29 | -6.65 | -5.93 | 2.46e-257 | 70 | 64 |
| 10 | inclination_sd | 9 | 6 | 0.761 | 0.662 | 0.86 | 1.98e-51 | 70 | 64 |
| 11 | droop_index | 10 | 10 | -0.928 | -1.1 | -0.759 | 7.4e-27 | 70 | 64 |
| 12 | inclination_mean |  | 9 | -0.753 | -0.896 | -0.61 | 5.89e-25 | 70 | 64 |


## E3 secondary: cyl_* traits, 15 h

| rank | trait | onset_day | onset_day_unadjusted | slope_interaction | ci_low | ci_high | p_value | n_control | n_deficit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cyl_n_points | 4 | 4 | -8.44 | -9.1 | -7.77 | 2.47e-136 | 57 | 49 |
| 2 | cyl_projected_area | 4 | 4 | -5.55 | -6.01 | -5.08 | 2.31e-119 | 57 | 49 |
| 3 | cyl_height_p95 | 5 | 5 | -5.4 | -6.03 | -4.78 | 2.09e-64 | 57 | 49 |
| 4 | cyl_height_p75 | 7 | 7 | -5.67 | -6.35 | -4.99 | 3.02e-60 | 57 | 49 |
| 5 | cyl_height_p50 | 7 | 7 | -6.44 | -7.24 | -5.65 | 1.46e-56 | 57 | 49 |


## E4 daily rhythm

| trait | response | slope_interaction | ci_low | ci_high | p_value | vpd_coef | vpd_p | slope_interaction_par_adjusted | onset_day | onset_day_unadjusted | unit | slots_present |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| height_p50 | midday_depression | -0.0115 | -0.239 | 0.216 | 0.921 | -1.57 | 0.0635 | -0.0201 |  | 10 | mm per day | 6,9,12,15,18 |
| height_p50 | overnight_recovery | 0.405 | 0.168 | 0.641 | 0.000792 | 4.8 | 2.35e-06 | 0.403 | 4 | 4 | mm per day | 6,9,12,15,18 |
| height_p75 | midday_depression | 0.0446 | -0.201 | 0.29 | 0.722 | -2.06 | 0.0233 | 0.0412 |  | 10 | mm per day | 6,9,12,15,18 |
| height_p75 | overnight_recovery | 0.484 | 0.232 | 0.735 | 0.000163 | 4.35 | 6.19e-05 | 0.489 |  | 3 | mm per day | 6,9,12,15,18 |
| height_p95 | midday_depression | 0.201 | -0.0878 | 0.489 | 0.173 | -1.41 | 0.187 | 0.198 |  | 10 | mm per day | 6,9,12,15,18 |
| height_p95 | overnight_recovery | 0.437 | 0.138 | 0.736 | 0.00417 | 4.09 | 0.00152 | 0.436 |  |  | mm per day | 6,9,12,15,18 |
| projected_area | midday_depression | 0.183 | -0.179 | 0.545 | 0.322 | -0.324 | 0.809 | 0.181 |  |  | cm2 per day | 6,9,12,15,18 |
| projected_area | overnight_recovery | -1.06 | -1.44 | -0.692 | 2.08e-08 | 10.2 | 1.97e-10 | -1.06 | 8 | 8 | cm2 per day | 6,9,12,15,18 |
| leaf_area_est | midday_depression | 0.157 | -0.568 | 0.882 | 0.67 | -3.96 | 0.141 | 0.155 |  | 10 | cm2 per day | 6,9,12,15,18 |
| leaf_area_est | overnight_recovery | -1.99 | -2.73 | -1.25 | 1.3e-07 | 16.9 | 1.19e-07 | -1.98 | 8 | 8 | cm2 per day | 6,9,12,15,18 |
| hull_volume | midday_depression | 6.05 | -2.35 | 14.5 | 0.158 | -19.1 | 0.54 | 6.8 |  |  | cm3 per day | 6,9,12,15,18 |
| hull_volume | overnight_recovery | -11.5 | -19.7 | -3.23 | 0.00635 | 85.2 | 0.0167 | -11.5 | 8 | 7 | cm3 per day | 6,9,12,15,18 |
| canopy_radius | midday_depression | -0.0159 | -0.102 | 0.0698 | 0.716 | 0.19 | 0.55 | -0.0156 |  |  | mm per day | 6,9,12,15,18 |
| canopy_radius | overnight_recovery | 0.0216 | -0.0597 | 0.103 | 0.602 | -0.143 | 0.684 | 0.0216 |  |  | mm per day | 6,9,12,15,18 |
| inclination_mean | midday_depression | 0.0122 | -0.0561 | 0.0804 | 0.727 | -0.961 | 0.000145 | -0.00772 |  | 9 | deg per day | 6,9,12,15,18 |
| inclination_mean | overnight_recovery | 0.00341 | -0.0633 | 0.0701 | 0.92 | -1.36 | 2.54e-06 | 0.00336 |  |  | deg per day | 6,9,12,15,18 |
| inclination_sd | midday_depression | -0.00855 | -0.0353 | 0.0182 | 0.531 | -0.204 | 0.0392 | -0.00776 |  |  | deg per day | 6,9,12,15,18 |
| inclination_sd | overnight_recovery | -0.0105 | -0.0365 | 0.0154 | 0.427 | -0.218 | 0.0519 | -0.0105 |  |  | deg per day | 6,9,12,15,18 |
| droop_index | midday_depression | -0.000945 | -0.0023 | 0.000413 | 0.173 | 0.000171 | 0.973 | -0.000982 |  | 10 | ratio per day | 6,9,12,15,18 |
| droop_index | overnight_recovery | 0.00153 | 0.000173 | 0.00288 | 0.0271 | 0.0219 | 0.000191 | 0.00154 |  | 8 | ratio per day | 6,9,12,15,18 |
| compactness | midday_depression | 0.00134 | -0.000362 | 0.00305 | 0.123 | -0.000662 | 0.917 | 0.00133 |  |  | ratio per day | 6,9,12,15,18 |
| compactness | overnight_recovery | -0.00213 | -0.00384 | -0.000423 | 0.0145 | 0.0547 | 1.22e-13 | -0.00211 |  | 10 | ratio per day | 6,9,12,15,18 |
| n_points | midday_depression | 6.07 | -8.93 | 21.1 | 0.428 | -89.9 | 0.106 | 4.87 |  |  | count per day | 6,9,12,15,18 |
| n_points | overnight_recovery | -40.4 | -54.9 | -26 | 3.85e-08 | 402 | 1.11e-10 | -40.3 | 8 | 6 | count per day | 6,9,12,15,18 |


## E4 sensitivity: dark runs excluded (midday depression only)

| trait | slope_interaction | ci_low | ci_high | p_value | onset_day | onset_day_unadjusted | unit | n_plants |
|---|---|---|---|---|---|---|---|---|
| height_p50 | -0.0187 | -0.298 | 0.261 | 0.896 |  |  | mm per day | 144 |
| height_p75 | -0.0269 | -0.335 | 0.282 | 0.864 |  |  | mm per day | 144 |
| height_p95 | 0.0474 | -0.316 | 0.411 | 0.798 |  |  | mm per day | 144 |
| projected_area | -0.368 | -0.786 | 0.0506 | 0.0849 | 7 | 7 | cm2 per day | 144 |
| leaf_area_est | -1.13 | -1.98 | -0.282 | 0.00902 | 7 | 6 | cm2 per day | 144 |
| hull_volume | -4.23 | -14.4 | 5.89 | 0.412 |  |  | cm3 per day | 144 |
| canopy_radius | -0.0246 | -0.13 | 0.0813 | 0.649 |  |  | mm per day | 144 |
| inclination_mean | 0.0175 | -0.0661 | 0.101 | 0.681 |  |  | deg per day | 144 |
| inclination_sd | -0.0281 | -0.0596 | 0.00335 | 0.0799 |  |  | deg per day | 144 |
| droop_index | -0.000572 | -0.00218 | 0.00104 | 0.487 |  |  | ratio per day | 144 |
| compactness | -0.001 | -0.00301 | 0.00101 | 0.33 | 7 | 7 | ratio per day | 144 |
| n_points | -13.9 | -30.8 | 3.04 | 0.108 | 7 | 7 | count per day | 144 |


## E5 turgor

Best structural proxy: **leaf_area_est** at lag +1 d, pooled repeated-measures r = -0.593, p < 1e-16, n = 334; per-plant median rho = -0.633 over 9 plants.

| plant | trait | lag_days | rho | p_value | n |
|---|---|---|---|---|---|
| pooled_W2 | canopy_radius | -2 | -0.0828 | 0.118 | 366 |
| pooled_W2 | canopy_radius | -1 | -0.106 | 0.0459 | 363 |
| pooled_W2 | canopy_radius | 0 | -0.0893 | 0.0954 | 358 |
| pooled_W2 | canopy_radius | 1 | -0.0935 | 0.092 | 334 |
| pooled_W2 | canopy_radius | 2 | -0.0648 | 0.27 | 300 |
| pooled_W2 | compactness | -2 | -0.341 | 3.22e-11 | 366 |
| pooled_W2 | compactness | -1 | -0.334 | 1.01e-10 | 363 |
| pooled_W2 | compactness | 0 | -0.33 | 2.52e-10 | 358 |
| pooled_W2 | compactness | 1 | -0.339 | 3.18e-10 | 334 |
| pooled_W2 | compactness | 2 | -0.255 | 1.01e-05 | 300 |
| pooled_W2 | droop_index | -2 | 0.0332 | 0.531 | 366 |
| pooled_W2 | droop_index | -1 | 0.0845 | 0.112 | 363 |
| pooled_W2 | droop_index | 0 | 0.0733 | 0.171 | 358 |
| pooled_W2 | droop_index | 1 | -0.038 | 0.495 | 334 |
| pooled_W2 | droop_index | 2 | -0.068 | 0.247 | 300 |
| pooled_W2 | height_p50 | -2 | -0.0305 | 0.565 | 366 |
| pooled_W2 | height_p50 | -1 | -0.0585 | 0.272 | 363 |
| pooled_W2 | height_p50 | 0 | -0.0888 | 0.0974 | 358 |
| pooled_W2 | height_p50 | 1 | -0.195 | 0.000391 | 334 |
| pooled_W2 | height_p50 | 2 | -0.192 | 0.000981 | 300 |
| pooled_W2 | height_p75 | -2 | -0.0383 | 0.47 | 366 |
| pooled_W2 | height_p75 | -1 | -0.0633 | 0.234 | 363 |
| pooled_W2 | height_p75 | 0 | -0.108 | 0.0438 | 358 |
| pooled_W2 | height_p75 | 1 | -0.202 | 0.000246 | 334 |
| pooled_W2 | height_p75 | 2 | -0.196 | 0.000756 | 300 |
| pooled_W2 | height_p95 | -2 | -0.11 | 0.0375 | 366 |
| pooled_W2 | height_p95 | -1 | -0.184 | 0.000478 | 363 |
| pooled_W2 | height_p95 | 0 | -0.208 | 8.98e-05 | 358 |
| pooled_W2 | height_p95 | 1 | -0.255 | 3.09e-06 | 334 |
| pooled_W2 | height_p95 | 2 | -0.237 | 4.27e-05 | 300 |
| pooled_W2 | hull_volume | -2 | -0.375 | 2.29e-13 | 366 |
| pooled_W2 | hull_volume | -1 | -0.468 | 9.84e-21 | 363 |
| pooled_W2 | hull_volume | 0 | -0.483 | 7.26e-22 | 358 |
| pooled_W2 | hull_volume | 1 | -0.52 | 5.91e-24 | 334 |
| pooled_W2 | hull_volume | 2 | -0.513 | 5.09e-21 | 300 |
| pooled_W2 | inclination_mean | -2 | 0.0223 | 0.674 | 366 |
| pooled_W2 | inclination_mean | -1 | -0.0749 | 0.159 | 363 |
| pooled_W2 | inclination_mean | 0 | -0.137 | 0.0104 | 358 |
| pooled_W2 | inclination_mean | 1 | -0.098 | 0.0772 | 334 |
| pooled_W2 | inclination_mean | 2 | -0.115 | 0.0496 | 300 |
| pooled_W2 | inclination_sd | -2 | 0.362 | 1.51e-12 | 366 |
| pooled_W2 | inclination_sd | -1 | 0.373 | 3.81e-13 | 363 |
| pooled_W2 | inclination_sd | 0 | 0.364 | 2.23e-12 | 358 |
| pooled_W2 | inclination_sd | 1 | 0.263 | 1.44e-06 | 334 |
| pooled_W2 | inclination_sd | 2 | 0.185 | 0.00154 | 300 |
| pooled_W2 | leaf_area_est | -2 | -0.456 | 8.78e-20 | 366 |
| pooled_W2 | leaf_area_est | -1 | -0.539 | 4.17e-28 | 363 |
| pooled_W2 | leaf_area_est | 0 | -0.556 | 8.12e-30 | 358 |
| pooled_W2 | leaf_area_est | 1 | -0.593 | 2.19e-32 | 334 |
| pooled_W2 | leaf_area_est | 2 | -0.581 | 8.38e-28 | 300 |
| pooled_W2 | n_points | -2 | -0.437 | 3.73e-18 | 366 |
| pooled_W2 | n_points | -1 | -0.508 | 1.14e-24 | 363 |
| pooled_W2 | n_points | 0 | -0.505 | 4.56e-24 | 358 |
| pooled_W2 | n_points | 1 | -0.543 | 2.29e-26 | 334 |
| pooled_W2 | n_points | 2 | -0.516 | 3.01e-21 | 300 |
| pooled_W2 | projected_area | -2 | -0.454 | 1.29e-19 | 366 |
| pooled_W2 | projected_area | -1 | -0.504 | 2.66e-24 | 363 |
| pooled_W2 | projected_area | 0 | -0.505 | 4.42e-24 | 358 |
| pooled_W2 | projected_area | 1 | -0.554 | 1.37e-27 | 334 |
| pooled_W2 | projected_area | 2 | -0.535 | 5.31e-23 | 300 |


## E5 sensitivity: pooled W2 r by variant (lags 0 and +1)

| trait | lag_days | all qc_pass runs (plan) | dark runs excluded | 15 h only | dark runs excluded, detrended |
|---|---|---|---|---|---|
| canopy_radius | 0 | -0.0893 | -0.117 | 0.0314 | 0.128 |
| canopy_radius | 1 | -0.0935 | -0.0905 | 0.00427 | 0.181 |
| compactness | 0 | -0.33 | -0.353 | -0.472 | -0.221 |
| compactness | 1 | -0.339 | -0.35 | -0.575 | -0.223 |
| droop_index | 0 | 0.0733 | 0.0404 | -0.0666 | -0.218 |
| droop_index | 1 | -0.038 | -0.00661 | -0.28 | -0.293 |
| height_p50 | 0 | -0.0888 | -0.0762 | -0.0295 | -0.341 |
| height_p50 | 1 | -0.195 | -0.149 | -0.243 | -0.422 |
| height_p75 | 0 | -0.108 | -0.0644 | 0.025 | -0.318 |
| height_p75 | 1 | -0.202 | -0.141 | -0.143 | -0.404 |
| height_p95 | 0 | -0.208 | -0.171 | -0.0273 | -0.308 |
| height_p95 | 1 | -0.255 | -0.226 | -0.133 | -0.356 |
| hull_volume | 0 | -0.483 | -0.497 | -0.458 | -0.127 |
| hull_volume | 1 | -0.52 | -0.528 | -0.541 | -0.159 |
| inclination_mean | 0 | -0.137 | -0.117 | -0.0186 | -0.323 |
| inclination_mean | 1 | -0.098 | -0.0542 | 0.0831 | -0.161 |
| inclination_sd | 0 | 0.364 | 0.389 | 0.357 | 0.354 |
| inclination_sd | 1 | 0.263 | 0.232 | 0.232 | 0.147 |
| leaf_area_est | 0 | -0.556 | -0.564 | -0.538 | -0.129 |
| leaf_area_est | 1 | -0.593 | -0.599 | -0.631 | -0.173 |
| n_points | 0 | -0.505 | -0.494 | -0.492 | -0.123 |
| n_points | 1 | -0.543 | -0.541 | -0.598 | -0.214 |
| projected_area | 0 | -0.505 | -0.509 | -0.493 | -0.0374 |
| projected_area | 1 | -0.554 | -0.566 | -0.602 | -0.132 |
