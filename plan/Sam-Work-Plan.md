# Sam — Work Plan: Challenges 2 & 4, Spectral/Thermal Physiology and Diurnal Modeling

- **Owner:** Sam
- **Challenges:** 2 — physiological & spectral index analysis (vegetation indices, canopy temperature, turgor/RWC prediction); 4 — diurnal & environmental modeling (daily rhythms, context-aware irrigation)
- **Dataset:** ADA 2025 ([paper](https://arxiv.org/html/2610.23609v1), [code](https://github.com/pippowell/ada2025), data DOI `10.26249/FK2/DMDLTE`)
- **Time box:** 2026-11-14 and 2026-11-15
- **Status:** the environmental sub-dataset (air, leaf clips, soil, RWC, leaf scans, PAR) is analysed (first-pass results in [`docs/Sam-Analysis-Notes.md`](../docs/Sam-Analysis-Notes.md) §4). No camera data has been downloaded yet. Build starts after the team master plan is agreed.

## 1. What I am building

A tool that shows **how many days of warning each sensing method gives before a
water-stressed plant visibly wilts**, and turns that into an irrigation trigger
a farmer could use. It has three parts:

1. **Pipeline:** hyperspectral + thermal images + microclimate → one tidy per-plant, per-run table of spectral indices, leaf − air temperature and environmental drivers.
2. **Statistics:** onset day per signal, diurnal response to VPD and light, RWC/turgor models, and an irrigation-trigger replay, all with the bed confound and honest splits handled.
3. **App pages:** early-warning race, bed map over time, plant profile, irrigation advisor.

It reuses the dataset authors' code for the hard parts: HSI flat-field
correction, the HSI plant-segmentation model and the thermal leaf − air
pipeline. The two days go into scaling them up and into the new analyses.

## 2. What the dataset gives for spectral, thermal and diurnal work

| Item | Fact |
|---|---|
| Hyperspectral | Resonon Pika L push-broom, 256 x 256 px x 300 bands, ~384–1025 nm, raw uint8 JP2 + JSON; 1 cube per plant per run |
| Thermal | FLIR A70, 640 x 480, °C float32 TIFF; 5 frames per plant per run |
| Runs | 06, 09, 12, 15, 18 h daily, 2025-08-12 to 2025-09-22 (42 days); re-runs within ±1 h count as the scheduled run |
| Air | 2 x BME280, 2 min: T, RH, pressure. Sensor 1 = W3, sensor 2 = W1, **none at W2** |
| PAR | Driesen sensor, 15 min, raw mV (uncalibrated), one for the greenhouse |
| Turgor | 21 AgriHouse leaf clips, 2 min, from 2025-09-02 (9 W2, 9 W3, 3 W1); **higher value = lower turgor** |
| RWC | destructive, W3 only: 63 plants, 70 leaves, 4–15 Sep; leaf scans at 200 dpi with a 1 cm² reference |
| Soil | 4 sensors in W2, **not usable** (authors) |
| Irrigation | fixed schedule, 11 and 19 h, 8 min per bed; off for W2/W3 from 2025-09-03 11:00 until 2025-09-15 18:35 |
| Plants | 216 (72 per bed). W1 = control. W2, W3 = water deficit |
| Fusion | thermal can be attached to RGB-D points (Kinjalk's pipeline). **Hyperspectral is not co-registered** with any other camera |
| Ground truth | turgor clips, RWC/EWT/LMA, BBCH and symptom log (wilting logged 8 Sep, yellowing 12 Sep, severe stress 15 Sep) |

Download sizes, HSI + thermal: 205.3 + 114.1 GB in all. The shared core set
(10 paper plants + 21 clip plants, 29 unique, since W1_J5 and W2_R7 are in
both) is **43.6 GB** for HSI + thermal, or 84.4 GB with RGB + depth. The 63 RWC
plants add **97.7 GB** of HSI + thermal. These sizes come from the osnaData
listing.

## 3. Constraints that shape the design

1. **One control bed.** Treatment is confounded with bed. Like Kinjalk, I use each plant's own pre-cutoff period as its baseline. I also restrict pre/post contrasts to **2025-09-02 onward**, because the doors near the plants were closed on 2 Sep and the dry bed's air was already +1.7 °C warmer that day.
2. **No W2 air sensor.** Leaf − air for W2 uses the W1 sensor. I report how sensitive the results are to that choice, and W3 (which has its own sensor) serves as the cross-check.
3. **Hyperspectral is not registered.** HSI traits are computed per cube with the authors' segmentation model, and joined to other modalities by keys only, never by pixel.
4. **Neighbours in view, overlap from about BBCH 12–14.** Plant masks are QC'd. The authors' HSI model was trained on 10 W1/W2 plants only, so I check it on W3 and on wilted plants before trusting it.
5. **Dark 06 and 18 h runs** (RGB/HSI). Spectral diurnal metrics use 09/12/15 h. Thermal is unaffected and uses all five runs.
6. **Scale mismatch.** RWC is one leaf; spectra and thermal are canopy-level. I expect a ceiling on RWC prediction skill.
7. **No counterfactual irrigation.** The schedule was fixed and the soil sensors are unusable, so irrigation work tests *when a rule would trigger*, not water saved.
8. **Not a benchmark.** The authors give no splits, so I define and freeze them before modelling (section 7).

## 4. Pipeline

```
HSI cube + JSON           thermal TIFFs              air / PAR / turgor parquet
  -> flat-field              -> per-frame leaf temp     -> 10-min window at capture time
     (R-D)/(W*m-D)              (coolest-50 % median;      (bed's own air sensor; W1 for W2)
  -> plant mask (authors'       RGB-mask variant later)  -> VPD, PAR, turgor shift
     model, QC)               -> median across 5 frames
  -> indices per plant
          \__________________________|_________________________/
                                     v
                  join on bed, plant, date, hour  ->  trait table (parquet + CSV)
```

| Step | Input | Output | Method |
|---|---|---|---|
| Reflectance | `hsi_000.jp2`, `hsi_000.json`, `black_reference.npy` | reflectance cube | hyperio; flat-field, `m` = 2.0 |
| HSI mask | reflectance cube | plant mask | authors' `model.pt2` (`generate_plant_masks.py`), threshold 0.5 |
| Indices | masked cube | one row per run | see section 5 (authors' definitions where they exist) |
| Leaf temp | 5 thermal TIFFs | leaf temp per run + spread | authors' `thermal_analysis.py`, re-run recovery on |
| Environment | parquet, PAR sheet | drivers at capture time | capture timestamps from `frames_thermal.csv` / HSI JSON |
| Turgor | leaf parquet | shift from pre-cutoff baseline | per-clip median 2 Sep 12:00 – 3 Sep 11:00 |
| Aggregate | all of the above | one row per plant-run + QC | join on the four shared keys |

## 5. Traits

| Trait | Definition | Why it matters |
|---|---|---|
| `ndvi`, `ci_rededge`, `sipi` | authors' definitions in `calc_spectral_index.py`, plant-masked mean | chlorophyll/pigment change; directly comparable to the paper |
| `pri` | (R531 − R570)/(R531 + R570) | photoprotection; midday depression is a diurnal stress marker |
| `wbi` | R900 / R970 | leaf water content; the 970 nm water band is inside the sensor range |
| `rep` | red-edge position (nm) | early chlorophyll shift before NDVI moves |
| `leaf_temp` | median of coolest 50 % thermal pixels, median over 5 frames (°C) | canopy temperature |
| `leaf_air_dT` | leaf_temp − air T (10-min window, bed's own sensor) (°C) | stomatal closure / reduced transpiration proxy |
| `leaf_air_dT_vpdnorm` | residual of leaf_air_dT against a VPD + PAR baseline fitted on W1 | context-aware stress signal; feeds the irrigation trigger |
| `air_T`, `rh`, `vpd`, `pressure`, `par_mV` | at capture time | diurnal drivers |
| `turgor_shift` | clip reading − pre-cutoff baseline (clip plants only) | ground truth for leaf water status |
| `mask_frac` | plant pixels / image pixels (HSI) | QC and canopy size cross-check against Kinjalk's `projected_area` |

QC flags per run: `hsi_mask_small`, `hsi_overexposed`, `dark_run`,
`thermal_frame_spread_high`, `missing_modality`, `rerun_used`, `air_from_W1`.

Leaf-level traits from the leaf scans (RWC table, keyed `bed`, `plant`, `date`, `leaf`):
`rwc`, `ewt_mg_cm2`, `lma_g_m2`, `leaf_area_cm2`, RGB colour indices. These are already computed.

## 6. Shared table schema (interface to teammates)

One row per plant per run. I adopt **Kinjalk's four join keys** unchanged.

| Column | Type | Example |
|---|---|---|
| `bed` | str | `W2` |
| `plant` | str | `J5` |
| `date` | date | `2025-09-08` |
| `hour` | int | `15` (scheduled slot; a re-run at 16 h is stored as 15) |
| `capture_hour` | int | `16` (actual run hour, if a re-run was used) |
| `treatment` | str | `control` / `deficit` |
| `days_since_cutoff` | int | `5` |
| `<trait>` | float | section 5 |
| `<trait>_spread` | float | spread over frames (thermal) |
| `qc_*` | bool | QC flags |

With the same keys, Kinjalk's 3D traits (height, droop, inclination) and my
spectral/thermal traits join into one table for the app and for multimodal models.

## 7. Experiments

| # | Question | Data | Method | Output |
|---|---|---|---|---|
| S1 | Does our setup reproduce the paper? | 10 paper plants, 15 h, 3–14 Sep | authors' scripts unchanged | match to the README "Expected results" table (sanity check) |
| S2 | Which signal shows stress first? (hero) | core set, 12/15 h, 2–15 Sep | Kinjalk's onset rule on baseline-normalised values, `trait ~ day * treatment + (1 \| plant)` | ranked onset day per signal, incl. his 3D traits, air ΔT, turgor, RGB greenness |
| S3 | Do spectral/thermal traits track turgor at plant level? | 21 clip plants | lagged Spearman per plant + mixed model | best image proxy for turgor; extends the paper's bed-mean correlations |
| S4 | Does the daily response to VPD/PAR change under stress? | core set, all runs | per plant-day slope of leaf_air_dT on VPD; 15 h − 09 h hysteresis; PRI midday depression; cosinor fit on 2-min turgor and air | day x hour heatmaps; per-plant VPD sensitivity |
| S5 | When would an irrigation rule have fired? | core set, 3–22 Sep | replay of 3 rules (microclimate only; leaf_air_dT; VPD-normalised leaf_air_dT) | lead time before wilting (8 Sep), false alarms on W1, recall on W2/W3 |
| S6 | Can RWC/EWT be predicted from canopy spectra + thermal? | 63 RWC plants, sampling-day run | PLSR on spectra + indices + leaf_air_dT; leave-one-date-out and grouped-by-plant CV | skill vs the "days since cutoff" baseline |
| S7 | Do the indices diverge in W3 as they do in W2? | core set | S2 model by bed | replication of the paper's W2 result in a second deficit bed |

Order of priority: S1, S2, S3, S5, S4, S7, S6. S6 runs only if the RWC plants'
HSI + thermal (97.7 GB) are downloaded by the end of Day 1.

Splits for S5 and S6, fixed in `splits.yaml` before modelling:
- **bed-held-out** (headline: thresholds set on W2, tested on W3)
- **time-blocked** with a 1-day gap
- **grouped by plant**

No random row splits.

Onset day uses the same definition as Kinjalk: the first day where the
treatment-control contrast is significant and stays significant on later days.

## 8. App pages I will deliver

1. **Early-warning race:** one timeline of onset day per signal (spectral, thermal, turgor, air, plus Kinjalk's 3D traits), with the logged wilting date marked. This is the hero chart.
2. **Bed map:** 216 pots in their real grid positions, a date/hour slider, and layers for NDVI, SIPI, WBI, leaf − air and turgor. Shows W2/W3 pulling away from W1.
3. **Plant profile:** reflectance spectrum over time (970 nm and red edge highlighted), leaf − air trace, turgor trace (clip plants), and leaf scan with measured RWC (RWC plants).
4. **Irrigation advisor replay:** traffic light per plant per run under each rule, with lead time and false-alarm counts.
5. **Honesty panel:** grouped vs leave-one-date-out skill, the scope (greenhouse, one variety, no yield data), and the confounds (door closure, W2 air, neighbours).

The app is Dash + Plotly (decided in `docs/SOFTWARE_STACK.md`). Each page is one
file I own in `app/pages/` (`early_warning.py`, `bed_map.py`, `plant_profile.py`,
`irrigation.py`, `honesty.py`). The pages need only the shared table plus about
20 thumbnail sets. The bed map is a Plotly heatmap/scatter on the real grid,
with a `dcc.Slider` for date and hour.

## 9. Schedule

| When | Work | Done when |
|---|---|---|
| Day 1, first | start download of the core set HSI + thermal (43.6 GB); set up the authors' environment | data on disk, environment runs |
| Day 1 | S1: run the authors' thermal + HSI scripts on the 10 paper plants | paper numbers reproduced |
| Day 1 | extend to all runs and all core plants; environment join; trait table | table for 29 plants x all runs |
| Day 1, late | start the RWC-plant download (97.7 GB); HSI masks for the core set run overnight | masks QC'd by eye on 3 plants per bed |
| Day 2, morning | S2, S3, S5 (then S4, S7; S6 if the RWC data is in) | figures and numbers |
| Day 2, midday | app pages 1, 4, 5, then 2, 3 | pages run on the shared table |
| Day 2, afternoon | joint onset chart with Kinjalk's traits; slides; integration | demo ready |

## 10. Out of scope for two days

These go on a "next steps" slide:

- All 216 plants (319 GB HSI + thermal) and full recovery analysis after 2025-09-15.
- Thermal leaf mask from RGB reprojection (better than coolest-50 %), and leaf temperature corrected for leaf angle (with Kinjalk's normals).
- CWSI with wet/dry reference surfaces.
- PAR calibration to µmol m⁻² s⁻¹; litres per irrigation event (asked of the authors).
- Learned multimodal stress classifiers beyond the three rules in S5.
- Domain adaptation to open-field data or other crops.

## 11. Other spectral / thermal datasets

None surveyed or verified yet. The goal would be to test whether the
VPD-normalised leaf − air trigger and the water-band index transfer to field
conditions. This is a "next steps" item, not part of the two days.

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| Download is slow | no HSI/thermal by midday Day 1 | start with the 10 paper plants (14.2 GB); environmental analyses already done |
| Authors' HSI mask fails on W3 or wilted plants | biased late-drought indices | QC flag; fall back to an NDVI-threshold mask; report the period where masks hold |
| Coolest-50 % thermal rule picks up drying soil | leaf − air biased warm in stressed beds | compare with an NDVI- or RGB-masked variant on a subset; report both |
| W3 turgor clips lack a pre-drought diel dip (possible artifact) | S3/S4 unreliable for W3 clips | use W2 clips as primary; flag W3 |
| HSI overexposure later in a run | bad spectra for some plants | `hsi_overexposed` flag from band saturation |
| RWC data not downloaded in time | S6 cannot run | present the leaf-scan RWC/EWT results already computed |
| One onset rule across many signals gives noisy dates | weak hero chart | bootstrap CI over plants; show the CI bars |

## 13. Questions for the master plan

1. Do we all use **one onset rule** (Kinjalk's, as above) so the early-warning race compares like with like?
2. **Shared plant set:** I agree with Kinjalk's 10 paper + 21 clip plants (29 unique). Can we add the **63 RWC plants** (W3) for HSI + thermal only?
3. Who owns the shared download, and where does it live? I can run it; the core set is 84.4 GB for all four cameras.
4. Should W2 leaf − air use the W1 or the W3 air sensor? I propose W1, with W3 as a sensitivity check.
5. Team-wide **split definitions** (`splits.yaml`) for any model, so no one reports random-split results?
6. `hyperio` install line for `requirements.txt` (`SOFTWARE_STACK.md` leaves it to the hyperspectral owner): the authors pin `hyperio @ git+https://github.com/unklar/hyperio`. I'll add it once I've confirmed it installs on Python 3.12. The HSI segmentation model also needs `torch`, which is not in our requirements yet. OK to add?
7. Who owns final integration and the slides?
