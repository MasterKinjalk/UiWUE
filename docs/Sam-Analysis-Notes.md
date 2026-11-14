# Analysis Plan — ADA 2025 Sugar-Beet Water-Deficit Dataset

**Dataset:** Powell, Franz, Smith & Ihorst (2026). *ADA 2025: A Multimodal Dataset for Water-Deficit Stress Detection in Sugar Beets.* osnaData, [doi:10.26249/FK2/DMDLTE](https://doi.org/10.26249/FK2/DMDLTE), CC BY 4.0.
**Companion code:** [github.com/pippowell/ada2025](https://github.com/pippowell/ada2025) (MIT): download/unzip tools, HSI loader and flat-field correction, a plant-segmentation model, the thermal pipeline, and camera calibration.
**Paper:** arXiv:2610.23609 (MetroAgriFor 2026). Cite both the paper and the dataset (BibTeX is in the companion README).

**Experiment:** 216 sugar beets (*Beta vulgaris*, CALLEDIA KWS Öko) in three beds of 72 plants: **W1** watered control, **W2** and **W3** water deficit. Seeded 4 Aug 2025. Irrigation of W2 and W3 stopped after the 11:00 run on **3 Sep** (about BBCH 14) and resumed **15 Sep at 18:35**. A gantry robot imaged every plant **5× per day** (06, 09, 12, 15 and 18 h) from 12 Aug to 22 Sep.

**Status (14 Nov 2026):** First pass done on the environmental sub-dataset (sensors, RWC and leaf scans). The camera data has not been downloaded yet. The plan is to **build on the companion repo, not re-implement it**.

---

## 1. Goals

| # | Workstream | Question |
|---|---|---|
| A | Vegetation-index tracking | How do NDVI, CI Red Edge, SIPI (and water-sensitive indices) evolve across all three beds and through recovery? |
| B | Canopy temperature & transpiration | Leaf − air temperature from thermal imagery as a proxy for stomatal closure, including diel patterns |
| C | Turgor & RWC prediction | Predict RWC and leaf water thickness non-destructively from hyperspectral, thermal and RGB data; validate against the leaf clips and the destructive RWC samples |
| D | Daily rhythm analysis | How do healthy vs stressed plants respond to daily swings in PAR, temperature and humidity across 5 runs a day for 42 days (12 Aug – 22 Sep)? |
| E | Context-aware irrigation | Can imagery plus microclimate (T, RH, pressure, PAR) drive an automated irrigation-scheduling rule? |

### What the dataset paper already covers (so we don't duplicate it)
- W1 vs **W2 only**, **5 plants per bed** (A2, A8, J5, R1, R7), **15 h run only**, **3–14 Sep**
- OLS treatment × day results, W2 relative to W1: NDVI −0.0189/day, CI Red Edge −0.0098/day, SIPI +0.0143/day, leaf − air temperature +0.596 °C/day (p = 0.03)
- Spearman correlations with bed-mean W2 turgor over 12 days: NDVI ρ = −0.93, SIPI ρ = 0.94, leaf − air ρ = 0.85
- The paper does **not** use W3, the RWC samples, the diel runs, the recovery period after 15 Sep, or plant-level matching to the leaf clips

### Where our analysis adds value
1. **Scale-up:** all plants, all three beds, every day **12 Aug – 22 Sep**, including recovery
2. **Diel resolution:** use all 5 runs per day for transpiration (leaf − air) dynamics, not just 15 h
3. **Plant-level ground truth:**
   - the 21 leaf-clip plants each have their own image time series
   - the 63 RWC-sampled W3 plants have imagery on their sampling day
4. **Water-specific spectral features:** the Pika L covers about 384–1025 nm, so the ~970 nm water band is available, for example WBI = R900/R970

---

## 2. Data inventory

### Environmental & metadata sub-dataset ([doi:10.26249/FK2/C0HOBN](https://doi.org/10.26249/FK2/C0HOBN), 1.5 GB). Already analysed.

| File | Contents | Notes |
|---|---|---|
| `air_readings.parquet` | 2 × BME280, 2-min interval | Sensor 1 = **W3**, sensor 2 = **W1**. **No air sensor at W2.** Shaded since 11 Aug |
| `leaf_readings.parquet` | 21 AgriHouse leaf-thickness clips, 2-min interval, from 2 Sep | 1–9 = W3, 10–18 = W2, 19–21 = W1. **Higher value = lower turgor.** Raw units |
| `soil_readings.parquet` | 4 sensors, all in W2 | **Not usable**, as confirmed by the authors |
| `metadata_rwc_par_en.xlsx` | Protocol, sensor map, BBCH/symptoms (7 dates), RWC (fw/sw/dw), PAR | D.M.Y dates. PAR is raw mV. RWC is from **W3 only** |
| `Leaf Scans.zip` | 70 RGB scans of the RWC leaves, 200 dpi, with a 1 cm² reference square | Scan order ≠ sheet order. Joined by the handwritten bag labels (see `analysis/features.py`) |
| `Plant Images.zip` (1.5 GB) | A photo of every plant on the last day | Not needed for the core analyses |

### Camera sub-datasets (one zip per plant and modality)

| Modality | Sensor | Per plant & run | All plants | Mean per plant |
|---|---|---|---|---|
| Hyperspectral ([PK9YDR](https://doi.org/10.26249/FK2/PK9YDR)) | Resonon Pika L, 256 × 256 px × 300 bands, ~384–1025 nm, raw uint8 JP2 + JSON | 1 cube | 205.3 GB | 0.95 GB |
| Thermal ([ASL0BV](https://doi.org/10.26249/FK2/ASL0BV)) | FLIR A70, 640 × 480, °C float32 TIFF | 5 frames | 114.1 GB | 0.53 GB |
| RGB ([FGLG94](https://doi.org/10.26249/FK2/FGLG94)) | Logitech C920, 1920 × 1080 JPEG | 15 frames | 194.5 GB | 0.90 GB |
| Depth ([SMHP1Y](https://doi.org/10.26249/FK2/SMHP1Y)) | pmd Pico Flexx, 224 × 171, depth + NIR intensity | 15 frames | 104.5 GB | 0.48 GB |

Layout after extraction: `<modality>/<bed>_<plant>/<YYYY_MM_DD>/<HH>/`.

### Data caveats (from the companion docs and our first pass)
- **HSI is raw.** Use the flat-field correction `C = (R − D)/(W·m − D)`, with the gray reference `W` from each `hsi_000.json`, the dark reference `D` = `black_reference.npy`, and `m` = 2.0.
- **HSI is not co-registered** with RGB, thermal or depth. RGB, thermal and depth *are* jointly calibrated (`analysis/combined/camera_calibration.py`).
- **Neighbouring plants are in view** in every image, and leaves overlap from about BBCH 12–14. Plant-level values need segmentation.
- The bundled HSI segmentation model was trained on **10 plants from W1/W2 only**. Check it on W3 and on wilted plants before relying on it.
- **About 6.4 % of runs are missing.** The full list is in `docs/data_notes.md`. Notably, all modalities are missing on 5 Sep 06 h and 22 Sep 12–18 h, and depth is missing from 19 Sep 15 h onward. Treat re-runs within ±1 h as the scheduled run.
- The 06 h and 18 h RGB/HSI images are often dark. HSI exposure is set once per run, so later frames can be overexposed.
- **Door closure on 2 Sep**, the day before irrigation stopped, changed the greenhouse microclimate (see §4B). Any before/after-cutoff contrast must account for it.
- The **leaf-clip plants and the RWC plants do not overlap**.

---

## 3. Compute & download strategy

The full camera set is **618 GB**. Store it on lab or cluster storage, never in git. Download with the companion tools:

```bash
git clone https://github.com/pippowell/ada2025 && cd ada2025
conda env create -f environment/environment.yml && conda activate ada2025
python utils/download_dataset.py --modalities hsi thermal --plants <IDs> --data-dir /path/to/ada2025_data --list-only
```

Download in tiers (HSI + thermal sizes computed from the repository listing):

| Tier | Plants | HSI + thermal | + RGB + depth | Purpose |
|---|---|---|---|---|
| 0 | Paper plants (W1/W2 × A2, A8, J5, R1, R7) | 14.2 GB | 27.4 GB | Reproduce the paper's numbers to validate our environment |
| 1 | 21 leaf-clip plants | 32.2 GB | 62.5 GB | Plant-level image ↔ turgor validation |
| 2 | 63 RWC-sampled W3 plants (`rwc_plant_ids.txt`) | 97.7 GB | 190.2 GB | RWC / EWT prediction models |
| 3 | All 216 plants | 319.4 GB | 618.4 GB | Bed-level index tracking incl. W3 and recovery |

Tiers 0 and 1 share two plants (W1_J5, W2_R7). Tiers 0–2 together are 92 plants and **141.3 GB** of HSI + thermal. A sensible Tier 3 alternative is a balanced subsample, e.g. 24 plants per bed (about 107 GB of HSI + thermal).

---

## 4. Preliminary results from the environmental data (exploratory)

### A. Leaf scans of the RWC leaves (W3, n = 70; Spearman ρ against days since cutoff)
| Variable | Day 1 → Day 12 | ρ |
|---|---|---|
| RWC | 92.9 → 83.1 % (lowest mean 77.9 % on 11 Sep) | −0.56 |
| EWT (leaf water mass / area) | 25.2 → 14.3 mg cm⁻² | **−0.87** |
| LMA | 25.6 → 72.6 g m⁻² | **+0.87** |
| NGRDI (RGB greenness) | 0.102 → 0.078, flat until about day 8 | −0.64 |

RGB colour responds late. EWT and LMA change early and strongly.

### B. Microclimate & leaf clips
- **Air ΔT (W3 − W1), daytime 10–17 h:** pre-cutoff −0.91 ± 0.65 °C. Drought +3.10 ± 0.78 °C. Rewatered +1.42 ± 0.52 °C.
  - Doors were closed on 2 Sep, and ΔT was already +1.7 °C that day. The cleaner drought estimate is **drought vs rewatered: +1.7 °C** (p < 0.001).
  - The daily PAR sum does not explain the difference.
  - This is *air* above the bed. Thermal imagery (Workstream B) will give real leaf − air values.
- **Leaf clips** (higher = lower turgor): by 14–15 Sep the dry-bed clips rose about **+15–17 units** above baseline (W2 +16.9, W3 +14.5; range +5 to +28 by sensor). Controls stayed at +0.6. W3 starts rising around 7 Sep and W2 about a day later. The clips **did not recover after rewatering**; the protocol notes that the clipped leaves did not regenerate.
- **Diel pattern:** W1 and W2 clips show a midday dip of about 3–4 units (raw signal). ⚠️ The **W3 clips show no dip even before the cutoff.** This looks like a hardware or ESP32 artifact. Check it before using W3 diel amplitude.

### C. RWC models from the RGB leaf scans (cross-validated R²)
| Target | Features | Grouped 10-fold | Leave-one-date-out |
|---|---|---|---|
| RWC | days since cutoff | 0.16 | 0.18 |
| RWC | scan colour/shape (RF) | 0.24 | 0.05 |
| RWC | scan + W3 clip shift (RF) | 0.34 | 0.12 |
| EWT | days since cutoff | 0.59 | 0.58 |
| EWT | scan + days (RF) | 0.62 | 0.36 |

**RGB leaf scans alone are not useful RWC predictors.** Most of their apparent skill comes from learning the date. This is why the hyperspectral data matters for Workstream C.

At date level (8 dates), the W3 clip shift tracks mean RWC (ρ = −0.95) and mean EWT (ρ = −0.93). Both also trend with time, so part of that correlation is shared trend.

---

## 5. Work plan

### Step 0 — Environment & repo
- [ ] Add the companion repo as a git submodule (`external/ada2025`) and use its environment
- [ ] Repo layout:
  - `analysis/` (our code)
  - `config.yaml` (data path; read it everywhere, no hard-coded paths)
  - `results/`, `figures/`
  - `data/` is git-ignored
- [ ] Download Tier 0 and reproduce the paper's results table (README "Expected results"). This is our pipeline sanity check.

### Step 1 — Workstream A: spectral index tracking
- [ ] Reflectance via the flat-field correction, plant masks via `generate_plant_masks.py` (check them on W3 and on wilted plants)
- [ ] Indices: NDVI, CI Red Edge, SIPI (paper definitions in `calc_spectral_index.py`), plus PRI, MCARI, red-edge position, and **water indices** WBI = R900/R970 and the 970 nm band depth
- [ ] Extend from 5 plants to all plants, 12 Aug – 22 Sep, beds W1/W2/W3, runs 09–15 h
- [ ] Mixed model: index ~ bed × day + (1 | plant). Use broken-stick or GAM fits to date the onset of divergence and to test recovery after 15 Sep
- [ ] Compare against the paper's 15 h / 5-plant results to show robustness

### Step 2 — Workstream B: canopy temperature & transpiration
- [ ] Start with `thermal_analysis.py` (median of the coolest 50 % of pixels against air in a 10-min window)
- [ ] Improve the leaf pixels: an RGB-derived vegetation mask reprojected to thermal via the calibration, since the coolest-50 % rule picks up the drying soil
- [ ] **Air reference:** W1 and W3 have their own sensors. **W2 has none.** Use the W1 sensor for W2 and test the sensitivity of the results to that choice
- [ ] Leaf − air temperature by bed × day × hour (all 5 runs). Diel shape and its change under drought; CWSI with wet/dry references if possible
- [ ] Correct for the 2 Sep door closure: restrict pre/post contrasts to 2–22 Sep, or include a door indicator
- [ ] Relate leaf − air temperature to VPD and to PAR (raw mV; ask the authors for calibration)

### Step 3 — Workstream C: turgor & RWC prediction
- [ ] **Leaf-clip plants (Tier 1):** for each of the 21 plants, match the image features at each run to the clip reading at that time. Use a per-plant time series and a mixed model. This gives a plant-level ground truth that the paper's bed-mean correlations do not have
- [ ] **RWC plants (Tier 2):** for each of the 63 W3 plants, take HSI + thermal (+ RGB/depth) from the run just before sampling on its sampling day. Features:
  - full-spectrum PLSR
  - water and red-edge indices
  - leaf − air temperature
  - canopy height and area from depth
- [ ] Targets: RWC, EWT, LMA (from scans + weights)
- [ ] Validation: grouped CV by plant **and** leave-one-date-out (primary). Report this alongside the "days since cutoff only" baseline, which a model must beat to be useful
- [ ] **Scale caveat:** the image spectra are canopy-level and RWC is one leaf. Expect a ceiling on predictive skill; consider masking the sampled leaf's position if identifiable

### Step 4 — Reporting
- [ ] Figures per workstream; one table of model skill; methods text and data dictionary
- [ ] Data-quality appendix (missing runs used, masks rejected)

---

## 6. Diurnal & environmental modeling (goals D and E)

### 6.1 Data available at each time scale
| Source | Resolution | Diel use | Limits |
|---|---|---|---|
| Camera runs (HSI, thermal, RGB, depth) | 06, 09, 12, 15, 18 h (+ re-runs within ±1 h) | 5 snapshots per plant per day | RGB/HSI often **dark at 06 and 18 h**. HSI exposure is fixed per run. ~6 % of runs missing (`docs/data_notes.md`) |
| Air T, RH, pressure (BME280) | 2 min | Continuous; VPD derived | W1 and W3 only, **no W2 sensor** |
| PAR (Driesen) | 15 min | Light driver; daily light integral | Raw mV, uncalibrated; one sensor for the whole greenhouse |
| Leaf clips (turgor) | 2 min, from 2 Sep | Continuous diel turgor on 21 plants | W3 clips show no pre-drought midday dip (possible artifact, §4B). Clips did not recover after rewatering |
| Irrigation | Fixed schedule: 11 h and 19 h, 8 min per bed (protocol) | Known water inputs (timing) | No flow or volume data. **Soil sensors unusable** |

### 6.2 Goal D — daily rhythm analysis
**Questions**
1. Does the **diel amplitude** of leaf − air temperature, turgor and light-sensitive indices change with stress, and in which order?
2. Do stressed plants respond more steeply to **VPD and PAR**? For example, the slope of leaf − air ΔT against VPD across the 5 runs.
3. Is there **morning–afternoon hysteresis**? At the same VPD, is a stressed plant warmer at 15 h than at 09 h because water has run down through the day?
4. Does **PRI** (photochemical reflectance index, 531/570 nm) show a deeper midday depression under stress? This indicates photoprotection, and the 5-run design makes it testable.

**Methods**
- [ ] Join every camera run to the environment at capture time. Use the per-frame timestamps (`frames_*.csv` and `hsi_000.json`) to get the 10-min mean T, RH, VPD and pressure for the bed's own air sensor, and the PAR of the nearest 15-min reading. Build one table: plant × run × features × environment
- [ ] **Per-plant, per-day response metrics:**
  - diel amplitude (max − min across runs)
  - slope of each signal against VPD and against PAR
  - 15 h − 09 h difference at matched VPD (hysteresis)
- [ ] Continuous diel models for the 2-min data (leaf clips, air): cosinor or cyclic-spline GAMs per bed and phase (pre-cutoff / drought / rewatered). Extract amplitude, acrophase and mesor (rhythm-adjusted mean), and their changes
- [ ] Mixed models:
  - `signal ~ VPD × bed × day + PAR + (1 + VPD | plant)`
  - random slopes give each plant its own sensitivity, which becomes a phenotype
- [ ] **Run-time confounds:** sun angle and lighting change between runs, and the HSI gray reference is taken per plant row per run. Compare runs only after flat-field correction, and keep the 12/15 h runs as the primary comparison
- [ ] Sensitivity checks: drop the 06/18 h runs for RGB/HSI-based metrics; check the door-closure effect (2 Sep) on the environmental drivers themselves

**Outputs:** diel response curves per bed × phase; a per-plant "stress sensitivity" phenotype (the VPD slope); and the order in which diel signals change (this feeds the early-warning race in §7).

### 6.3 Goal E — context-aware irrigation algorithms
**Framing:** this greenhouse ran a fixed schedule, so the data can't show what would have happened under a different schedule (no counterfactual soil water, no soil moisture, no ET). What we *can* test is the **trigger**: does a rule that combines plant signals with microclimate flag water deficit early and reliably?

**Candidate rules (simplest first)**
1. **Microclimate-only baseline:** irrigate when cumulative VPD × PAR since the last watering exceeds a threshold. No plant sensing.
2. **Plant-signal threshold:** leaf − air ΔT, or CWSI, above a threshold at the 12/15 h run.
3. **Context-aware (main candidate):** leaf − air ΔT *normalized for VPD and PAR*. This is the residual against a VPD/PAR baseline fitted on watered W1 plants, so a hot dry afternoon doesn't trigger a false alarm.
4. **Multimodal model:** gradient-boosted or logistic classifier on HSI water/red-edge indices + thermal + microclimate + time of day, predicting "water deficit present".
5. **Turgor-informed:** add leaf-clip readings where available, as an upper bound on what a contact sensor adds.

**Evaluation (each rule with the same protocol)**
- [ ] Label: plant is in a water-deficit bed and ≥ N days after the cutoff (N from the §7 detection analysis). Also run a stricter label anchored on RWC and turgor
- [ ] Metrics:
  - **detection lead time** before logged wilting (8 Sep)
  - **false-alarm rate on W1** (never stressed)
  - recall on W2/W3
  - time to "all clear" after rewatering
- [ ] **Replay mode:** step through 3–22 Sep run by run and log when each rule would have triggered irrigation, plant by plant and bed by bed
- [ ] Water use is reported only as **trigger counts and timing**, not litres saved: there is no flow data. Any volume estimate must be labelled as assumption-based

### 6.4 Train/test splits (the authors provide none, so we define them up front)
- [ ] **No random row splits.** Runs from the same plant and nearby days are strongly autocorrelated
- [ ] Primary splits:
  - **bed-held-out:** train on W1 + W2, test on W3 and the reverse. W3 has its own air sensor, so this also tests transfer across microclimates
  - **time-blocked:** train on early days, test on later days, with a gap of ≥ 1 day between them
  - **plant-grouped:** grouped k-fold by plant
- [ ] Freeze the split definitions in `splits.yaml` before modelling, and publish them in the repo so others can reuse them
- [ ] Report all metrics per split; the bed-held-out result is the headline

### 6.5 Generalization caveats (from the authors; state them in every output)
- The dataset is a **data resource, not an ML benchmark**. All preprocessing and splits are ours and must be documented
- **One greenhouse, one variety** (CALLEDIA KWS Öko), one season, about six weeks. Open-field conditions differ: wind, full sun, soil heterogeneity, larger VPD ranges
- Models therefore need **domain adaptation** before use on other crops or in the field. Prefer physically grounded, normalized features (CWSI-style VPD normalization, water-band indices) over raw pixel values, because they transfer better
- Field validation path:
  - check that the context-aware rule holds on an open-field drought dataset (e.g. a rainout-shelter experiment)
  - re-fit only the W1-equivalent VPD baseline per site

---

## 7. Hackathon UI — "How many days of warning does each sensor give?"

**Pitch:** stress you can see in a photo arrives too late to act on. The UI shows *which* sensing method detects water deficit first, and how many days of warning a farmer gets before visible wilting (logged 8 Sep). It ties all three workstreams (§1 A–C) to one irrigation decision.

### 7.1 Screens
| # | Screen | What judges see | Supports | Farmer benefit |
|---|---|---|---|---|
| 1 | **Early-warning race** (hero) | One timeline, days after the 3 Sep cutoff. A bar for each signal: air ΔT, thermal leaf − air, leaf-clip turgor, WBI (970 nm), NDVI / CI Red Edge / SIPI, RGB greenness. A line marks the logged wilting | A, B, C | Days of warning before visible damage = the window for an irrigation decision |
| 2 | **Live bed map** | 216 pots in their real checkerboard positions, a time slider (41 days × 5 runs), and layer toggles (NDVI, SIPI, leaf − air, turgor, predicted RWC). Animation shows W2/W3 diverging from W1 | A, B, C | Spot field areas that need water before others |
| 3 | **"What your eyes miss"** | The same plant on the same day as photo vs NDVI vs thermal. Around day 4–5 the photo still looks healthy | A, B | Why to invest beyond visual scouting |
| 4 | **Plant profile** (click a pot) | Reflectance spectrum over time (970 nm water dip, red edge highlighted), thermal frame, turgor trace (clip plants), leaf scan + measured vs predicted RWC (63 RWC plants) | A, B, C | Plant-level diagnosis and the ground truth behind it |
| 5 | **Irrigation advisor** (uses §6.3 rules) | Per-plant traffic light (water now / watch / fine) from leaf − air thresholds and predicted RWC. Replays the experiment to show when each plant would have been flagged vs logged symptoms | B, C, E | An action, not just a chart |
| 6 | **Sensor ladder** | Cost tier (≈$5 air sensor → phone RGB → thermal camera → hyperspectral) vs days of warning | A, B | Which sensor is worth buying; cheap options may capture most of the signal |
| 7 | **Recovery** | Which stressed plants rebounded after rewatering on 15 Sep (indices, turgor) | A, C | Is late irrigation still worth it? (Not covered in the dataset paper) |
| 8 | *Optional* **3D plant** | Point cloud coloured by temperature (companion repo fusion code) | B | Wow factor; keep it out of the critical path |

### 7.2 Methods behind the hero chart
- [ ] **Detection-day rule (pre-registered, one rule for every signal):** the first day on which the dry-bed minus control difference stays above the pre-cutoff noise band (e.g. mean + 2 SD of 25 Aug – 2 Sep) for ≥ 2 consecutive days. Report a bootstrap CI over plants
- [ ] Use the 2–22 Sep window for contrasts sensitive to the door closure (§2 caveats)
- [ ] Logged symptom anchors: wilting 8 Sep, yellowing 12 Sep, severe stress 15 Sep (protocol sheet); BBCH/symptom notes for plant-level checks
- [ ] Advisor thresholds are set on W2 and **validated on W3** (or vice versa), so the replay is not fitted to itself
- [ ] Sensor costs: approximate list prices with sources, labelled as approximate

### 7.3 Honesty panel (on screen, not hidden)
- Leave-one-date-out skill next to grouped-CV skill for every RWC model (the leaf-scan models in §4C collapse when whole dates are held out)
- Scope: one greenhouse, one variety, about six weeks, three beds. **No yield data**, so the claims are about warning time and water-status detection, not yield or profit
- Known confounds: door closure on 2 Sep, no air sensor at W2, neighbouring plants in view, ~6 % missing runs

### 7.4 Build plan
- [ ] **Precompute offline** (lab machine or cluster): one row per plant × run with indices, leaf − air, canopy area/height, predicted RWC → `ui_data/plant_runs.parquet` (≈ 40k rows, a few MB). Never ship raw images to the app
- [ ] Thumbnails (RGB, NDVI, thermal) for about 20 showcase plants: the clip plants, a few RWC plants, and controls
- [ ] Stack: **Dash + Plotly** (team decision, `docs/SOFTWARE_STACK.md`); one page file per screen in `app/pages/`, all reading the same parquet
- [ ] A prototype on the environmental data alone (air, turgor, RWC, leaf scans) can be built now, with image layers plugged in after the Tier 0–2 downloads
- [ ] **3-minute demo:** farmer's problem → "what your eyes miss" → bed-map animation → early-warning race → advisor replay → sensor ladder

| Role | Owns |
|---|---|
| Data | Downloads, preprocessing to `plant_runs.parquet`, thumbnails |
| Science | Detection-day rule, models, honesty panel numbers |
| Front end | Screens 1–5, map, slider |
| Story | Pitch, demo script, cost sources, slides |

---

## 8. Open questions (for the team and the dataset authors)
1. **Clock time of RWC sampling** on each day. Needed to pick the right imaging run.
2. **W3 leaf clips** show no midday dip even before the cutoff. A known hardware or wiring issue?
3. **Leaf-clip units:** a conversion to thickness or pressure, or per-clip calibration?
4. **PAR calibration** (mV → µmol m⁻² s⁻¹) for the Driesen sensor.
5. Are the **DFKI air sensors** (5534 at W1, 4965 at W3) available? They would also help check the BME280 readings.
6. Plans for the **automatic quality labels** that are still marked TODO in `data_notes.md`?
7. **Irrigation volume:** flow rate or litres per 8-min watering event, so trigger counts can be converted to water use
8. **Storage & compute:** which server or cluster will hold the data (Tiers 0–2 HSI + thermal ≈ 141 GB)?

Contact for dataset questions: Piper Powell (Universität Osnabrück), listed as the dataset contact on osnaData.

---

## 9. Files from the first pass
| File | Description |
|---|---|
| `analysis/features.py` | Leaf-scan segmentation, scan-to-plant mapping, RGB features |
| `analysis/rwc_merge.py` | Joins RWC with scan features; computes RWC, EWT, LMA |
| `analysis/analysis.py` | Microclimate, leaf-clip, soil, scan-based models, figures |
| `results/*.csv`, `results/results.json` | Model performance, per-leaf table, date means, summary stats |
| `results/rwc_plant_ids.txt` | The 63 W3 plants with RWC samples (Tier 2 download list) |
| `figures/fig1–fig5` | First-pass figures |

The first-pass scripts and results are not committed to the team repo yet (the team rule is no building before the master plan). They still use local paths; switch them to a config file before others run them.
