# Yuming — Work Plan: Challenge 4, Diurnal & Environmental Modeling

- **Owner:** Yuming
- **Challenge:** 4 — diurnal & environmental modeling (daily rhythms, context-aware irrigation)
- **Dataset:** ADA 2025 ([paper](https://arxiv.org/html/2610.23609v1), [code](https://github.com/pippowell/ada2025), data DOI `10.26249/FK2/DMDLTE`)
- **Time box:** 2026-11-14 and 2026-11-15. The brief's submission deadline is **Sunday 2026-11-15, 12:00 PM** (see question 5).
- **App framework:** Dash + Plotly, as decided in `docs/SOFTWARE_STACK.md`
- **Status:** plan only in this repo. On my own machine the environmental
  sub-dataset is analysed and thermal data for 27 plants is downloaded (see
  section 2). I did that, and wrote two exploratory scripts, before I read the
  build-after-master-plan rule. Nothing from them is in this pull request, and
  I will not build further until the master plan is agreed.

**Overlap with Sam's plan.** Sam's plan
([PR #1](https://github.com/MasterKinjalk/UiWUE/pull/1)) also covers
challenge 4: his experiments S4 (daily response to VPD and light) and S5
(replay of three irrigation rules), and his irrigation advisor page. I did not
know that when I started. Section 1 lists what this plan adds on top of his,
and question 1 asks the team to decide who owns what.

Numbers in this plan come from my own first pass unless marked otherwise.
Items marked *(unverified)* still need checking.

---

## 1. What I am building

A tool that answers two questions for a grower with a thermal camera:

1. **When in the day should plants be scanned to catch water stress earliest,
   and can one flyover a day replace five?**
2. **Can the stress alert work without a well-watered control bed**, the way a
   real grower would have to run it?

What this adds to Sam's challenge 4 work:

| Topic | Sam's plan | This plan |
|---|---|---|
| Daily response to VPD and light | S4: per plant-day slopes, hysteresis, cosinor fits, spectral and thermal | thermal only; adds measured outside radiation to split sunny from overcast flyovers |
| Which time of day to scan | not covered | separability of stressed and control plants by day and time of day; alert day for each flyover schedule |
| Reference for the stress signal | leaf − air, bed's own air sensor, VPD-normalised on W1 | leaf temperature against the control bed in the same run, because the two air sensors disagree (section 2); plus a score that needs no control bed |
| Irrigation rules | S5: replay of three rules, lead time before wilting | same replay idea; adds a published sugar beet baseline, a held-out false-alert check, and demand built up since the last watering |
| External data | none yet (his section 11) | weather-station radiation and a published baseline (section 2) |

High-level functions of the software for this part:

1. **Bring the data together:** thermal flyover images, greenhouse climate readings and outside weather become one record per plant and flyover.
2. **Show the daily rhythm:** how a plant's or bed's leaf temperature moves through the day against its healthy pattern and the conditions at the time.
3. **Score water stress:** a stress score per plant and flyover with a confidence level, with or without a well-watered reference bed.
4. **Raise an irrigation alert:** a plain status per plant or bed (fine, watch, irrigate), with the reason and the evaporative demand built up since the last watering.
5. **Advise on the scan schedule:** which time or times of day to scan, and what is lost by scanning less often.
6. **Say when it does not know:** flag flyovers that carry little information (overcast, early morning, missing runs or sensors) and hold the alert back when evidence is weak.
7. **Report how well it works:** alerts compared with the leaf clips and manual symptom notes, with the false-alert rate.
8. **Hand results to the unified tool:** stress score, alert status and recommended scan time in the shared table.

## 2. Data I need

### ADA 2025

| Item | Use | State on my machine |
|---|---|---|
| Thermal, 27 plants (14.3 GB): the 21 leaf-clip plants plus W1 A2, A8, R1, R7, F5, N3 | leaf temperature at all five daily runs | downloaded, checksums passed. About 10 minutes at 30 MB/s |
| `air_readings.parquet`, `leaf_readings.parquet`, `metadata_rwc_par_en.xlsx` | air temperature, humidity, light, clip ground truth, symptom notes | downloaded and analysed |

This is 25 of the team's 29 core plants. The four missing ones are the W2
paper plants A2, A8, J5 and R1 (about 2 GB of thermal), which I can add.

### External data

The brief allows public data if source, license and role are documented.

| Source | License | Role | State |
|---|---|---|---|
| DWD station Belm (id 00342, 11 km from the greenhouse): 10-minute global radiation, air temperature, humidity, from `opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/` | CC BY 4.0 (DWD terms of use); cite Deutscher Wetterdienst | label each flyover sunny or overcast; give the on-site light sensor (raw mV) a physical scale | files confirmed to cover Aug–Sep 2025; not downloaded |
| Idso (1982) sugar beet baseline: canopy − air = 2.50 − 1.92 × VPD | published value | a stress score that needs no control bed | *(unverified)* quoted from a 2022 paper; original table and VPD unit not yet checked |
| Optional: Open-Meteo archive API for the site | CC BY 4.0 *(unverified, from the AWS registry entry)* | outdoor evaporative demand in millimetres | tested, returns Sep 2025 data |
| Stretch: Lisbon vineyard thermal dataset (Mendeley `p4ts8hy99s`) | CC BY 4.0 | check the time-of-day finding in another crop | size and layout unknown |

Considered and left out: Eschikon sugar beet set (no thermal, images every two
weeks), Sugar Beets 2016 (5 TB), PhenoBench (segmentation only), TERRA-REF
(no confirmed drought subset).

### First-pass findings the team can use now

- **The two air sensors disagree in the afternoon.** Over 1–16 Sep the W3-end
  sensor reads warmer than the W1-end sensor by a mean of 2.2 °C at 12 h,
  2.9 °C at 15 h and 3.7 °C at 18 h (up to 7.3 °C), and 0.5 °C cooler at 06 h.
  Leaf − air computed with each bed's own sensor would carry this difference
  into any comparison between beds. This bears on Sam's question 4.
- **Folder hour is the run's start hour, not the capture time.** For W1_A2 on
  7 Sep the "15" run was captured at 15:39 local time. Air readings should be
  matched to the timestamps in `frames_thermal.csv`.
- **Leaf clips, per plant.** The three control clips stay within ±3 raw units
  of their 2–3 Sep level until 19 Sep. Treated clips peak 9 to 35 units above
  it. The rise starts on different days: 4 Sep for W2_H5, between 6 and 10 Sep
  in W3, between 7 and 13 Sep for the rest of W2. W2_K2 and W2_R7 respond
  weakly. Only W2_L7 and W2_O6 return to near baseline after re-watering.
- **Clip daily cycle.** W1 and W2 clips dip by 3–4 units at midday; W3 clips
  do not, before or after the cutoff. This matches Sam's note and argues
  against using W3 clips for daily-rhythm work.
- **Earlier wilting notes exist.** The per-plant symptom sheet has "wilting"
  on 6 Sep for W2_A6 and W3_B7, two days before the 8 Sep date used for lead
  time in Sam's plan.
- **Thermal files** are ZSTD-compressed TIFFs and need `imagecodecs`, which is
  already in `requirements.txt`. Reading one plant (206 runs) takes about 6 s.

## 3. Pipeline

```
thermal TIFFs + frames_thermal.csv        air / light parquet        DWD radiation
  -> leaf temperature per run               -> value at capture time    -> value at capture time
     (coolest-50 % median, authors' rule)      (both air sensors)          (sunny / overcast)
              \______________________________|____________________________/
                                             v
              one row per plant per run (bed, plant, date, hour)
                                             v
     reference: control-bed median leaf temperature in the same run
     baseline:  the plant's own healthy history at the same time of day
                                             v
            stress score -> alert status -> schedule comparison
```

| Step | Input | Output | Method |
|---|---|---|---|
| Leaf temperature | 5 thermal TIFFs per run | leaf temperature, background temperature, frame count | authors' `thermal_analysis.py` rule; a re-run within ±1 h stands in for a failed run |
| Environment | air parquet, light sheet, DWD files | drivers at capture time | nearest reading to the median frame timestamp |
| Control-bed score | leaf temperature | score per plant and run | plant minus control-bed median in the same run, as a z-score against the plant's own pre-cutoff values at that time of day |
| No-control-bed score | leaf − air, VPD | score per plant and run | distance above a well-watered line: (a) the Idso line, (b) a line fitted on healthy data from all beds |
| Alert | scores | fine / watch / irrigate | score above a threshold on two scans in a row; threshold set so the control bed raises no alert |
| Schedule comparison | alerts | alert day per schedule | each single time of day, 12 h + 15 h, all five |

## 4. Outputs

`data/derived/diurnal/thermal_runs.parquet`, one row per plant per run, with
the shared keys `bed`, `plant`, `date`, `hour`.

| Column | Type | Example |
|---|---|---|
| `bed`, `plant`, `date`, `hour` | shared keys | `W2`, `J5`, `2025-09-08`, `15` |
| `slot`, `timestamp_utc` | int, datetime | scheduled slot; median frame time |
| `treatment`, `days_since_cutoff` | str, int | `deficit`, `5` |
| `leaf_temp`, `leaf_temp_p10`, `bg_temp` | float, °C | coolest-50 % median; stricter variant; warmest-50 % median |
| `air_T_w1`, `air_T_w3`, `rh_w1`, `rh_w3`, `vpd_w1`, `vpd_w3` | float | both air sensors at capture time |
| `par_mV`, `rad_dwd_wm2`, `sky` | float, float, str | on-site light, outside radiation, `sunny` / `overcast` |
| `score_ref`, `score_idso`, `score_local` | float | the three score variants |
| `alert` | str | `fine` / `watch` / `irrigate` |
| `evidence` | str | `ok` / `weak` (overcast, early morning, missing input) |
| `n_frames_ok`, `qc_*` | int, bool | QC |

`leaf_temp` uses the same name and definition as Sam's trait, so one thermal
table can serve both plans.

## 5. Experiments or analyses

| # | Question | Data | Method | Output |
|---|---|---|---|---|
| Y0 | Can the results be trusted? | all 27 plants | coverage against `data_notes.md`; no separation of beds before the cutoff; 15 h leaf − air trend in W2 against W1 has the paper's sign; DWD radiation tracks the on-site light sensor | pass or fail for each check |
| Y1 | Which time of day shows stress first? | all runs, 2–15 Sep | separability of treated and control plants (AUC) per day since cutoff and time of day | day by hour heatmap |
| Y2 | Can one flyover a day replace five? | all runs | alert day under each schedule, against each plant's clip onset and first wilting note | alert day and lead time per schedule |
| Y3 | When is a scan informative? | all runs + DWD | stress signal against VPD and against outside radiation, sunny and overcast flyovers apart | conditions under which a scan should be trusted |
| Y4 | Does the alert work without a control bed? | all runs | Idso line and local line against the control-bed score: alert day and false alerts | cost of having no control bed |
| Y5 | How much demand builds up before stress starts? | clip plants | cumulative on-site VPD from the last watering to clip onset and to the alert | a demand threshold for the irrigation rule |
| Y6 | How often does it raise a false alert? | healthy data not used for the baseline: all plants 29 Aug–2 Sep, control bed 3–14 Sep | count alerts for every score variant | false-alert rate next to every result |
| Y7 | Stretch: does the time-of-day result hold in another crop? | Lisbon vineyard set | thermal signal against stem water potential by time of day | one comparison chart |

Bed roles: the time of day and schedule are chosen on W2 and checked unchanged
on W3. Baseline to beat: the dataset paper's setup, one 15 h flyover scored as
leaf − air. Priority if time runs short: Y0, Y1, Y2, Y6, Y4, Y3, Y5, Y7.

For bed-level onset I use Kinjalk's rule (first day the treatment-control
contrast is significant and stays so). The per-plant alert above is separate
and feeds the irrigation status.

## 6. App pages I will deliver

Described by function only; layout follows the app shell once it exists.

1. **Daily rhythm** (`app/pages/diurnal_rhythm.py`): pick a plant or bed and a
   date range; see leaf temperature through the day against its healthy
   pattern, with the conditions at each flyover and the evidence flag.
2. **Scan schedule** (`app/pages/scan_schedule.py`): the day by hour
   separability map, and the alert day and false-alert rate for each flyover
   schedule, so a grower can see what scanning less often costs.

For the irrigation advisor I propose not to build a second page. My scores
(control bed, Idso line, local line) and the evidence flag go into the shared
table for Sam's `irrigation.py` to display, if the team agrees (question 1).

## 7. Schedule

| When | Work | Done when |
|---|---|---|
| Sat evening, block 1 | unpack thermal, build the per-run table, fetch DWD, Y0 | four checks reported |
| Sat evening, block 2 | Y1, Y2, Y6 | heatmap, schedule table, false-alert counts |
| Sat late | Y4, Y3, Y5 | three score variants compared; table published to the team |
| Sun early morning | two Dash pages on the shared table | pages run in the team app |
| Sun by 10:00 | numbers and one chart for the joint story; text for the form and video | handed to whoever owns the submission |

## 8. Risks

| Risk | Effect | Response |
|---|---|---|
| One control bed | treatment is confounded with bed | each plant is scored against its own history; Y0 checks that beds do not separate before the cutoff |
| Doors near the plants were closed before the cutoff (2 Sep per Sam's notes; the protocol sheet confirms the closure) | my baseline window (24 Aug–2 Sep) mostly predates it | treat 2 Sep and the 3 Sep morning runs, when all beds were still watered, as the cleanest check; re-baseline on those if a step shows up |
| Only 9 control plants | false-alert rate is loosely estimated | report counts, not only rates; add the held-out healthy window (Y6) |
| Beds are scanned about 10 minutes apart | noise at 09 h and 18 h, when temperature changes fast | per-plant, per-time-of-day baseline; report the noise by time of day |
| Coolest-50 % pixels include neighbours and drying soil | leaf temperature biased in stressed beds | compare with the coolest-20 % variant; use Sam's masked variant if it exists |
| 06 h thermal run missing on 5, 6, 7 and 10 Sep | 06 h schedule is handicapped | report coverage with every schedule result |
| Idso score depends on absolute temperatures | inherits the camera's ±2 °C error and the air-sensor disagreement; the field baseline may not fit a greenhouse | always show the local-line score beside it; report the offset of the control bed from the Idso line |
| Outside weather is not greenhouse climate | wrong demand estimates | outside data is used for light conditions only |

## 9. Questions for the master plan

1. **Who owns challenge 4?** Sam's plan and this one overlap. One option: I
   take the thermal daily rhythm, scan schedule and no-control-bed score, and
   publish the per-run thermal table; Sam keeps the spectral side, the RWC and
   turgor models and the irrigation advisor page, which reads my scores.
2. **What does `hour` mean?** Kinjalk's schema stores the folder hour in `hour`
   and the scheduled slot in `slot`. Sam's stores the scheduled slot in `hour`
   and the folder hour in `capture_hour`. The tables will not join until one
   is chosen.
3. **Air reference.** Given the afternoon disagreement between the two air
   sensors, I propose that comparisons between beds use either the control bed
   as reference or one sensor for all beds, never each bed's own sensor.
4. **Shared thermal data.** I have thermal for 25 of the 29 core plants and can
   add the missing four. Where should the per-run thermal table live so Sam
   and Kinjalk can use it today?
5. **Deadline.** The brief says Sunday 12:00 PM, with a 2-minute video.
   Schedules with work on Sunday afternoon would miss it. When do we freeze,
   and who records the video?
6. **External data.** Is the team fine with DWD radiation and the Idso
   baseline? The brief asks for source, license and role, so I suggest one
   `docs/DATA_SOURCES.md` for everyone.
7. **Wilting date.** Lead times in Sam's plan use 8 Sep. The per-plant sheet
   has wilting notes on 6 Sep. Which date does the team use?
