# Submission form (draft)

Status: draft, Sat 14 Nov 2026, 20:00. Owner: Yuming.

- **Bracketed items:** each owner fills their own from their result card
  (`docs/results/<name>.md`) by Sun 09:30.
- **Then:** paste into the iKompete portal by Sun 11:30.
- **Portal fields:** project title, domain lead, technical lead, track and
  challenge, project overview (problem statement, solution, technical
  approach), video link.

---

**Project title:** UiWUE: when does a sugar beet need water? A multimodal
water-stress workbench

**Domain lead:** [team to name]

**Technical lead:** [team to name]

**Track and challenge:** Track 1, Challenge 1: Detect and characterize
water-deficit stress using multimodal plant sensing (ADA 2025)

**Demo video:** [link, unlisted. Check it plays in a logged-out browser
before submitting]

## Project overview

### Problem statement

A plant biologist running a water-deficit experiment with automated imaging
collects tens of thousands of thermal, depth, RGB and hyperspectral images.
Three decisions depend on them:

1. Which sensor and trait to trust as an early stress indicator.
2. When in the day to image.
3. When a given plant needs water.

Answering them today means a custom script per sensor and a separate
statistical check per question.

ADA 2025 is the test case: 216 sugar beets in three beds, imaged five times
a day. Irrigation of two beds stopped on 3 Sep 2025 and resumed 12 days later.
The question we put to it: **how many days of warning does each sensor give
before a stressed plant shows it, and which flyover should trigger watering?**

### Solution

UiWUE is a workbench. The biologist points it at the image folders. It:

- extracts per-plant traits from each sensor into one table;
- tests when stress became detectable in each signal, against the watered
  control bed and each plant's own healthy days;
- turns that into a plain status per plant: fine, watch, or irrigate, with the
  reason and how far to trust it.

Workflow:

```
image folders + greenhouse sensors + weather station
  -> one extraction module per sensor (thermal, 3D depth, RGB, hyperspectral, ground truth)
  -> one run-level table: bed, plant, date, scheduled flyover
  -> shared statistics: baselines, placebo window, onset day, held-out check
  -> app: Home · Plant profile · Bed map · 3D Browser · 3D Compare · 3D Export
          · [Early warning] · [Sensor comparison] · Daily rhythm · Irrigation advisor
```

The user acts on the Irrigation advisor:

- **Which plants:** the bed map shows which plants to water today.
- **Why:** each flag names the signal that raised it and the alert rule's
  track record.
- **How far to trust it:** flyovers with weak evidence (missing data, or a
  light reading under half the usual for that time of day) are skipped and do
  not count toward an alert. The weather service's sky record is shown next
  to each flyover.

### Technical approach

- **Thermal.** Leaf temperature per plant and flyover is the median of the
  coolest half of the pixels (the dataset authors' rule).
  - It is scored three ways: against the control bed in the same run;
    against the plant's own healthy days at the same time of day; and
    against a healthy leaf-minus-air line fitted on the healthy days before
    the cutoff, so no control bed is needed.
  - Also tested: Idso's (1982) published sugar beet baseline.
- **3D (depth).** [Kinjalk: traits, e.g. height and projected area, and how
  they are extracted]
- **Hyperspectral.** [Sam: indices]
- **RGB and detection.** [Jack: model and sensor benchmark]
- **Ground truth.** Leaf-clip turgor onset, written symptom dates, and leaf
  relative water content.
- **Environment.**
  - Greenhouse air sensors give VPD (the air's drying power).
  - The German weather service station 11 km away gives radiation and
    sunshine, which label each flyover sunny or cloudy.
- **Alert rule.** A plant is flagged "irrigate" when its score is above a
  threshold on two usable scans in a row.
- **Leak-free validation.**
  - Thresholds are set on bed W2 with 4 control plants, then checked
    unchanged on bed W3 and the other 9 control plants.
  - Days before the cutoff serve as a placebo window: no alert may fire
    there.
  - Whole plants and whole beds are held out, never single scans.
- **Baseline to beat.** The dataset paper's set-up: one 15 h flyover, scored
  as leaf minus air temperature.

### Results

- **Thermal, one flyover at 09 h** (scored against the control bed):
  - **Flagged:** all 9 held-out stressed plants by **day 4**.
  - **Earlier than the leaf clip:** 1 day before it registers turgor loss
    (2 days on the tuning bed).
  - **False alerts:** 0 of 9 held-out watered plants, and 0 of 18 plants
    before the cutoff.
  - **Paper baseline:** needs until day 8, 2–3 days *after* the clip.
  - **Second choice:** 15 h.
- **No control bed needed.** Scoring against a healthy-plant line (air
  temperature and humidity only) also reaches day 4 with no false alerts.
  - It needs five flyovers a day.
  - Idso's published line works only once its threshold is set on site:
    watered greenhouse beets sit 2 °C below it.
- **Time of day.**
  - The 06 h flyover never separates stressed from watered plants.
  - On sunny flyovers the stress signal is about 2 °C larger than on cloudy
    ones.
- **3D.** [Kinjalk: headline number]
- **Early warning across sensors.** [Jack: days of warning per sensor, and the
  cheapest set-up that comes close]
- **Hyperspectral.** [Sam: headline number]

### Limits, and what deployment would need

- **One control bed.** Treatment and bed position are confounded. Before the
  cutoff, the beds already differ slightly at 06 h and 09 h.
- **One greenhouse, one variety, one season.** Thresholds must be re-set on
  site; the published field baseline does not transfer as is.
- **Fixed watering schedule.** The results show *when* a rule would have
  fired, not how much water it would save.
- **Deployment needs** a well-watered reference group, or a few healthy
  days to fit the no-control line. Also: an air temperature and humidity
  sensor, a midmorning-to-afternoon thermal flyover, and a sky record.

### Data and sources

- **ADA 2025** (the central input): Powell, Franz, Smith, Ihorst (2026),
  osnaData, https://doi.org/10.26249/FK2/DMDLTE
- **DWD station Belm** (id 00342): 10-minute radiation and sunshine, CC BY 4.0,
  "Source: Deutscher Wetterdienst". Used to label each flyover's sky and to put
  the greenhouse light sensor on a W/m² scale.
- **Idso, S.B. (1982)**, *Agricultural Meteorology* 27:59–70: published sugar
  beet non-water-stressed baseline. Used as a comparison score.
- [Others: any further outside data, with source, licence and role]
