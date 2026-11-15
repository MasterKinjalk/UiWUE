# Result card: daily rhythm and irrigation timing (Yuming)

Data: thermal camera for 39 plants (13 watered controls, 17 in W2, 9 in W3),
24 Aug – 22 Sep 2025, all five daily runs (5,402 plant scans), plus the
greenhouse air and light sensors. Day 0 = 3 Sep, when irrigation of W2 and W3
stopped. Outside data: the German weather service station at Belm, 11 km away,
and Idso's published sugar beet baseline (sources below).

## Three numbers

1. **Day 4, a day or two before the leaf clip.** One thermal flyover a day at
   09 h, each plant scored against the control bed, flags all **9 of 9**
   held-out stressed plants (W3) by **day 4**, 1 day before the leaf clip
   registers turgor loss (2 days on W2). There are **0 of 9** false alerts on
   held-out watered plants and **0 of 18** before the cutoff. **15 h is the
   second choice**: W3 by day 3, but W2 only by day 6 (16 of 17 plants). The
   paper's set-up (one 15 h flyover, leaf minus air temperature) needs until
   **day 8**, 2–3 days *after* the clip.
2. **No control bed needed.** Scoring each plant against a healthy-plant line
   (leaf minus air temperature as a function of VPD, from air temperature and
   humidity only) also flags **9 of 9** by **day 4**, with no false alerts. It
   needs all five daily flyovers to get there.
   - **Against the published line.** The sugar beet line Tc − Ta = 2.50 − 1.92 ×
     VPD (Idso 1982) reaches **day 5** with 12 h and 15 h flyovers, against day 6
     for raw leaf minus air on the same flyovers.
   - **Not as published.** Watered greenhouse beets sit **2.0 °C below** that
     field line at 12 h and 15 h. "Above the line" without re-tuning flags 5 of
     13 watered plants and 7 of 22 plants before the cutoff, so the threshold
     has to be set on site.
   - **Why it sits lower.** Idso's line was fitted on sunlit canopies in
     Arizona. In the same paper, shaded canopies sit on average 3.8 °C below
     the sunlit line. A greenhouse gets less light, and our leaf temperature is
     the coolest half of the pixels, so an offset in this direction is
     expected.
3. **Time of day and sky matter more than flyover count.**
   - Stressed and watered plants separate (AUC ≥ 0.9 on two days running) from
     **day 3 at 09 h**, day 4 at 15 h, day 5 at 12 h and day 7 at 18 h.
   - **The 06 h run never separates them**: before dawn, leaves are not
     transpiring, so they are not cooler than a stressed leaf.
   - **Sunny flyovers show a larger stress gap**: about **+2 °C** at 09 h and
     15 h against cloudy flyovers on the same stress day, and +0.5 °C at 12 h.
     Sunny or cloudy is judged from the weather station's sunshine record,
     independent of the greenhouse.

## Figure

![Separability by day and time of day](yuming_separability.png)

Each cell: how well the thermal score separates deficit from control plants
(AUC; 0.5 = no difference, 1 = perfect). Red appears first in the 09 h and
15 h rows.

## Two limits

1. **One control bed.** Before the cutoff, when every plant was watered, the
   beds already differ at 06 h (AUC 0.26) and 09 h (0.62), so part of the 09 h
   lead may be a bed difference rather than stress. Correcting for the air
   warming between bed scans did not remove it. 15 h is close to no
   difference (0.56), which is why it is the second choice. Thresholds were
   tuned on W2 with only 4 control plants; W3 and the other 9 controls were
   never used for tuning.
2. **No counterfactual irrigation.** The greenhouse watered on a fixed
   schedule and the soil sensors are unusable, so the results say *when* a
   rule would have fired, not how much water it would save. The
   "demand since last watering" rule fires on day 1 only because it knows
   irrigation stopped; it is a reference, not a stress detector.

Also: the sunny-against-cloudy split rests on 3–7 runs per time of day, so it
shows a direction, not a precise size.

## Outside data

| Source | Use | Licence and credit |
|---|---|---|
| DWD station Belm (id 00342), 10-minute global radiation and sunshine duration, file `10minutenwerte_SOLAR_00342_20200101_20251231_hist.zip` | Sky at each flyover (sunny, cloudy, dark). Puts the greenhouse light sensor on a W/m² scale: hourly r = 0.92 with outside radiation, 1 mV ≈ 110 W/m². This is outside-equivalent, so it includes losses through the greenhouse cover. | CC BY 4.0, "Source: Deutscher Wetterdienst" |
| Idso, S.B. (1982), Non-water-stressed baselines: a key to measuring and interpreting plant water stress, *Agric. Meteorol.* 27:59–70, Table I: sugar beet, sunlit, N = 47, r = 0.898, °C and kPa | Published sugar beet baseline for the "Leaf − air above the published line" rule | Coefficients cited from the paper; the paper itself is not redistributed |

## Reproduce

```bash
export UIWUE_DATA=/path/to/ada2025/data     # raw thermal + environmental files
python -m uiwue.env                          # downloads the DWD file (1.5 MB) on first run
python -m uiwue.thermal
python -m uiwue.diurnal                      # prints the checks and the schedule table
```

Pages: Daily rhythm and Irrigation advisor (`cd app && python app.py`).
