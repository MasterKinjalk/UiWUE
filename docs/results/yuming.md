# Result card: daily rhythm and irrigation timing (Yuming)

Data: thermal camera for 39 plants (13 watered controls, 17 in W2, 9 in W3),
24 Aug – 22 Sep 2025, all five daily runs (5,402 plant scans), plus the
greenhouse air and light sensors. Day 0 = 3 Sep, when irrigation of W2 and W3
stopped. Outside data: the German weather service station at Belm, 11 km away,
and Idso's published sugar beet baseline (sources below).

## Three numbers

1. **Day 5, with flyovers at 12 h and 15 h.** Each plant scored against the
   control bed (threshold 2.02 z, set on W2 with 4 controls) flags **all 26**
   stressed plants: **24 of 26 by day 5** and **26 of 26 by day 7**. The median
   first alert is day 5 on both W2 (tuning) and W3 (held out; all 9 by day 5).
   There are **0 of 13** false alerts on watered plants during the stress
   window and **0 of 39** plants alerting before the cutoff. Only **3.7 %**
   of single watered scans (27 of 736) are above the threshold. Six plants
   alert as early as day 3 (4 in W2, 2 in W3), so "day 3" is the earliest
   alert, not the typical one. Against the leaf clip, the alert comes
   0 days before the turgor shift on W3 (median) and 2 days before on W2.
   - **Why not 09 h.** 09 h alone flags W3 by day 4, but it fails the
     watered-day check: before the cutoff, **24 %** of W2-set scans are above
     its threshold, and on day −10 **86 %** of the plants are, while all of
     them were watered. Its "0 false alerts before the cutoff" holds only
     because a cloudy day broke the two-in-a-row run. That is a morning-sun or
     bed effect, not stress, so 09 h is not recommended.
   - **06 h never separates the groups, and 18 h is late** (W2 median day 8)
     and fails the watered-day check too (38 % of plants above on day −2).
   - The paper's set-up (one 15 h flyover, leaf minus air temperature) needs
     until **day 8**, 2–3 days *after* the clip.
2. **No control bed needed, at five flyovers a day.** Scoring each plant
   against a healthy-plant line (leaf minus air temperature as a function of
   VPD, fitted on the baseline days 30 Aug – 2 Sep, from air temperature and
   humidity only) has a median first alert of **day 4** on W2 and W3, with no
   false alerts. Its alerts spread out more, though: all plants are flagged
   only by day 8 on W2 and day 7 on W3, against day 7 and day 5 for the
   control-bed rule. It needs all five daily flyovers.
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
(AUC; 0.5 = no difference, 1 = perfect). Separation appears first in the 09 h
and 15 h rows; the 09 h row also shows the bed difference on watered days.

## Advice in the app

The Irrigation advisor opens on the control-bed rule at 12 h + 15 h (chosen
on W2 alone by `uiwue.diurnal.best_rule`) and on the latest day with a usable
scan. A plant that alerts stays "irrigate" until re-watering (day 12,
`uiwue.diurnal.daily_status`). A day without a usable scan is marked
"no scan", not "fine". With that latch, the count of plants to water climbs
6 → 7 → 24 → 25 → 26 over days 3–7 and stays at 26 through day 12.

## Two limits

1. **One control bed.** Before the cutoff, when every plant was watered, the
   beds already differ at 06 h (AUC 0.26) and 09 h (0.62), so the early 09 h
   signal is at least partly a bed difference rather than stress. Correcting
   for the air warming between bed scans did not remove it. 12 h (0.51) and
   15 h (0.56) are close to no difference. Thresholds were tuned on W2 with
   only 4 control plants; W3 and the other 9 controls were never used for
   tuning. The watered-day check (at most 5 % of watered scans and at most a
   quarter of the plants on any watered day above threshold) is our choice
   of cut-off, not a published one, and the recommended 12 h + 15 h rule
   passes it with some margin on rate (4.4 % on the W2 set, 2.8 % on the W3 set)
   and plenty on the worst day (12 % and 14 %). The 3.7 % figure above pools
   all 39 plants, so it includes the tuning plants. The watered days include
   the baseline days the scores are referenced to, which flatters the rate a
   little.
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
python -m uiwue.diurnal                      # rebuilds derived/diurnal/ from derived/ alone; prints the checks
```

Pages: Daily rhythm and Irrigation advisor (`cd app && python app.py`).
