# 3D workflow: notes for the presentation

Notes from the four-perspective review of the 3D pages (biologist, engineer, data scientist, UI
designer; 14 Nov 2026) and the rebuild that followed (15 Nov 2026). Use this as the source for the
3D slides. The app page is `/threed` ("3D canopy").

## The one-line story

**3D does not warn first. It measures how bad the damage is and whether the plant recovers.**
Deficit canopies collapse; after re-watering, height comes back and leaf area does not.

## Headline numbers (216-plant table, 15 Nov)

Method: QC-pass scans, plant-day mean over the 09 to 18 h slots, each plant compared with its own
peak window (days 1 to 3), mean over plants with a plant-bootstrap 95 % CI. Control bed W1 against
deficit bed W2, 72 plants each. Day 0 = 3 Sep 2025; re-watered 15 Sep, 18:35. The page computes
these live from `derived/threed/traits_runs.parquet` (`app/threed_canopy.py`, `headline()`).

| Finding | Deficit (W2) | Control (W1) |
|---|---|---|
| Height, peak → trough (days 10 to 12) | −30.7 % (−37.7 to −23.8), n = 68 | +20.2 % (+16.6 to +23.9), n = 71 |
| Leaf area, peak → trough | −61.0 % (−64.8 to −56.6), n = 68 | +0.8 % (−3.6 to +5.9), n = 71 |
| Volume, peak → trough | −66.7 % | +9.3 % |
| Height, peak → re-watered (days 15 to 16) | +6.5 % (+1.9 to +10.9), n = 70 | +22.7 % (+19.3 to +26.1), n = 67 |
| Leaf area, peak → re-watered | −48.6 % (−52.3 to −44.9), n = 70 | +3.0 % (−1.8 to +8.0), n = 67 |

- **Two axes, two meanings.** Height is the reversible axis (turgor: it returns with water, to
  above its peak). Leaf area is the lasting-damage axis (still about half gone days later).
- **The same split shows on Early warning's recovery panel**, from independent tables (W2):
  function recovers (leaf minus air temperature 89 %, NGRDI 69 %), leaf area does not (32 %), and
  height overshoots (215 %: the plants stand up and keep growing).
- **Not every plant sags.** 13 of 68 deficit plants lose 6 % of height or less (11 of them grow);
  32 lose more than 40 %; the range is −70 % to +44 %, median −35.7 %. But 64 of 68 lose at least
  a third of their leaf area. Leaf area is the consistent signal; height is the dramatic one.
- **A per-plant rule that now has a fair test.** Leaf area below −20 % of the plant's own baseline
  on two days fires for 63 of 72 deficit plants (87.5 %, median first day 9) and 2 of 72 controls.
- **Timing (detection re-run on the 216-plant table, 15 Nov).** Robust onsets, W2 against W1:
  thermal day 4 (single slots 4–7; 17 + 13 plants), 3D leaf area day 5 (5–6; 62 + 64), RGB day 6
  (6–10; 9 + 9), size alone day 6, 3D volume day 7 and height day 8 (both with a bed effect). 3D
  is still about a day behind thermal. Was (28 plants): thermal 4, leaf area 5, RGB 6, volume 6,
  height 9. Pooled with W3, leaf area moves to day 4, level with thermal, but 63 of the 72 W3
  plants also lost a leaf to sampling: Early warning marks those 3D results `W3 sampled`.

### How the numbers moved from 28 to 216 plants

| | 28 plants | 216 plants |
|---|---|---|
| Height, peak → trough, deficit | −38.6 % | −30.7 % |
| Leaf area, peak → trough, deficit | −63.8 % | −61.0 % |
| Height, peak → re-watered | −3.1 % | +6.5 % |
| Leaf area, peak → re-watered | −54.2 % | −48.6 % |

The story held with eight times the plants and much tighter intervals. What changed: the height
loss is smaller and no longer clearly bimodal, and control leaf area is flat (+0.8 %) instead of
growing (+11 %).

## What we removed and why

Decision: noisy signals are left out of the analysis and the app, not hidden behind a toggle.

| Removed | Why |
|---|---|
| `droop_index` | No signal (CI spans 0); SNR 1.6 against day-to-day noise; its description was backwards |
| `inclination_mean`, `inclination_sd` | No stress signal; rises in both beds as the canopy grows |
| `canopy_radius` | Saturated at 80 to 81 mm by the 8.5 cm segmentation cylinder |
| `projected_area` | Saturated and redundant with leaf area; kept only as the "size only" reference |
| `n_points`, `height_p50`, `height_p95`, `compactness`, `cyl_*` | Redundant (ρ 0.94 to 0.98 with a kept trait) or late |
| Midday-dip and overnight-recovery models | A null: midday minus 09 h did not differ between beds; 06 h scans are all dark |
| Trait-vs-turgor tab | Detrended and within-day correlations were about 0 |

Kept: **leaf area** (leads), **height p75**, **hull volume**. The same filter now applies on Early
warning (24 → 12 signals), the Methods placebo chart, and the depth pickers on Plant profile, Bed
map and Daily rhythm.

Review findings on the 28-plant table that motivated the cut (not recomputed on 216): only about
three independent axes in the twelve traits (three PCA components explain 83 % of the variance);
within a day, 3D leaf area is nearly uncorrelated with the thermal signal across deficit plants
(ρ = −0.15), so it adds per-plant severity; it largely restates RGB green cover (ρ = 0.70).

A lesson worth a sentence: the old "reliable" badge compared the signal to within-scan noise
(about 2 mm). Against day-to-day repeatability (about 5 mm) droop and inclination fail.

## Caveats to say out loud

- One control bed, so treatment and bed cannot be separated. On the new results all three kept
  traits carry the `bed effect` chip.
- Missing not at random: deficit scans fail QC more often in days 6 to 11 and the failed ones are
  the smaller plants, so QC-pass means understate the loss.
- "Area" is measured inside a fixed 8.5 cm cylinder, not on the whole plant; tall controls are
  clipped by it late in the run.
- Bed W3 lost leaves to sampling. It is shown on the page with a `not pooled` chip and never enters
  the bed-level numbers.
- Detached points at the crop wall (a neighbour's leaf) are hidden in the picture in 3.6 % of
  scans; the trait table is untouched and the hover says how many points are not drawn.
- The onsets were recomputed on the 216-plant table (`python -m uiwue.detect.run`, finished 15 Nov
  07:56 UTC), so the page's `older table` chip is gone. They rest on 62 deficit and 64 control
  plants: a plant counts on a day when it has a QC-pass 12 h or 15 h run and a baseline at that
  slot, fewer than the 72 + 72 behind the tiles.
- The −20 % rule's threshold was first proposed on the 28-plant table, then tested on 216.

## What was wrong with the old 3D pages

- Three pages (browser, compare, export) and no headline on any of them: a run browser, not an
  answer.
- They did not use the design system at all (stock Bootstrap, default Plotly colours, calendar
  dates). The design system was mostly not the cause.
- About 93 % of the points drawn were soil and neighbouring plants; raw under-exposed colour on
  white; one frame with no reference. The default view was a scan that failed QC.
- The pages read a different table from the rest of the app, with W3 pooled into "deficit".
- Off this machine the page drew a synthetic cloud beside real numbers.
- Speed was never the problem: a cloud renders in about 50 ms.

## What we built

One page, `3D canopy`, replacing the three old ones (old links still land on it):

- four live headline tiles, the app's one stat tile: height lost, leaf area lost, after re-watering
  (against each plant's peak: height comes back, leaf area does not), first seen against thermal with
  the `Size only · day 6` chip beside it
- **then and now**: the same plant on a baseline day and a chosen day, plant points only, coloured
  by height, on a soil disc with a scale bar; one shared camera, so rotating one turns the other
- a plant card with the drawn scan's own change from baseline (the plant's ID links to its profile),
  and a side-view silhouette (then as an outline, now filled), where wilting reads best
- a day strip: baseline, no-water window, re-watering, the selected day in navy
- collapse and recovery: each bed's median and middle half, every plant a clickable dot
- deficit minus control with a confidence band (the onset strip that sat under it is gone: the
  first-seen tile holds the onset and every other one is on its hover; Early warning ranks them)
- one `bed effect` chip in the header, since all three traits are flagged, instead of one per chart;
  every caveat chip links to its entry on Methods & limits
- next steps at the foot, Early warning and the plant's profile, both carrying the plant and day
- the opening plant is chosen by a rule (a typical deficit responder whose picture and numbers
  agree), currently W2 O8: leaf area −67 %, height −50 %, volume −80 % on day 12

Rendering choice: points won over a triangulated surface (it tore layered canopies into shards)
and a height-map surface (it invented walls between leaves). Light ground only.

How 3D joins the rest of the app: the page sits under "See" in the grouped nav (See, Detect, Act,
Data; one menu button on a phone); on Home, the "3D sees stress" and "Leaf area lost" finding tiles
link to it and a "3D canopy" page tile sits in the See group; Plant profile's leaf-area tiles and its
first next step; Bed map's next step for the selected plant; a "3D canopy · plant-days" table on
Export; thermal first, 3D next on Early warning, whose "Loss regained" panel reads the same
re-watering against each plant's own baseline (leaf area 32 %, height 215 % of the loss regained).

Verification (15 Nov, after the detection re-run): full test suite 290 passed, 5 skipped (4 of them
for torchgeo and rasterio, not installed here); no page scrolls sideways at an exact 390 px viewport (headless Edge with
DevTools device metrics: a plain `--window-size=390` shot is drawn at Edge's 518 px minimum and
cut off). Before the rebuild, 18 of 18 scripted browser interactions passed.

Screenshots for slides are in `docs/threed/screenshots/`: `before-browser.png`,
`before-compare.png` (the old pages), `after-desktop.png`, `after-phone.png` (the new page, default
plant), `after-control-plant.png` (W1 J1), `after-non-responder.png` (W2 Q4) and
`render-alternatives.png` (points against the two surface renderings).

## Open items

Done on 15 Nov: the detection tables were re-run on the 216-plant table (onsets above; Sensor
comparison, same 9 + 9 plants: thermal alone 0.96, best combination depth + thermal 1.00, size
alone 0.35, cheapest close still thermal alone; before: 1.00, RGB + depth + thermal 0.96 with a
bed effect, 0.37). Plant profile's 3D charts now read the page's plant-day values (W1 J5 leaf
area on day 12 read +70 % there, +108 % here; both now +108 %). `docs/HOW_IT_WORKS.pdf` is
rebuilt from its Markdown with headless Edge (`scripts/build_how_it_works_pdf.py`). The bed
checks on this page read the W2 rows of the placebo, and leaf area, flagged by the E2 growth
check, now carries its chip on its charts too.

- `uiwue.detect` pools W3 into its "all" deficit group (W2 + W3) for 3D too. Early warning and
  Sensor comparison now open on W2 against W1 and mark the pooled groups (`W3 sampled`, with the
  W2-only onset on hover); a W2-only 3D group in the detect run would still be cleaner.
- Proposals not built: cloud colour = height lost on the stress ramp; a separate colour identity
  for sensors (thermal currently shares teal with control on Early warning).
