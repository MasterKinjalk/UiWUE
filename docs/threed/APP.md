# UiWUE app: the 3D canopy page

One page, **3D canopy** at `/threed`, holds the Challenge 3 work in the app (deliverable D5 in
`plan/Kinjalk-Work-Plan.md`, section 8). It replaced three earlier pages (3D Browser, 3D Compare,
3D Export) in November 2026; the review behind that is in `docs/threed/PRESENTATION-NOTES.md`.

The answer it gives: **3D does not warn first. It measures how far each plant's canopy collapses
without water and whether it comes back.** Deficit plants lose leaf area and height; after
re-watering, height returns and leaf area does not.

## Run

From the repository root:

```bash
.venv/Scripts/python.exe app/app.py                  # http://127.0.0.1:8050/threed
.venv/Scripts/python.exe app/app.py --port 8060      # another port
.venv/Scripts/python.exe app/app.py --debug          # hot reload and the Dash error panel
```

On macOS or Linux, use `.venv/bin/python`. `UIWUE_DATA=<dir>` reads the local cloud cache and RGB
frames from another data folder (default `data/`).

Deep link: `/threed?bed=W2&plant=O8&day=12` (`date=2025-09-15` works too). `bed` and `plant` pick
the plant, the page's one selection, drawn in navy; `day` is the day drawn as NOW against a
baseline day (THEN). A day whose runs all failed QC opens on the nearest drawable day. The page
writes the selection back into the address bar.

Without a link the page opens on a default chosen by rule (`threed_canopy.default_plant`): of the
W2 plants whose peak-to-trough losses in height and leaf area both sit inside the bed's middle half
and whose NOW run on a trough day is below their baseline on all three traits (so the card agrees
with the tiles), the one with the largest visible height drop from THEN to NOW that has drawable
clouds on a baseline day and a trough day. On the 216-plant table that is W2 O8 (day −3 against
day 12: leaf area −67 %, height −50 %, volume −80 %).

## What the page shows

The layout lives in `app/pages/threed.py` and the figures in `app/threed_figures.py`. Every number
comes from the data layer `app/threed_canopy.py`:

| Part | What it shows | Data |
|---|---|---|
| Bed tiles | the shared stat tile (`components.stat`): height and leaf area lost, peak to trough, with the controls' change; after re-watering, height and leaf area against each plant's peak ("height comes back, leaf area does not"); first seen, the 3D onset against thermal with the `Size only · day 6` chip, a link to Early warning (every onset is on its hover) | `headline()`, `onsets()` |
| The plant, then and now | plant-only point clouds of THEN and NOW with one camera and one scale, soil drawn as a disc with the 85 mm crop ring, a 50 mm scale bar. NOW draws the day's most typical QC-pass run (closest to the plant's per-trait median over that day and the days either side, so one odd run is never the picture); THEN draws the baseline-day run closest to the plant's baseline. Detached edge points are not drawn (below) | `cloud_days()`, `then_run()`, `plant_cloud()`, `axis_ranges()` |
| Plant card | the plant's ID (a link to its profile, carrying the day) and the three traits of the run drawn as NOW, as change from the plant's baseline, so the numbers describe the picture; the day's mean over its QC-pass runs is on hover. No status mark (the leaf-area rule also flags controls) | `cloud_days()` (`<trait>_run_pct`), `plant_days()` |
| Side view | highest kept plant point per 5 mm of x of the same two runs, THEN dotted and NOW filled: where wilting reads best, and a view that needs no WebGL | `side_profile()` |
| Deficit − control | per day, the mean change of W2 minus that of W1 (thick amber, drawn on top), with a 95 % plant-bootstrap band and the spread seen on watered days; the chosen plant minus the W1 mean as a thin navy line under it | `gap()`, `plant_days()` |
| Collapse and recovery | every plant against its own peak (days 1 to 3): trough (days 10 to 12), re-watered (days 15 to 16). Per bed the median plant (line) and the middle half of the plants (band), one dot per plant (click to show it), the chosen plant in navy | `windows()` |
| Onset | no longer a chart on this page: the first-seen tile carries the leaf-area onset, every 3D, thermal, RGB and size onset is on its hover, and Early warning ranks them all; an `older table` chip with the n when `derived/detect` predates the 3D table | `onsets()`, `onsets_info()` |
| `bed effect` chip | on the traits the placebo (its W2 rows) or the E2 check flags (on the 216-plant table: all three; leaf area by E2 alone). When every trait on the page is flagged, as now, the page says it once, in the header (`design/DESIGN.md` 7.4); otherwise it sits on every tile, card row and chart of a flagged trait. It links to `/methods#bed-effect` | `bed_effects()` |
| Header chips | at most two: `bed effect` (or `1 control bed` when not every trait is flagged) and `method` (how the numbers are made, a link to `/methods#own-baseline`); a W3 plant adds `not pooled` (a link to `/methods#w3-sampled`) | `bed_effects()`, `sampled_leaf()` |
| Next steps | at the foot, carrying the plant and day: Early warning (filled), Plant profile | – |

**Detached edge points** (`stray_mask`, display only). Plant points are clustered by single linkage
at 12 mm. A cluster is not drawn when it floats over the plant (its lowest point is above the
highest point kept and it is at least 15 mm from every point kept) and it either enters through the
crop wall (a point within 3 mm of the 85 mm cylinder, at most 10 % of the points) or is a speck
(under 1 % of the points): a neighbour's leaf or debris the crop cut. The trait table keeps every
point; the panel's run label says on hover how many points were not drawn. Of the 9,746 QC-pass
daytime runs of W1 and W2 with a cloud, 346 (3.6 %) have such points, a median of 6 (at most 9 % of a
cloud); checked by image on tall controls (their top leaves stay: a leaf of the plant joins it from
below) and on sparse wilted plants.

When the cloud cache is missing (it is not in git), the cloud panel shows the empty state and the
trait charts still draw. When `derived/threed/traits_runs.parquet` is missing, the page shows the
empty state. The page never draws a synthetic cloud or synthetic numbers.

## Data

| File | Read by | Used for |
|---|---|---|
| `derived/threed/traits_runs.parquet` (committed) | `runs()`, through `uiwue.core.read_table` like every page | every number; W1 (control, 72 plants) against W2 (deficit, 72 plants); W3 (72 plants) shown one by one |
| `derived/detect/onsets.parquet`, `onsets_by_slot.parquet` | `onsets()`, `onsets_info()` | onsets, read as computed by `uiwue.detect`; no second onset rule. Refresh with `python -m uiwue.detect.run` after a new 3D table |
| `derived/truth/rwc_leaves.parquet` | `sampled_leaf()` | the day a W3 plant lost a leaf to the water-content sampling |
| `derived/detect/placebo.parquet`, `derived/threed/results/e2_bed_baseline.csv` | `bed_effects()` | the `bed effect` chip |
| `data/derived/threed/clouds/<bed>_<plant>/<YYYY-MM-DD>_<HH>.npz` (local) | `cloud_days()`, `plant_cloud()` | the point clouds |
| `data/rgb/<bed>_<plant>/<YYYY_MM_DD>/<HH>/rgb_<frame>.jpg` (local) | `rgb_frame()` | the RGB frame a cloud was built from |

W3 is never pooled into the deficit group (most of its plants lost a leaf to the water-content
sampling), so every bed number, band, dot and onset compares W1 with W2 only. W3 plants can still be
picked one by one: they carry a `not pooled` chip, and their navy line is drawn against the W1 and
W2 distributions. Tables and clouds are cached by path and modification time
(`threed_canopy.table_key()`): a rewritten file is picked up on the next callback, with no restart.

### Method

- **Day:** days since irrigation off, day 0 = 3 Sep 2025; W2 was re-watered on day 12 at 18:35.
- **Plant-day value:** the mean of the plant's QC-pass runs at 9, 12, 15 and 18 h. 06 h runs are
  dark and are not used. QC-failed runs never enter a mean.
- **Baseline:** the median of the plant's plant-day values on days −4 to −1 (30 Aug – 2 Sep), the
  app's convention (`design/DESIGN.md` section 7.2). Values are % change from it. A plant without a
  baseline keeps its rows; its change shows `–`.
- **Loss and recovery:** each plant's trough and re-watered windows against its own peak window
  (paired, so they need no baseline).
- **Intervals:** 2,000 resamples of plants, seeded, so a band is the same on every load.

## The three traits

| Trait | Label | Unit | Role |
|---|---|---|---|
| `depth_leaf_area_est` | Leaf area | cm² | leads: separates first and does not come back; carries the `bed effect` chip (E2) |
| `depth_height_p75` | Height | mm | the reversible (turgor) axis; carries the `bed effect` chip (placebo and E2) |
| `depth_hull_volume` | Volume | cm³ | secondary; carries the `bed effect` chip (placebo and E2) |

On the committed 216-plant table (computed live by the page; numbers as of 15 Nov 2026; mean over
plants with a 95 % plant-bootstrap interval):

| Finding | Deficit (W2) | Control (W1) |
|---|---|---|
| Leaf area, peak → trough | −61 % (−65 to −57), n 68 | +1 % (−4 to +6), n 71 |
| Height, peak → trough | −31 % (−38 to −24), n 68 | +20 % (+17 to +24), n 71 |
| Leaf area, peak → re-watered | −49 % (−52 to −45), n 70 | +3 % (−2 to +8), n 67 |
| Height, peak → re-watered | +6 % (+2 to +11), n 70 | +23 % (+19 to +26), n 67 |

Height loss spreads widely (−70 % to +44 %; median −36 %): 32 of 68 deficit plants lose more than
40 %, 13 lose 6 % or less (11 of them grow). 64 of 68 lose at least 33 % of their leaf area. On the
28-plant table the numbers were −39 % height, −64 % leaf area, −3 % and −54 % after re-watering.
Robust onsets, W2 against W1 (`derived/detect`, recomputed on the 216-plant table on 15 Nov; n
62 + 64 for 3D): thermal day 4, leaf area day 5, RGB and size day 6, volume day 7, height day 8
(on the 28-plant table: 4, 5, 6, 6, 9). 3D is about a day behind thermal; its part is how much
each plant lost and what came back.

### What was removed, and why

Noisy or redundant signals are left out of the analysis and the page, not hidden behind a toggle.

| Removed | Why |
|---|---|
| `droop_index` | no signal (its interval spans 0); too noisy from day to day |
| `inclination_mean`, `inclination_sd` | no stress signal; both rise in both beds as the canopy grows |
| `canopy_radius` | saturated at 80 to 81 mm by the 85 mm segmentation cylinder |
| `projected_area` | saturated: the control is flat from day −2. It stays in the app as the size-only reference beside detection results (`design/DESIGN.md` section 7.6) |
| `n_points`, `height_p50`, `height_p95`, `compactness`, `cyl_*` | redundant with a kept trait (ρ 0.94 to 0.98) or later |
| midday-dip and overnight-recovery models | a null: midday minus 09 h does not differ between beds; 06 h runs are dark |
| trait-against-turgor tab | correlations near 0 once detrended; its table existed on one machine only |
| the "reliable" badge (E1) | it compared the signal with within-run noise, not with day-to-day repeatability |

The detection run leaves the removed traits out at the source (`uiwue.detect.data.DEPTH_EXCLUDED`),
and the pages read every detect table through the same definition (`components.kept_trait`). The
fusion models (`docs/fusion.md`) never read the trait table: they embed the raw depth images, or
use hand-made thermal features (v2); only their size-only baseline reads `depth_projected_area`.

### Caveats (on hover and chips on the page)

- One control bed: treatment and bed cannot be separated.
- W2 heights already grew more slowly before day 0 (E2), hence change from each plant's own
  baseline and the `bed effect` chip on height.
- W2 runs fail QC more often on days 6 to 11, and the failed runs are the smaller plants, so
  QC-pass means understate the loss.
- Every plant of the 216-plant table has a baseline; a plant without one would keep its rows and
  show `–`.
- The onsets come from `derived/detect`, recomputed on the 216-plant table. They count 62 deficit
  and 64 control plants (a QC-pass 12 h or 15 h run and a baseline at that slot). When the 3D
  table is rewritten after them, the page marks them `older table` with their n until
  `python -m uiwue.detect.run` is run again.
- Heights jump between day −3 and day −2; area traits show a day-0 gap in the placebo that is gone
  on days 1 to 3.
- "Leaf area" is measured inside the fixed 85 mm plant cylinder, not on the whole plant.
- Onset days move by 1 to 3 days with the time-of-day slot: quote the robust daytime onset and
  its slot range.

## Links in and out

- **Plant profile:** its first next step, `3D canopy` at the page's foot, opens
  `/threed?bed=…&plant=…&day=…` for every plant with depth data. Its three 3D charts read
  `threed_canopy.plant_days` (`profile_views.threed_series`), so a plant shows the same daily values
  on both pages; before, Plant profile took the median of the 12 h and 15 h runs, QC-failed ones
  included (W1 J5's leaf area on day 12 read +70 % there, +108 % here). Its `Leaf area, trough` and
  `After re-watering` tiles give the plant's leaf-area change from its own baseline at the end of the
  no-water window (days 10 to 12) and after re-watering (days 15 to 16), from the same windows. They
  are readouts, not an alert: on the committed table the structure rule (below −20 % of baseline on
  two usable days in a row while unwatered) fires for 63 of 72 deficit plants (median day 9), but also
  for 2 of 72 control plants (W1 L3 from day 1, W1 O8 from day 9); the rule's verdict and the live
  control count are on the tile's hover. W3 plants link to the page too and are shown on their own.
- **Bed map:** the selected plant's next steps (`Plant profile`, `3D canopy`) open it with the day.
  Its squares stay one run each, against the same slot's baseline (run by run is the page's
  question); a run that failed QC is `no scan`, and the hover of a 3D trait says the other pages
  show the daily mean.
- **Daily rhythm:** QC-pass runs only; a 3D trait grouped by treatment is W1 against W2, with a
  `W1 + W2 only` chip (by bed, W3 is its own line). Its per-slot means are raw values, as the hover
  says. Its `Midday dip · 3D` tile reads E4 (`threed_canopy.midday_dip`): whether drought deepens the
  15 h − 09 h dip in height. It says `none` while the interval holds zero (now +0.04 mm/day); the
  hover adds the leaf-area dip once dark runs are left out, when that one is clear.
- **Home:** the finding tiles `3D sees stress` (leaf area, day 5) and `Leaf area lost` (−61 %, with
  the `bed effect` chip) link here, and one `3D canopy` page tile, `forest-700` on `mint`
  (`design/DESIGN.md` section 2.1), sits in the `See` group.
- **Early warning:** opens on W2 against W1, the comparison this page uses: its `3D` tile is leaf
  area at day 5 against thermal at day 4, and links here. With W3 in the group (`W3` or `W2 + W3`),
  63 of whose 72 plants lost a leaf to sampling, leaf area reaches day 4, so the header carries a
  `W3 sampled` chip (the W2-only onset on hover). Its `Loss regained` panel reads the same
  re-watering as the share of each signal's change that came back by 18–19 Sep (leaf area 32 %,
  height 215 %), against the plant's own baseline; this page reads it against each plant's peak
  (leaf area −49 %, height +6 %). Both say height comes back and leaf area does not. Only the kept 3D
  traits (and projected area as `Size only`) are ranked there.
- **Export:** the table `3D canopy · plant-days` replaces the old 3D export: one row per plant and
  day with the three traits, each with its baseline (`_base`) and % change (`_pct`), the run
  counts, and a data dictionary. The merged run table keeps every `depth_` column at run level.
- **Old links:** `/threed-browser`, `/threed-compare` and `/threed-export` land on the 3D canopy
  page with their query, through the page's `path_template` (`/threed<legacy>`); they have no nav
  entry. (Dash's `redirect_from` would not do: it answers with a fixed 301 and drops the query.)
- **Out:** the next steps at the foot, Early warning and the chosen plant's Plant profile, both
  carrying the plant and day; the plant's ID in the card links to its profile; the first-seen tile
  links to Early warning; every caveat chip links to its entry on Methods & limits.

## Code layout

```
app/
  pages/threed.py     the 3D canopy page
  threed_canopy.py    data layer: runs, plant-days, gap, windows, headline, onsets, bed checks,
                      structure status, the run drawn per day and THEN, detached edge points,
                      the default plant, clouds, side elevation, RGB frame (pandas, numpy, scipy)
  threed_figures.py   the cloud stage and the other 3D figures
  assets/threed.css   page styles, on the token variables
  profile_views.py    threed_href, threed_series, structure_summary and threed_days: the links
                      in, the Plant profile charts and card, the Export table
tests/app/test_threed_canopy.py   the data layer, on the committed tables and small sandboxes
tests/app/test_threed_pages.py    the page
tests/app/test_profile_pages.py   the links in, the Plant profile card, the Home tile, the Export table
```

Run the tests with `.venv/Scripts/python.exe -m pytest -q tests/app`.

## Adding a page (teammates)

1. Create one file `app/pages/<name>.py`:

   ```python
   import dash
   from components import header
   from dash import html

   QUESTION = "Which plants show stress first?"
   dash.register_page(__name__, path="/ndvi-maps", name="NDVI maps", order=40, description=QUESTION)


   def layout(**query):  # a function, so the data are re-read on each visit
       return html.Div([header("NDVI maps", QUESTION)])
   ```

   `order` is the position in the navbar (3D canopy uses 25). The page's question goes in
   `description` and in the title's hover, never as visible text (`CLAUDE.md`).
2. Prefix every component id with your page, for example `ndvi-`. All pages share one id namespace.
3. Import helpers that sit next to `app.py` by their module name (`import threed_canopy`).
   `app.py` puts `app/` on `sys.path`.
4. Use the shared parts in `app/components.py` and the tokens; put page CSS in `app/assets/`,
   scoped to your ids. Dash loads every file there on every page.
5. Do not edit `app.py`. The navbar and the Home page list every registered page.
6. Pages register their callbacks globally, so create the app once per process. Tests that need
   the whole app run it in a fresh process (see `tests/app/test_profile_pages.py`).
