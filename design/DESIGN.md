# UiWUE design system

This file is the source of truth for how every page of the UiWUE app looks and behaves. The
machine-readable values are in [`tokens.json`](tokens.json) and [`tokens.css`](tokens.css). The
mockups in [`mockups/`](mockups/) show the rules applied to every page in master plan section 7.

- **Base:** the team's "Seminar" design system (copy in [`seminar/`](seminar/)), adapted from
  slides to screens. Where this file and Seminar differ, this file wins.
- **Canvas (live mockups):** https://claude.ai/artifact/BMdxAEHvVaEb1DBVpN8kZQ. It is private
  to its owner, who has to share it before teammates can open it.

Two levels of rule:

- **MUST** (sections 1 to 8): follow these exactly on every page. A page that breaks one is not done.
- **SHOULD** (section 9): the page layouts in the mockups. Follow them unless the data or the
  question needs a different layout; when you change one, keep sections 1 to 8.

---

## 1. Text: it is a UI, not a document (MUST)

| Element | Rule | Example |
|---|---|---|
| Page title | a short noun phrase, nothing under it | `3D canopy`, `Early warning` |
| The page's question | in `dash.register_page(description=...)` and as the `title` (hover) of the `<h1>`, never as visible text | "Which signal detects stress first?" |
| Buttons, tabs, toggles | 1 or 2 words | `Compare`, `RGB`, `vs control bed` |
| Legend entries | 1 or 2 words, or a value | `fine`, `no scan`, `+8 °C` |
| Method, definitions, caveats | hover text (`title=`, `dbc.Tooltip`) or a chip (section 5) | chip `bed effect` that explains on hover |
| Numbers | always with a unit; changes signed with `+` / `−` (U+2212) | `−63%`, `+4.7 °C` |
| Missing value | en dash `–`, with the reason on hover; never `0`, never blank | `–` "no depth table for this plant yet" |
| Days | days since irrigation off first (day 0 = 3 Sep 2025), calendar date second | `Day 4 · 7 Sep` |

No paragraphs, no subtitles, no instructions ("click a plant to…") and no sentences in cards. When
a page needs explanation, put it in the chat reply, the PR description or `docs/`, not on screen.

**One name per signal.** A signal reads by the same plain name on every page, from one dictionary,
`components.PLAIN` (`Leaf − air temperature`, `Leaf area`, `Height`, `Green–red index`, `Projected leaf
area`, ...); the catalogue's technical name (`Canopy height, p75`, `NGRDI`) goes on hover
(`components.tech`). A signal missing from `PLAIN` reads by its catalogue label. Labels are in sentence
case (`Leaf area lost`, `Water now`), never all lower case.

**Finding first.** Every page opens with its answer in numbers a person can repeat: a row of stat tiles
(section 5) above the page's one focal figure. Methods & limits is the one exception: its first entries
are the limits.

## 2. Colour (MUST)

### 2.1 Tokens

| Token | Hex | Use |
|---|---|---|
| `surface` | `#ffffff` | page and chart ground |
| `surface-2` | `#f6f8f9` | inactive controls, zebra rows, out-of-window days |
| `mint` | `#eef4ee` | baseline window (30 Aug – 2 Sep), neutral cards |
| `sky` | `#dceef2` | watered control: bed header, plant panel, stat tile |
| `sand` | `#fbf1de` | water deficit, the no-water window (day 0 – 12), caveat chips |
| `gray` | `#eef1f2` | ground truth, neutral and "no change" |
| `lilac` | `#f0e8f5` | rare: the third group of page tiles on Home |
| `ink-display` | `#17362a` | titles and big numbers |
| `ink` | `#1e2a24` | body text |
| `ink-muted` | `#5e6b64` | units, axis labels, secondary labels (keep it off `amber-200`) |
| `slate-600` | `#4a565d` | labels inside gray cards |
| `navy` | `#0b2e3f` | the one selected plant or day, table headers, mode toggles that are on |
| `rule` | `#cfe0d4` | hairlines, grid lines, dashed "missing" outlines |
| `gray-500` | `#8a959c` | control borders, non-highlighted series, QC-failed marks (no text) |
| `teal-700` | `#1c7293` | primary action, selected option, control series, highlighted series |
| `teal-600` | `#3f7f93` | strong "healthier than baseline" |
| `teal-300` | `#9fd0db` | mild "healthier than baseline" |
| `teal-200` | `#bfe0e8` | fill of the "fine" mark, weak evidence cells |
| `amber-200` | `#fbe3b3` | mild "more stressed than baseline" |
| `amber-700` | `#c98300` | deficit series, "watch", strong stress (fills and lines only, no small text) |
| `amber-900` | `#5a3e00` | "irrigate", strongest stress, every amber text |
| `forest-700` | `#2e6b4f` | 3D page tiles and 3D bars |
| `plum-700` | `#6b3f7a` | single-plant page tiles on Home |
| `leaf-ramp` | `#cfe3c9 → #8db57a → #2e6b4f → #17362a` | point-cloud height (0 → 130 mm), UiWUE addition |
| `soil` | `#d5dcd8` | non-plant points in a cloud, UiWUE addition |

### 2.2 Colour roles

Each colour has one meaning, and the meaning is the same on every page.

- **Treatment is identity.** Control means `sky` tint with `teal-700`; deficit means `sand` tint with
  `amber-700` lines and `amber-900` text. The same mapping applies in every chart, map and table.
- **Value is lightness, on one diverging ramp:**
  `teal-600 · teal-300 · gray · amber-200 · amber-700 · amber-900`.
  The amber end always means "more stressed", taken from the catalogue's `direction`. The grey
  band spans the trait's noise floor (E1).
- **Evidence (AUC lower bound) is a sequential teal ramp:** `gray · teal-200 · teal-600 · navy`,
  breaking at < 0.5, 0.5, 0.7 and ≥ 0.9. Raw AUC heatmaps (Daily rhythm) use the diverging
  ramp, with 0.5 shown as `gray`.
- **Time bands:** the baseline is `mint` and the no-water window is `sand`. Cutoff and re-watering
  are `ink-muted` hairlines.
- **Selection is `navy`:** one plant, one day, one view at a time.
- **Forbidden:** red and green as a pair, hue alone to tell two things apart, gradients other than
  the functional ramps above (legends, cloud height), shadows, transparency as decoration.

`app/components.py` reads these tokens from `tokens.json` (`components.C["teal-700"]`), and
`app/app.py` injects `tokens.css`, so page CSS uses `var(--uw-teal-700)`. The shared page styles
are in `app/assets/uiwue.css`.

## 3. Marks: status by shape (MUST)

Every status is shape plus colour, so it still reads in greyscale and for colour-blind readers.

| Mark | Meaning | Build |
|---|---|---|
| circle | fine: at or below threshold | 20 px, fill `teal-200`, 2 px `teal-700` ring |
| triangle | watch: above threshold once | 24 px, `amber-700` |
| diamond | irrigate: above threshold on two scans in a row | 19 px square rotated 45°, `amber-900` |
| square | a value (change from baseline), binned on the value ramp | 28 px, radius 4, 1 px `gray-500` edge |
| dashed circle | no usable scan: the plant is in the table, but this day has no run passing QC | 18 px, 2 px dashed `gray-500` |
| dashed square | no table: the module has not pushed this plant | 18 px, 2 px dashed `gray-500` |
| dot | position not imaged | 4 px, `rule` |
| navy outline | selected | 2 px `navy` outline, `sky` cell |

Plotly symbols: `circle`, `triangle-up`, `diamond`, `square`, `circle-open-dot`, `square-open`.

## 4. Type, space, shape (MUST)

- **Faces:**
  - `display` = `Cambria, "Source Serif 4", Georgia, serif`: page titles, bed and plant IDs,
    status words, big numbers.
  - `sans` = `Calibri, "Source Sans 3", "Segoe UI", system-ui, sans-serif`: everything else.
  - Load the hosted fallbacks from Google Fonts. Body copy is never set in the serif.
- **Scale (px, size / line):**
  - page title 30/36 bold display
  - stat 28–37 bold display
  - readout 24/28 bold display
  - bed or section 21/26 bold display
  - body 15/21
  - secondary 13/18 `ink-muted`
  - axis 12/16 (the floor for reading text)
  - kicker 11/14, bold, caps, 0.18em letter-spacing
- **Numbers in tables:** `font-variant-numeric: tabular-nums`, aligned right.
- **Spacing:** 4 / 8 / 12 / 16 / 24 / 32. Page gutter 32 px (16 px on a phone), content max-width 1440 px.
- **Radius:** 4 for cells and chips, 8 for controls and cards, 12 for panels.
- **Borders:** 1 px `rule`. The only dashed outline is 2 px dashed `rule`, and it means "missing".
- **Controls:** at least 44 px tall. Map cells are the one exception, at 38 px.

## 5. Components (MUST, when the need arises)

Reuse these. Do not invent a new pattern for a need one of them already covers.

| Component | Spec | Mockup |
|---|---|---|
| Nav | `UiWUE` wordmark (display 21), then the pages as a path in four groups, `See` (Bed map, Plant profile, 3D canopy), `Detect` (Early warning, Sensor comparison, Daily rhythm), `Act` (Irrigation advisor) and `Data` (Export, Methods & limits): a caps kicker (11, its question on hover) over a row of links (sans 14, what the page answers on hover). The current page is `ink-display` bold with a 3 px `teal-700` underline and `aria-current="page"`. Below 1040 px the groups fold into one 44 px menu button that names the current page; the open menu lists the same groups, the current page marked by a 3 px `teal-700` left edge on `surface-2`; it closes after a pick and on Escape. A nav link carries `?bed=&plant=&day=` to a page that shows one plant. (`components.NAV`, `app/app.py`, `assets/nav.js`) | `Nav.dc.html` (drawn before the groups) |
| Page header | a caps kicker (11, `teal-700`: the page's nav group, `See` / `Detect` / `Act` / `Data`) over the title, then context chips, then a spacer, then controls aligned right (`components.header`) | every page |
| Segmented group | buttons with `aria-pressed`. The selected one is `teal-700` with white text; a mode switch that is on (Compare, Group by) is `navy` with white text; the rest are `surface-2` with a `gray-500` border. | Bed map, 3D browser |
| Day strip | 46 × 48 cells, with the day number in display 17 and the date in 11 under it. Baseline days `mint`, no-water days `sand`, the selected day `navy`. On Irrigation advisor each cell also carries a small `amber-900` bar for the plants to water that day (76 px tall). | Bed map, Irrigation |
| Stat tile | the one tile of every page (`components.stat`): label 12 in sentence case with at most one or two chips, value display 32 (28 on a phone) with its unit 12 beside it (the unit drops under the value as a whole when the tile is narrow), an optional second line 13; two values side by side with `pairs`. Tints: `sand` deficit, `sky` control, `mint` neutral, `gray` sensor results; the best of a set gets a 2 px `teal-700` edge. **Link variant** (`href`): the whole tile is a link to the page that shows the number, marked by a `›` in its top-right corner and a `teal-700` edge on hover; its chips are plain text there. **Missing:** `stat_missing`, a 2 px dashed `rule` edge and `–`, the reason on hover. In a `uw-tiles` grid; on a phone two to a row, an odd last tile takes the row. | Early warning, Export |
| Readout row | label · value (display 24) · unit; the method goes on hover | Bed map panel |
| Bed grid | rows A–R down, columns 1–8 across, staggered, three beds side by side. Never sort or pack it. | Bed map |
| Context chip | caps 12, tinted: `WATER DEFICIT` on `sand` in `amber-900` | 3D browser |
| Caveat chip | `sand` with `amber-900` bold text; the detail goes on hover in plain words (`bed effect`, `W1 + W2 only`). A caveat with an entry on Methods & limits is a link to it (`components.chip(..., anchor=)`, `/methods#bed-effect`), underlined on hover. | Bed map, Irrigation |
| Rule chip | outlined in `gray-500` (`method`, `Size only · day 6`); the definition goes on hover, and it links to its Methods entry when it has one | Early warning |
| Source tag | 11 px bold caps: `3D`, `IR`, `RGB`, `HSI`, `CLIP` | Trait explorer, Sensors |
| Empty state | 2 px dashed `rule` box with the module name and "no table yet" | Profile, Home |
| Evidence grid | signals × days 0–12, cells on the evidence ramp, onset as a white dot with an `amber-700` ring | Early warning |
| Actions | one filled `teal-700` link per panel (`uw-btn uw-btn-primary`); the rest outlined in `teal-700` (`uw-btn uw-btn-outline`), never an inline style. Links carry bed, plant, date and hour. | every page |
| Next steps | the page's one or two next pages, at its foot (`components.next_steps`): the first filled, the second outlined, along the nav's path See → Detect → Act → Data. They carry `?bed=&plant=&day=` to a page that shows one plant (`components.page_href`), and follow the selection when it changes. Export and Methods end the path and point back to Home. | every page but Home, whose page tiles are its next steps |
| Plant link | a plant's ID (`W2 R7`) in display type is a link to its profile carrying the day (`components.plant_link`): no underline until hover, then a 1 px underline in `navy`. Every plant named on a page is one. | every page |
| Disclosure | a rarely used option or a long list behind one native `details` / `summary` (a 44 px control, `▸` closed, `▾` open), closed by default, a 1 px `rule` edge (`components.details`): `All signals`, `Options`, `Data coverage`. | Early warning, Sensor comparison, Home |
| Hover tip | every `title` shows in one box (`assets/nav.js`): `surface`, 1 px `gray-500` edge, radius 8, sans 13/18, at most 320 px wide; on hover after 250 ms, on keyboard focus, and on a tap of a chip. One or two plain sentences. | every page |

## 6. Charts: Plotly and Dash (MUST)

- **Ground:** `paper_bgcolor` and `plot_bgcolor` are `#ffffff`.
- **Font:** sans 13 in `#1e2a24`.
- **Axes:** grid `#cfe0d4`, axis line `#8a959c`, no zero line unless the zero means something.
- **Titles:** no chart title inside the figure; the title sits in the card header. Axis titles are
  the unit only.
- **Default x axis:** days since irrigation off, with the date in the hover.
- **Bands:** the no-water window is a `vrect` in `#fbf1de` (layer below, no line); the baseline is
  a `vrect` in `#eef4ee`; cutoff and re-watering are `#5e6b64` dotted `vline`s, unlabelled
  (the label goes in the hover).
- **Treatment series:**
  - control is a line in `#1c7293` (width 2.5) with a ±1 SD band in `rgba(28,114,147,0.14)`
  - deficit is a line in `#c98300` with a band in `rgba(201,131,0,0.18)`
- **Other series:** `#8a959c` at width 1.25, and only the selected one in `#1c7293`. Never more than
  one highlighted series.
- **Marks:** QC-failed runs are `#8a959c` `x` markers; onset is a `#0b2e3f` dashed `vline` with a
  small `onset` label.
- **Legend:** horizontal, top right, 1–2 words per entry.
- **Hover:** plant · day · date · value with unit · source.

```python
UIWUE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family='Calibri, "Source Sans 3", "Segoe UI", system-ui, sans-serif', size=13, color="#1e2a24"),
    colorway=["#1c7293", "#c98300", "#2e6b4f", "#6b3f7a", "#8a959c"],
    xaxis=dict(gridcolor="#cfe0d4", linecolor="#8a959c", zeroline=False, ticks=""),
    yaxis=dict(gridcolor="#cfe0d4", linecolor="#8a959c", zeroline=False, ticks=""),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", title_text=""),
    hoverlabel=dict(bgcolor="#ffffff", font_color="#1e2a24"),
    margin=dict(l=48, r=16, t=24, b=40),
)
```

Point clouds use `aspectmode="data"`, the `leaf-ramp` for height with one scale per plant, `soil`
for non-plant points and no axes.

## 7. Data honesty (MUST)

1. Show only what a real table says. A missing module, plant or day shows the empty state or a
   dashed mark, never a guess. Synthetic data always carries the synthetic banner.
2. Show values as change from the plant's own baseline (median over 30 Aug – 2 Sep at the same
   slot), unless the raw value is the point.
3. Keep the method next to the number, on hover: flyover hour, threshold, baseline window, rule
   and source table.
4. Show limits where the result is shown. A trait flagged by the placebo gets the `bed effect` chip
   on every chart that uses it. When every trait on a page is flagged, the page says it once instead:
   one `bed effect` chip in the header (3D canopy, whose three traits are all flagged) or on the chart
   that ranks them (`all flagged` on Sensor comparison's fusion view); a chip on every tile and chart
   would say nothing a reader does not already know. The chip links to `/methods#bed-effect`.
5. Keep the three kinds of "nothing" distinct: no usable scan, no table, position not imaged.
6. Next to every detection result, show the size-only baseline (`depth_projected_area`), always
   named `Size only`, W2 against W1. Where a page ranks sensors in tiles (Home, Early warning, Sensor
   comparison) it is the last sensor tile and the row `Size only` in the chart; where a page shows one
   detection result (Plant profile's peak stress, 3D canopy's first seen, Daily rhythm's best flyover,
   Irrigation advisor's first alert) it is the rule chip `Size only · day 6` beside that number
   (`components.size_chip`, a link to `/methods#size-baseline`).

## 8. Accessibility (MUST)

- Text contrast is at least 4.5:1 (3:1 at 24 px and above). `amber-700` and `gray-500` are never
  used for small text.
- Use real `button`, `a href`, `input` and `label` elements. Icon-only buttons get `aria-label`;
  toggles get `aria-pressed`.
- Charts get `role="img"` and an `aria-label` that names the series and the date range.
- The focus ring is 2 px `navy` with a 2 px offset.
- Pages work at phone width: side panels stack, toolbars wrap, and wide tables and grids scroll
  inside their own box.

## 9. Page layouts (SHOULD)

One mockup per page in master plan section 7. Use it as the starting layout and adapt it when the
data needs something else. The table below is the app as it is now (15 Nov 2026), in the nav's order;
every page opens with stat tiles and ends with its next steps (section 5). **Where the mockups no longer
match:** every mockup predates the grouped nav, the shared stat tile, the next-steps row and the plain
signal names; `Home`, `Irrigation`, `Rhythm`, `Profile`, `Main`, `EarlyWarning`, `Sensors` and `Browser3D`
also differ in layout as noted in the table. The mockups are kept as drawn; this table wins.

| Page | Group | Mockup | Layout now | Differs from the mockup |
|---|---|---|---|---|
| Home | – | `Home.dc.html` | hero: title, the trial as a timeline to scale (watered, baseline, no water, re-watered), bed cards (W1, W2, W3 with plant counts), four numbers (plants, flyovers, dry days, leaf clips), and two photos of one deficit plant (day −1, day 10); finding tiles (first day each sensor sees stress, `Size only`, leaf area lost), each a link to its page; `First sign`, one track per sensor over day 0–12 with its onset mark; page tiles in the nav's four groups, numbered 1–4; `Your data`, one tile to New run; data coverage behind a disclosure | hero, photos and the first-sign figure added; coverage timeline behind a disclosure; finding tiles replace the bed tiles |
| Bed map | See | `Main.dc.html` | signal and run pickers; tiles (median per bed, the three plants to look at, each a plant link); day chip, play and slider; the three bed grids; selected plant on the right (ID, value, trend); next steps Plant profile, 3D canopy | the plant's actions moved to the next steps |
| Plant profile | See | `Profile.dc.html` | bed and plant pickers with a stepper; tiles (peak stress with `Size only`, first sign, leaf area at the trough, after re-watering); one focal timeline with a trait picker; one small chart per sensor; next steps 3D canopy, Early warning | no right-hand status column: the tiles say it |
| 3D canopy (was 3D browser) | See | `Browser3D.dc.html` (drawn for the 3D browser; see `docs/threed/APP.md`) | tiles (height lost, leaf area lost, after re-watering against each plant's peak, first seen with `Size only`); then and now clouds, plant card and side view on the right; day strip; collapse and recovery, deficit − control by day; next steps Early warning, Plant profile | onset strip removed (first seen tile; Early warning has every onset) |
| Early warning | Detect | `EarlyWarning.dc.html` | deficit-bed switch; tiles (first day per sensor, `Size only`, leaf clip); onset by signal; daily score of one signal; loss regained after re-watering on the right; all signals behind a disclosure; next steps Sensor comparison, Daily rhythm | onset dot chart instead of the evidence grid |
| Sensor comparison | Detect | `Sensors.dc.html` | sensors / fusion switch; tiles (score per setup with its onset, best buy, `Size only`); detection score per setup, with thermal alone at each resolution (score bar against chance, onset) on the right; options and all setups behind disclosures; next steps Daily rhythm, Irrigation advisor | dot chart with intervals instead of the setup table |
| Daily rhythm | Detect | `Rhythm.dc.html` | tiles (midday heat, 3D midday dip, best flyover with `Size only`); day profile; when to image on the right; separation by hour × day below, the baseline and no-water bands named over it and the recommended flyovers underlined; next steps Irrigation advisor, Early warning | day profile leads; the heatmap moved below |
| Irrigation advisor | Act | `Irrigation.dc.html` | rule picker (its definition on hover), day stepper, caveat chips; tiles (water now, first alert with `Size only`, false alarms, held-out W3); day strip with a bar per day for the plants to water; every plant against the threshold, the plants to water first, CSV; the selected plant's scans on the right, in view while the list scrolls; next steps Export, Methods & limits | the W3 check and track record are tiles, not a side column |
| Export | Data | `Export.dc.html` | download buttons in the header; filters on the left; tiles (rows, plants, days, columns); rows or column meanings; next steps Methods & limits, Home | – |
| Methods & limits | Data | `Methods.dc.html` | jump links; the four limits that bound every claim as cards, then more limits, watered-day check, numbered checks, words and scope full width; every entry has an anchor that chips on other pages link to; next steps Home, Early warning | grouped entries instead of four cards |
| New run | – (Home tile, not in the nav) | none | folder field and `Check`; tiles (plants and dates per camera folder, the two sensor files, dashed when missing); `Steps`: the analysis steps and the three models as segmented toggles (a step without input is dashed and off, the models start off and turn `navy`), options, `Start` (off, with a `start off` chip, unless the app runs with `--runs`), the command; `Runs`: each workspace with its state chip, step chips, `Stop` or `View` / `Open`, log; next steps Home, Methods & limits | no mockup |
| Trait explorer | – | `Traits.dc.html` | not built: Plant profile and 3D canopy cover it | – |
| (kit) | – | `System.dc.html` | colours, marks, type, parts | – |

The numbers in the mockups are real values from the tables, as of 2026-11-14. The onset days were
computed for the mockups with the master plan's 6.6 rule. They are not output from `uiwue.stats`.

## 10. Changing the system

Propose a change in the team chat first. Then update `DESIGN.md`, `tokens.json` and `tokens.css`
in the same commit, and the mockup when the change is visible. Never restyle a page silently.
