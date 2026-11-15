# UiWUE: instructions for every Claude Code session and every teammate

Read [`plan/Master-Plan.md`](plan/Master-Plan.md) for who builds what, and follow its contract
(section 6). This file adds the rules for the app's look.

## UI design system: binding on every app page

The source of truth is [`design/DESIGN.md`](design/DESIGN.md), with values in
[`design/tokens.json`](design/tokens.json) and [`design/tokens.css`](design/tokens.css), and one
mockup per page in [`design/mockups/`](design/mockups/). Before you create or change anything
under `app/`, read `design/DESIGN.md` sections 1 to 8 and the mockup of the page you are touching.

### Follow strictly (a page that breaks one of these is not done)

1. **Minimal text. It is a UI, not a text editor.**
   - The page title is a short noun phrase, with no subtitle and no question line. The page's
     question goes in `register_page(description=...)` and in the title's hover.
   - Buttons and legend entries are 1 or 2 words.
   - Methods, definitions and caveats go in hover text or a chip, never in a paragraph.
2. **Colour tokens only** (`design/tokens.json`). No other hex values, no red/green pair, no
   gradients except the legend and point-cloud ramps, no shadows.
3. **One meaning per colour:**
   - control is `sky` / `teal-700`, deficit is `sand` / `amber`
   - value is lightness on one teal-to-amber ramp, with amber = more stressed
   - the baseline is `mint`, the no-water window is `sand`
   - the single selection is `navy`
4. **Status by shape and colour:** fine is a circle, watch a triangle, irrigate a diamond.
   No-scan, no-table and not-imaged are three different marks (DESIGN.md section 3).
5. **Type:**
   - Cambria / Source Serif 4 for titles, IDs and big numbers; Calibri / Source Sans 3 for the
     rest. Nothing below 12 px except the 11 px caps kicker.
   - Spacing 4/8/12/16/24/32, radius 4/8/12, controls at least 44 px.
6. **Charts** use the Plotly template and the band, series and onset conventions in DESIGN.md
   section 6. The x axis counts days since irrigation off (day 0 = 3 Sep 2025).
7. **Data honesty:**
   - real tables only, and the empty state when a table is missing
   - values as change from the plant's own baseline
   - the method on hover
   - a `bed effect` chip where the placebo flags a trait
   - units on every number, `–` for missing values
8. **Accessibility:** 4.5:1 text contrast, real buttons and links, `aria-pressed` on toggles,
   `aria-label` on icon buttons and charts, layouts that work at phone width.

### Follow loosely (the default, adapt when the data needs it)

- The page layouts and component placement in `design/mockups/<Page>.dc.html`, mapped in
  DESIGN.md section 9.
- Exact chart sizes, the order of controls, and which secondary readouts a side panel shows.

### When a request conflicts with these rules

Do what was asked, say in the reply which rule it breaks, and offer the compliant option. Never
silently restyle a page or change a token. Changes to the system go through the team chat and
update `design/DESIGN.md`, `design/tokens.json` and `design/tokens.css` together (DESIGN.md
section 10).

`app/components.py` (shared, owned by Yuming) still carries the older palette. Do not edit it
without his agreement. New page code uses the tokens directly until it is migrated.
