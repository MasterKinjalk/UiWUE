# design/

| Path | What |
|---|---|
| [`DESIGN.md`](DESIGN.md) | the UiWUE design system: rules every page follows |
| [`tokens.json`](tokens.json), [`tokens.css`](tokens.css) | the same values for code (`--uw-*` CSS variables) |
| [`mockups/`](mockups/) | one `.dc.html` mockup per app page, plus `Nav`, `System` (the kit) and `canvas.json` |
| [`mockups/assets/`](mockups/assets/) | point-cloud renders and RGB frames of W2 A8 used by the 3D browser mockup |
| [`seminar/`](seminar/) | the base design system the tokens come from (read-only copy) |

The mockups are exports of the Claude Design canvas at
https://claude.ai/artifact/BMdxAEHvVaEb1DBVpN8kZQ. They need that canvas's runtime (`support.js`)
and its uploaded images (`/_blob/...`), so they do not render when opened from disk. Read them as
the reference markup and real numbers for each page, or open the canvas.

To change a mockup, edit it on the canvas, then copy the file back here.
