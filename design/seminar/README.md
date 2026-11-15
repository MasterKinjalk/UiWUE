Slides built to start a discussion. Every slide carries one claim in its title, the evidence for it in the body, and a bar at the bottom that says why the room should care. Use this system for lab meetings, subgroup meetings, journal clubs and committee talks.

## Content fundamentals

- **Title = the claim.** Set every slide title in `slide-title` and write it as the sentence you want remembered: "AgroNT: scoring crop variants before the wet lab", not "AgroNT overview". A title that wraps past two lines drops to `slide-title-long`; a title that still wraps is two claims, so split the slide.
- **Kicker = the context.** Put the modality, section or paper above the title in `kicker` (all caps, letter-spaced): NUCLEOTIDES, PROTEINS, THE PROBLEM, CLASSICAL ROUTE. Colour it with the slide's accent (`teal-700`, `forest-700`, `plum-700`, `amber-900`).
- **One sentence of orientation.** Under the title, one `subtitle` line in `ink-muted` saying what the slide shows, in plain words: "From trusting the manuscript to verifying the evidence behind it."
- **Problem before method.** When presenting a paper or model, lead with the question and the three facts that make it hard (ProblemCard), then the method as three or four named steps (StepStack), then the result. The order is the argument.
- **Bold the phrase you will say.** Each card gets at most one `body-strong` phrase: "far more candidate variants than can ever be tested". Never bold a whole bullet.
- **End with why it matters.** Every evidence slide closes with a TakeawayBar: "Why matters: breeders can shortlist the few regulatory variants worth editing or validating." Write it for the person who will push back.
- **Cite where the number sits.** A figure, number or claim from a paper carries its source inline in `caption`: "(Wilkinson et al. 2016, Scientific Data)", "Fig. 1 · Yu 2026". Do not collect citations on a final slide.
- **Numbers over adjectives.** "+2.45% Dice and +6.91% IoU over the next-best model" beats "substantially better". Keep the unit, the baseline and the dataset.
- **Plain register, no hype.** Sentence case everywhere except kickers. No exclamation marks, no emoji, no "revolutionary". Abbreviations are expanded once per slide (FM = foundation model).
- **Three bullets, three lines.** A bullet list holds at most three items in `body`, each at most three lines. More becomes a StepStack or a MatrixTable.
- **Leave the room something to argue with.** Every talk has at least one DiscussionQuestion slide, and ends on OpenQuestions, not "Thank you". Phrase questions so that two reasonable people could answer differently: "Should reviewers be required to run the code, or only to see it?"

## Visual foundations

### Colour

- Ground is `surface` (white). Content sits in tinted cards: `surface-mint` by default, `surface-sky` for the problem, `surface-sand` for caution and "optional", `surface-lilac` for the plum modality, `surface-gray` for "not specified" and neutral steps.
- Five accents, each with a meaning that holds across the whole deck:
  - `teal-700` — the primary accent: step numbers, icon discs, takeaway bars, the active step, nucleotide/data topics.
  - `forest-700` — the hypothesis-driven and "AI path" track; results; protein topics.
  - `amber-700` / `amber-900` — the data-driven track; caution; "moving forward"; anything provisional.
  - `plum-700` — single-cell and transcriptomics; also the "future direction" note.
  - `ink-navy` — the dark hub and the matrix header: the thing everything else revolves around.
- Pick the accent by topic, then keep it: a modality coloured plum on slide 10 stays plum on slide 14.
- Text is `ink-display` for titles, `ink` for body, `ink-muted` for subtitles and captions. All three read on every tint at 10:1 or better; `ink-muted` reads at 4.45:1 on `amber-200`, so keep muted text off that cell.
- Accent as text: `teal-700`, `forest-700`, `plum-700`, `amber-900`, `slate` pass 4.5:1 on `surface` and the tints. `amber-700`, `amber-600`, `leaf-500`, `gray-500` and `ink-soft` do not: use them for fills, strokes, discs and headings of 24px or larger only. The source deck sets "Moving forward" in `amber-700` at 10px; this system keeps the colour and moves such text to `amber-900`.
- Text on a solid accent is `on-dark`: `teal-700` (5.4:1), `forest-700` (6.3:1), `plum-700` (8:1), `ink-navy` (14:1) all pass; `amber-600` (3.4:1) only at `step-title` bold or larger.
- Matrix cells encode a scale with lightness, not hue: `teal-600` (mandatory) → `teal-300` (partly) → `amber-200` (expected) → `surface-gray` (not specified). Always pair the cell with a Legend.
- No gradients, no shadows, no transparency. Depth comes from tint on white and from one solid accent per element.

### Type

- Two faces: `display` (Cambria; Source Serif 4 as the hosted fallback) for titles, numbers and letters-as-shapes; `sans` (Calibri; Source Sans 3 as the hosted fallback) for everything else. Never set body copy in the serif.
- Scale, on a 960 × 540 canvas: `slide-title` 35, `slide-title-long` 28, `section-title` 21, `group-title` 19, `card-title` 15, `step-title` 14, `body` / `subtitle` 13, `body-small` / `cell-head` 11, `kicker` / `caption` / `label` 10. In PowerPoint the same sizes are 26 / 21 / 16 / 14 / 11 / 10.5 / 9.5 / 8.5 / 7.5 pt.
- 11px is the floor for text the audience reads; 10px only for kickers, captions and inline labels. Nothing smaller, ever, on a projector.
- Weight carries hierarchy: titles and step titles bold, body regular. Italic is reserved for `caption`.
- Alignment is left everywhere except numbers in a StatRow and letters in a disc (centred).

### Layout

- Canvas 960 × 540 (`slide-w` × `slide-h`), inset `slide-margin` (38px) on the top, left and right. The title block (kicker, title, subtitle) occupies the first 110px; content starts at y ≈ 150.
- Content width is `content-w` (884px). Split it 2 : 3 for problem-and-method slides (ProblemCard 340px, arrow, StepStack 500px), or into equal columns of `space-3` gutters for tiles and timelines.
- Cards: `radius-md`, padding `space-4` (`space-3` when a slide holds six or more). Full-height panels use `radius-lg`. Chips and table cells use `radius-sm`.
- Discs are `radius-pill` at `disc-sm` (28) for numbers inside rows, `disc-md` (44) for icons, `disc-lg` (56) for the FAIR letters.
- Whitespace is `space-5` (24px) between a problem and its method, `space-3` (12px) between sibling cards, `space-2` (8px) between stacked rows. Do not fill a slide; a slide with 40% white is right.
- Figures from papers go full-width or in the left 60%, with the explanatory StepStack beside them; captions sit under in `caption` with the source.

### Diagrams

- A diagram shows a mechanism, not a decoration. Prefer: ProcessFlow for sequences with one highlighted stage; CycleDiagram for loops that return to a hub; Timeline for change across eras; MatrixTable for "which venue requires what"; TileGrid for parallel options.
- Arrows are `leaf-300` block arrows when they mean "leads to", `ink-soft` hairlines with a triangle head when they connect steps. One arrow style per diagram.
- Label every node with a word; never rely on colour alone. Numbers inside discs run 1 → n in reading order.
- Highlight one thing per diagram (the active step in `ink-navy`, the "moving forward" column in a dashed `amber-700` outline). Two highlights is none.

### Borders, states, motion

- Cards have no border; the tint is the edge. The only borders are hairline rules (`rule`, `rule-w`) inside tables and the dashed `amber-700` outline marking what is provisional.
- No animation or build-in. If a slide needs to be revealed in stages, duplicate the slide and add one element per copy.

## Iconography

- The 14 glyphs in `assets/Icons` are white (`on-dark`) PNGs at 256 × 256, drawn from the deck. Place each on a `teal-700`, `forest-700` or `plum-700` disc at `disc-md`, never bare on a tint (a white glyph vanishes).
- Use them to name a stage, not to decorate: `database` = pretraining data, `gears` = fine-tuning, `flask` = in-silico or wet-lab experiment, `bar-chart` = result, `dna` = genomes, `microscope` = microscopy, `layers` = multi-omics, `eye` = vision / imaging, `seedling` = plants, `cubes` = models, `book` = literature, `magnifier` = retrieval, `cycle` = continuous update, `growth-curve` = exponential growth (this one is drawn in `amber-700` and `on-dark`; place it on `surface`).
- No emoji, no clip art, no photographs as decoration. A picture on a slide is a figure with a caption and a source.

## Using the components

Each component under `components/` is one slide pattern from the deck, rendered as HTML with classes from `components/bundle.css` on top of `tokens.css`. Rebuild them in PowerPoint or Claude Slides with the same tokens; the previews are the reference for proportions. The pattern-to-purpose map and the running order of a discussion deck are in `discussion-playbook.md`.
