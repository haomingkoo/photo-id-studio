# Photo ID Studio design

Extends the shared kooexperience design core (`~/.agent-memory/design/DESIGN-core.md`). Where this file is silent, the core applies.

## Audience and job
Someone renewing a passport or IC who needs a photo that will not be rejected. They arrive with one photo and one question: will it pass?

## Mood
Calm, official, trustworthy. An official form done well. Light theme, because passport photos are judged on white and a light page previews the result honestly.

## Tokens (`web/site.css`)
- Ground `#f4f5f7`, surface `#ffffff`, sunken `#eef0f3`, line `#d7dbe1`, control border `#8a929d` (3.1:1 on white).
- Ink `#161a20`, muted `#4f5764` (7.3:1 on white, 6.7:1 on ground).
- One accent, calm blue `#1f5fa6`: primary button, links, focus ring, active nav. Nothing else.
- Status only for check results: pass `#1c6e43`, worth checking `#8a5300`, fail `#b3261e`, each on its own pale wash, always with an icon and words.
- Type: IBM Plex Sans 400/500/600, self-hosted. Chosen for its institutional, form-like clarity and distinct numerals for pixel sizes. Sizes 13, 15, 18, 24, 32. Tabular numerals everywhere.
- Radius 6 controls, 10 cards, 16 panels. No shadows on the page.

## Signature move
The spec card: the cropped photo shown at the document's real proportions with its pixel width and height drawn as rulers. The empty state uses the same card with a dashed head-and-shoulders outline, so the first screen already shows the target.

## Rules for this app
- Upload is the first thing on the screen. Advanced options live under "More options".
- Results read as a checklist in three groups: Needs fixing, Worth checking, Passed (folded). Every line is a plain sentence with what to do next.
- Never show check codes, backend names, scores or ratios. Every code in `app/services/pipeline.py` needs plain copy in the `CHECKS` map in `web/index.html`; `tests/test_web_copy.py` enforces this.
- Every state is designed: empty, loading ("Checking your photo"), result, request error with retry, rules failed to load.
- Engineering detail belongs in the README, not the Photo tips page.
- Legal pages use the same header, footer and stylesheet.

## Dials
ENERGY 1 / RHYTHM 1 / MOTION 1. One motion: the loading bar, frozen under reduced motion.
