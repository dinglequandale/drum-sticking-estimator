# Drum Sticking Grammar Bank (corpus seed)

Purpose: the symbolic source material for the **sticking-grammar prior** (leg 3 of the
fusion architecture). This trains P(limb | history, surface, metrical position, density)
with **no real audio required**. See memory `project-fusion-architecture`.

This file is a human-readable, hand-verifiable seed. It will later be parsed into the
corpus schema (per stroke: `limb`, `surface`, `metrical_position`, `accent_role`,
`stroke_type`). Konrad's uploaded method-book PDFs are the authoritative cross-check and
the source of bulk volume; this file is the spine they hang on.

### Notation legend
- `R` / `L` — normal tap (right / left hand)
- `R!` / `L!` — **accented** stroke
- `(l)` / `(r)` — **grace note** (flam = one grace; drag = two, written `(ll)` / `(rr)`)
- `°` — buzzed/multiple-bounce stroke
- Every rudiment has a **mirror** form (swap all R↔L). Both are valid sticking; the corpus
  MUST include both, because lead-hand is contextual, not fixed — this is the honest-ceiling
  truth (audio alone can't fix the lead hand). Mirrors are implied, not all written out.

### Verification status
- ✓ = standardized; high confidence.
- ⚠ = grace-note placement / variant convention is subtle → **verify against PDFs** before
  committing to the corpus. Do NOT trust these as-written yet.

---

## 1. Roll family

| # | Rudiment | Sticking (R-lead) | Subdiv | Status |
|---|----------|-------------------|--------|--------|
| 1 | Single Stroke Roll | `R L R L R L R L …` | any | ✓ |
| 2 | Single Stroke Four | `R L R L` (one beat, e.g. sextuplet pair) | trip | ✓ |
| 3 | Single Stroke Seven | `R L R L R L R!` | trip | ✓ |
| 4 | Multiple Bounce (buzz) Roll | `R° L° R° L° …` (each hand buzzed) | any | ✓ (special: not discrete) |
| 5 | Triple Stroke Roll | `R R R L L L …` | trip | ✓ |
| 6 | Double Stroke Roll (open/long) | `R R L L R R L L …` | any | ✓ |
| 7 | Five Stroke Roll | `R R L L R!` | 16th+acc | ✓ |
| 8 | Six Stroke Roll | `R! L L R R L!` (accent–double–double–accent) | — | ⚠ multiple accepted forms |
| 9 | Seven Stroke Roll | `R R L L R R L!` | — | ✓ |
| 10 | Nine Stroke Roll | `R R L L R R L L R!` | — | ✓ |
| 11 | Ten Stroke Roll | `R R L L R R L L R! L!`? | — | ⚠ even-roll variant |
| 12 | Eleven Stroke Roll | `R R L L R R L L R R L!` | — | ✓ |
| 13 | Thirteen Stroke Roll | `R R L L R R L L R R L L R!` | — | ✓ |
| 14 | Fifteen Stroke Roll | `R R L L R R L L R R L L R R L!` | — | ✓ |
| 15 | Seventeen Stroke Roll | `R R L L R R L L R R L L R R L L R!` | — | ✓ |

Pattern note (odd rolls): *k* double strokes + one accented release single. This rule
generalizes cleanly and is the high-value structural fact for the grammar.

## 2. Diddle / paradiddle family

| # | Rudiment | Sticking (R-lead) | Status |
|---|----------|-------------------|--------|
| 16 | Single Paradiddle | `R! L R R` ` L! R L L` | ✓ |
| 17 | Double Paradiddle | `R! L R L R R` ` L! R L R L L` | ✓ |
| 18 | Triple Paradiddle | `R! L R L R L R R` ` L! …` | ✓ |
| 19 | Single Paradiddle-diddle | `R! L R R L L` (usually continuous, R-lead) | ✓ |

Structural fact: paradiddles place the **accent on the lead hand** and use the diddle
(double) to *switch* which hand leads the next group — this is the core mechanism the
grammar should learn for accent→hand and lead-hand handoff.

## 3. Flam family

| # | Rudiment | Sticking (R-lead) | Status |
|---|----------|-------------------|--------|
| 20 | Flam | `(l)R!` | ✓ |
| 21 | Flam Accent | `(l)R! L R` ` (r)L! R L` (triplet) | ✓ |
| 22 | Flam Tap | `(l)R! R` ` (r)L! L` | ✓ |
| 23 | Flamacue | `(l)R L! R L (r)L`? | ⚠ verify accent+grace |
| 24 | Flam Paradiddle | `(l)R! L R R` ` (r)L! R L L` | ✓ |
| 25 | Single Flammed Mill | `(l)R R L R`? | ⚠ |
| 26 | Flam Paradiddle-diddle | `(l)R! L R R L L` ` (r)L! …` | ✓ (verify grace only on note 1) |
| 27 | Pataflafla | `(l)R L R (r)L` | ⚠ flams on outer notes — verify |
| 28 | Swiss Army Triplet | `(l)R! R L` ` (r)L! L R` | ✓ |
| 29 | Inverted Flam Tap | `(l)R (r)L` alternating, flam→tap inverted | ⚠ |
| 30 | Flam Drag | `(l)R (rr)L R`? | ⚠ |

## 4. Drag family (incl. ratamacues)

| # | Rudiment | Sticking (R-lead) | Status |
|---|----------|-------------------|--------|
| 31 | Drag (ruff) | `(ll)R` | ✓ |
| 32 | Single Drag Tap | `(ll)R! L` ` (rr)L! R` | ✓ |
| 33 | Double Drag Tap | `(ll)R (ll)R! L` ` …` | ⚠ accent placement |
| 34 | Lesson 25 | `(ll)R L R R`? | ⚠ |
| 35 | Single Ratamacue | `(ll)R L R L!`? | ⚠ |
| 36 | Double Ratamacue | `(ll)R (ll)R L R L!`? | ⚠ |
| 37 | Triple Ratamacue | `(ll)R (ll)R (ll)R L R L!`? | ⚠ |
| 38 | Single Dragadiddle | `R R L R R`? (paradiddle w/ drag) | ⚠ |
| 39 | Drag Paradiddle #1 | `R! (ll)R L R R`? | ⚠ |
| 40 | Drag Paradiddle #2 | `R! R! (ll)R L R R`? | ⚠ |

> **The 13 ⚠ rudiments are the priority targets for your PDFs** — the grace-note hand and
> accent placement in the flam/drag/ratamacue families is exactly what flaky web sources
> mangle. Send anything covering these and I'll lock them down.

---

## 5. Conventions (the real priors — beyond rudiment vocabulary)

These are the *contextual* rules that turn a vocabulary into a grammar. Each is a **soft,
defeasible** prior (never a hard constraint — honest-ceiling discipline).

### 5a. Ergonomic / surface→hand
- **Nearest-hand rule:** a surface tends to be struck by the hand on its side of the kit to
  avoid crossover (left-side hat/snare → L; right-side floor tom/ride → R), unless an
  intentional crossover is used.
- **Ostinato occupies a hand:** in grooves, a continuous hi-hat/ride ostinato is held by one
  hand (usually dominant). That hand is "busy," so simultaneous/interleaved snare & kick voices
  are assigned around it (classic hat=R, snare-backbeat=L). Linear grooves relax this.
- **Doubles as transit:** a double stroke is often inserted to *move the lead hand* to a new
  drum (get the strong hand onto a tom for the next accent) — sticking is chosen for the next
  target, not just the current note.

### 5b. Density / tempo → sticking
- **Single-hand speed floor:** below a per-hand max single-stroke rate, alternating singles are
  feasible; **above it, doubles / paradiddles / buzz take over.** (This floor already exists in
  `stage5` as the speed gate — the grammar should condition on it.)
- **Fast sustained = double-stroke (open) roll or buzz**, not singles.
- **Note density selects the family:** sparse → singles/alternation; dense even → doubles;
  dense with accents → paradiddle family.

### 5c. Accent / dynamics → hand
- **Accents → lead (dominant) hand;** ghost notes → off-hand. Rudiment/inversion choice is often
  driven by getting the accent onto the strong hand.

### 5d. Metric
- **Lead hand tends to align with strong metric positions** (the metric-anchor prior in
  `stage5`) — but **defeasibly**: fast fills that alternate through the barline break it
  (already observed on sample_3 — see `project-metric-anchor-tradeoff`).

---

## 6. Open items / what the PDFs should supply
1. Lock the 13 ⚠ rudiment stickings (grace-note hand + accent placement).
2. Six- and ten-stroke roll: pick the convention(s) your sources use.
3. **Bulk volume:** *Stick Control*-style permutation families (systematic single/double/
   paradiddle/flam permutations) — hundreds of valid sticking lines; high value for the prior.
4. **Real grooves & fills with sticking** (genre-tagged if possible) — the part that lets the
   grammar transcend pure rudiment theory.
5. Any explicit ergonomic/hand-to-drum mapping diagrams (kit-geometry priors for 5a).
