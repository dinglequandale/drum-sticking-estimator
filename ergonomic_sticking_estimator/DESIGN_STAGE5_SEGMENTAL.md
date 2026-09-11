# Stage 5 v2 — Segmental (grouping-based) ergonomic sticking

**Status:** design, not yet built. Supersedes the note-wise HMM in the vision project
(`vision_sticking_estimator/src/stage5_hmm.py`) and the `HANDOFF_LR_PRIORS.md` plan for
*this* project only. The old note-wise model is kept, intact, in the vision project.

---

## 0. The reframed goal (read first — it changes what "correct" means)

**Old problem (now `vision_sticking_estimator/`):** recover *the sticking the drummer
actually used*. Audio cannot do this on many hits (the honest ceiling is *unknowable*),
which is why that project reaches for vision + a learned grammar and hits a
sticking-labeled-**data** bottleneck.

**This project:** given a drum transcription, output the **most economical, physically
plausible, idiomatic sticking**, with calibrated uncertainty. We are not claiming to read
the drummer's mind. We compute the sticking a competent right-handed player would *most
likely* choose, and we report how sure we are.

Consequences to internalize:

- **The confidence number changes meaning.** Old: `P(this is the hand the drummer used)`.
  New: `P(this is the economical/idiomatic choice)` — computable, not ceiling-bound.
- **The flag changes meaning.** Old: "unknowable." New: **low margin between the top
  candidate stickings** — i.e. two readings are near-equal in cost. When that happens it
  is also, by definition, a **don't-care**: if nothing acoustic or ergonomic separates
  `RLRR|LRLL` from a flat `RLRL`, we are *allowed* to return either and flag it. We do not
  fabricate a distinction that the sound and the physics don't contain.
- **The data bottleneck is gone.** The backbone cost is **parametric biomechanics** (reach,
  speed floor, rebound, accent), which needs *no labeled corpus*. A small curated template
  **vocabulary** (dozens of cells) replaces the large **frequency corpus** the old project
  needed. Corpus frequency, if used at all, is a tiny capped tie-breaker only.

**Fixed assumptions for v1 (deliberate simplification):** right-handed, crossed setup.
`lead = "R"`, `off = "L"`, geometry = `KIT_GEOMETRY["right_crossed"]`. No per-player
calibration, no left-handed generality. A fixed lead hand is a *phase anchor by
construction* — it directly removes the "floating phase" failure the old handoff fought.

---

## 1. The object: cover the note stream with cells, each wearing a template

Feet are hard-assigned upstream (kick = `F`), but they stay *in the stream* so linear
groupings like `R L K` are expressible. A **solution** is:

1. a **segmentation** of the strike stream into contiguous **cells** (groupings), with cell
   boundaries restricted to **metric boundaries** from Stage 2 (beat / half-beat /
   quarter-beat). The metric grid *is* the grouping structure — a beat of 16ths is a
   "group of 4."
2. one **template** assigned to each cell, drawn from a vocabulary.

Hands are *not* chosen per note. They fall out of the chosen templates. This is the whole
point: stickings are groupings, not independently-chosen notes.

### Template (the vocabulary unit)

A template is a short, lead-relative sticking cell carrying **three** things:

| part | example |
|---|---|
| hand tokens | `R L R R`, `R L L`, `R L K`, `R l r r L` |
| **accent pattern** per slot | `{accent, normal, ghost}` — e.g. paradiddle = `[accent, normal, normal, normal]` |
| grace structure (optional) | `(l)R` = flam; a near-coincident sub-strike Stage 3 already clusters |

A paradiddle is not "RLRR" — it is **accent-R, L, R, R**. The accent is part of the cell's
identity, which is exactly why matching *dynamics* selects the *sticking*.

Vocabulary source: hand-curated from `vision_sticking_estimator/DRUM_GRAMMAR.md`, "distinct
musical idea once" (see that project's `corpus/schema.md`). We need the *inventory* (a few
dozen cells with accent patterns), **not** frequency statistics over thousands of phrases.

---

## 2. The cost function (this is the entire model)

Minimize over all segmentations `S` and template assignments `{t_c}`:

```
Cost(S) =  Σ_cells [ C_accent(t,c) + C_feasible(t,c) + C_reach(t,c) + C_idiom(t) ]
         + Σ_seams  C_seam(exit_c -> entry_{c+1})
```

Everything except `C_idiom` is **corpus-free physics**.

### C_accent — accents are a first-class *selector*, not a tie-breaker
For each strike compute a **local** accent level `â_i = velocity_i - median(local window)`.
Each template slot declares a target `τ ∈ {+1 accent, 0 normal, -1 ghost}`. Cost is the
mismatch:
```
C_accent = w_a · Σ_i | τ(t[i]) - â_i |
```
This does two jobs at once: it selects the template **and its phase** — because accents sit
on the lead hand, "the loud notes are here" tells you which slots are lead-`R`. A cleanly
accented groove locks the lead hand hard; a flat even roll leaves it free → flagged,
correctly. Use *relative/local* dynamics (accent vs. ghost is relative to the local level,
not the clip peak).

### C_feasible — the physics that makes "buzzy fast = RRLL" fall out
For each same-hand pair in the template (a diddle, e.g. `RR`), score its IOI against the
single-hand speed floor (`single_hand_min_ioi_ms`, `double_prior_speed`):
- As IOI → the floor, a **double gets cheap** and **alternation gets expensive** (it would
  require the same hand back across a surface it just left). Fast even mono-surface passage
  → resolves to `RRLL…` with nothing hardcoded.
- At slow IOI the sign flips: a gratuitous diddle pays a small cost vs. comfortable singles.

### C_reach — kit travel (and a free honesty check)
Intra-cell hand travel from `KIT_GEOMETRY["right_crossed"]`, asymmetric (crossover past kit
center costs more). The "groups of three across toms → RLL" convention should **emerge**
because RLL minimizes travel across a left→right sweep — do **not** hardcode it. If RLL does
not fall out of the reach cost, the reach model is wrong.
- **Known caveat:** ADTOF lumps all toms into one `tom` surface, so a physical left→right
  tom sweep is *not observable from surface alone*. Intra-cell tom reach is therefore weak
  in audio-only v1; treat RLL-around-toms as a soft idiom (`C_idiom`) until/unless tom
  sub-classification exists. Log this honestly rather than pretending we see tom position.

### C_seam — how batches talk to each other (one term per boundary, not per note)
Cost of the hand that *ends* cell `c` vs. the hand that *starts* cell `c+1`: reward
economical continuation, penalize an implied same-hand jump across a big reach. Metric
anchor applies here too (cell downbeats favor the lead hand).

### C_idiom — capped idiom nudge (the ONLY place a corpus could enter)
A small, **capped** log-prior over templates (paradiddle common, `RLLR` rare). Tie-breaker
only; can never outvote physics. Keeps any corpus firmly off the critical path.

---

## 3. Decode: semi-Markov (segmental) Viterbi

Standard segmental DP over strike index `j`:
```
best[j, s] = min over (i<j, template t):
                 best[i, s'] + C_cell(t, strikes[i:j]) + C_seam(s' -> s)
```
- Boundary state `s = (exit_hand, exit_surface)` so seam cost is computable. Small.
- Candidate breakpoints `i` restricted to **metric boundaries** (keeps search bounded and
  encodes that groupings are metric).
- Complexity is trivial: cells ≤ ~8 notes, vocabulary of dozens, tiny boundary state.

**Confidence = template margin.** At each region, compare best vs. second-best covering.
Even roll where `RLRL` and `RRLL` tie → near-zero margin → flagged (and don't-care). Clearly
accented paradiddle → high margin → confident. The honest ceiling is *preserved and made
measurable* — expressed as segmentation/template ambiguity instead of per-note coin-flips.

---

## 4. The one guard you cannot skip — the generic fallback template

A segmental model **is** a bank of patterns, and the original spec forbids a hard pattern
bank because it manufactures confidence (sees rudiments everywhere). Therefore the
vocabulary **must** include a **generic fallback template**: an arbitrary hand sequence
scored on **physics only**, with `C_idiom = 0` (no idiom discount). This makes the
vocabulary provide **discounts, never exclusivity**:
- Off-vocabulary playing competes fairly and, if no idiom beats raw physics, the generic
  cover wins → region reads as "unusual, lower confidence" instead of being force-fit to a
  paradiddle.
- An idiom can only win where it *genuinely* fits better than physics alone.

Without this term we rebuild exactly the false-confidence trap. This single guard is the
difference between the honest and the dishonest version of the segmental model.

---

## 5. Evaluation (the small, *acquirable* validation set)

We are not matching a single ground-truth label (there isn't one). Grade **calibration
against inter-drummer agreement**:
- Have 3–4 competent right-handed drummers stick the same ~10 clips.
- Where they **agree**, the model should be confident and match them.
- Where they **disagree**, the model's margin should be **low** by a similar amount.

This turns the honest ceiling into a measurable target and needs ~10 clips, not thousands
of labeled phrases — the orders-of-magnitude data collapse that the reframe buys. Reusable
from the backbone: onset F-measure (mir_eval) still validates Stages 0–3 independently.

---

## 6. What carries over / what is new

- **Reused as-is (copied into `src/`):** Stages 0–4 (`stage0_ingest`, `stage1_onsets`,
  `stage2_beats`, `stage3_consolidate`, `stage4_constraints`), `types.py`,
  `_madmom_compat.py`. Do not diverge these from the vision project without reason.
- **New, to build here:** `stage5_segmental.py` (templates + cost terms + semi-Markov
  decoder + margin/flag), a small template vocabulary module, a new `pipeline.py` that
  stops at the segmental decoder, and Stage 7-style output (json + grid) adapted to the
  cell/template structure. Stage 6 self-similarity nearly falls out for free — repeated
  cells are repeated template assignments.
- **`PipelineConfig` will shed** left-handed generality and the note-wise HMM knobs
  (`alternation_weight`, `marginal_neighbor_weight`, etc.), replaced by segmental weights
  (`w_accent`, `w_feasible`, `w_reach`, `w_seam`, `idiom_cap`). Keep the physical ceilings
  (`single_hand_min_ioi_ms`, `double_prior_speed`) and `ambiguity_threshold`.

## 7. Build order (suggested)

1. Template vocabulary module: a few dozen hand-verified cells with accent patterns +
   **the generic fallback**. No decoder yet.
2. `C_accent` + `C_feasible` as standalone scorers over (cell, template); unit-check on
   hand-made examples (a loud-RLRR should beat flat-RLRR when the first note is loud).
3. Semi-Markov decoder with metric-boundary candidates; margin → flag.
4. `C_reach` + `C_seam`; confirm RLL-around-toms tendency emerges (or document it as idiom
   given the tom-lumping caveat).
5. `C_idiom` capped; confirm it only breaks ties.
6. Output (json + grid) + wire `pipeline.py`.
7. Collect the ~10-clip inter-drummer agreement set; tune weights against *calibration*,
   not single-label accuracy.
```
