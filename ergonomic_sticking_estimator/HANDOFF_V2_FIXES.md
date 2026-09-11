# HANDOFF — V2 fixes for the segmental sticking decoder

**Audience:** an implementing model. Follow this document exactly and in order. Where it
says "expect", verify. Where a decision is not specified, prefer the smallest change.

**Diagnosis this fixes (2026-07-04, all demonstrated by probes):**

1. **Confidently wrong on the canonical groove.** 8th-note hihat ostinato + snare backbeat
   decodes as alternating `R L R L` on the hihat (conf 0.8–0.96) and puts the snare backbeat
   on **R, 20 ms after an R hihat hit** — physically impossible. Causes: (a) slow
   same-surface alternation is *rewarded*, and no "one hand rides the timekeeping surface"
   concept exists; (b) the hard 3.0 speed-floor penalty applies only *same-surface*, so
   impossible same-hand near-chords across surfaces pay only soft reach; (c) `_side_affinity`
   pushes the hihat toward **L**, i.e. encodes *open-handed* play while config says
   `style="crossed"` (crossed = right hand crosses over to the hihat).
2. **Flagship smoke case regressed.** `python -m src.decoder` case 1 (accented paradiddle)
   yields `R L L R`×4 with 12/16 flagged instead of paradiddles. Contributing wart:
   `w_metric_start` is a per-cell *bonus*, so the decoder is paid to over-segment.
3. **Confidence is O(n²) and non-posterior.** Two full re-decodes per hand strike; measures a
   max-margin, not probability mass; and conflates "which relative pattern" (usually knowable)
   with "which absolute hand" (often honestly ~0.5), producing walls of flags.
4. **Repetition unexploited.** Identical repeated figures can decode with different stickings.
5. **No committed regression harness** — which is how (2) shipped silently.

---

## 0. Ground rules

**Read before coding** (nothing else is required):
- `DESIGN_STAGE5_SEGMENTAL.md` (the design contract — still in force)
- `src/decoder.py`, `src/scoring.py`, `src/templates.py`, `src/types.py`
- `src/pipeline.py` (only if doing Phase 6)

**Do NOT:**
- Touch `src/stage0_ingest.py`, `stage1_onsets.py`, `stage1p5_toms.py`, `stage2_beats.py`,
  `stage3_consolidate.py` — or anything in `../vision_sticking_estimator/`.
- Remove or weaken the generic fallback template, or add any hard pattern bank.
- Add a cost term charged per-cell that depends on segmentation without threading its state
  through the DP state — every per-adjacency term must be charged identically whether the
  adjacency is inside a cell or across a seam (segmentation invariance; see
  `pair_feasibility`'s docstring for the precedent).
- Inflate confidence. If a case is genuinely symmetric, it must stay ~0.5 on the *parity*
  axis. The fixes below make output more *legible*, never more certain than the evidence.
- Tune weights beyond the defaults given here. Calibration waits for the inter-drummer set.

**Env (Windows):** run everything from this directory with the shared venv:
`..\.venv\Scripts\python.exe -m src.<module>` (always `-m`; `src/types.py` shadows stdlib
`types` if run as a script). Set `PYTHONUTF8=1`. Phases 0–5 need no audio stack; only
Phase 6 needs ADTOF/madmom (`TF_USE_LEGACY_KERAS=1`, numpy<2 pinned — do not upgrade numpy).

**Workflow:** phases are strictly ordered. After each phase run
`..\.venv\Scripts\python.exe -m src.synth_checks` and update `EXPECTED_FAIL` (Phase 0
defines it). A phase is done only when its gate passes AND no previously-passing case broke.

---

## Phase 0 — committed regression harness (`src/synth_checks.py`)

**Why:** the paradiddle regression shipped because the only checks were non-asserting
`__main__` prints. Everything below is verified through this harness.

Create `src/synth_checks.py`, runnable via `python -m src.synth_checks`, no audio deps
(imports only `types`, `templates`, `scoring`, `decoder`, stdlib, numpy). Reuse the
stream-building helper style from `decoder.py`'s `__main__` (build fake constraint-dicts
with `Strike` lists; `qpos` = beat position).

Each case is a function returning `(ok: bool, detail: str)`. The runner prints a table and
exits 1 if any case fails that is not listed in `EXPECTED_FAIL`, **or** any case listed in
`EXPECTED_FAIL` passes (stale entry — remove it). Seed `EXPECTED_FAIL` by running once
after writing the cases and recording reality (expected initial state is noted per case).

Hand-strike extraction for assertions: `hands = [a for a in assigned if a.limb != "F"]`,
time-sorted.

**Case A — `groove_ride`.** One bar, 120 bpm. Hihat 8ths: 8 strikes, surface `hihat`,
vel 0.55, IOI 250 ms, `qpos = k*0.5`. Snare backbeat: vel 0.85 at 20 ms after the hihat
hits at qpos 1.0 and 3.0 (same qpos). Assert: every hihat hit is `R`; every snare hit is
`L`; and **no two hits < 70 ms apart share a hand** (any surface combination).
*Initially FAILS. Must pass after Phase 3 (the <70 ms sub-assertion after Phase 1).*

**Case B — `impossible_pair`.** Two strikes 20 ms apart: `hihat` then `snare`, both
vel 0.6. Assert different hands. *Initially FAILS. Passes after Phase 1.*

**Case C — `accented_paradiddle`.** 16 snare strikes, IOI 120 ms, vels
`[0.95,0.4,0.4,0.4]`×4, `qpos = beat + k*0.25`. Assert the limb sequence is exactly
`R L R R L R L L R L R R L R L L` **or** `R L R R` repeated 4× (both are legitimate
paradiddle readings; accept either), and ≥ 12/16 hits unflagged.
*Initially FAILS (currently `R L L R`×4, 12 flagged). Passes after Phase 2.*

**Case D — `fast_roll_doubles`.** 8 snare strikes, IOI 60 ms, vel 0.6, `qpos = k*0.25`.
Assert sequence is `R R L L R R L L` or `L L R R L L R R` (doubles; parity free).
*Currently PASSES. Must never break.*

**Case E — `slow_snare_alternation`.** 8 snare strikes, IOI 250 ms, vel 0.6,
`qpos = k*0.5`. Assert strict alternation starting on `R` (metric anchor pins the lead).
*Currently PASSES (verify). Must never break — this is the anti-regression guard for
Phase 3: the ostinato waiver must NOT apply to the snare.*

**Case F — `cross_kit_fill`.** 8 strikes alternating surfaces
`snare,tom0,snare,tom0,...`, IOI 180 ms, vel 0.6, `qpos = k*0.25`. Assert strict
alternation `R L R L R L R L` or with ≤ 2 flags, and no same-hand run ≥ 3.
*Initially FAILS (currently emits an `L L` and flags 6/8). Passes after Phase 2.*

**Case G — `linear_with_kicks`.** The `decoder.py` `__main__` case 4 spec verbatim
(`snare,snare,kick,kick`×2). Assert hands read `R L … R L` (kicks `F` untouched).
*Initially FAILS (second group reads `R R?`). Passes after Phase 2.*

**Case H — `flat_roll_parity_split`** *(add in Phase 4, not before)*. Case D's input.
Assert: per-hit `pattern_confidence ≥ 0.8` while clip `parity_confidence ≤ 0.65`, and the
absolute `confidence` stays ≤ ~0.75 (honesty preserved).

**Case I — `repeated_figure`** *(add in Phase 5)*. A 4-strike accented figure (Case C's
first beat) at t≈0 and the identical figure (same IOIs, vels, qpos mod 4) at t≈4 s, with
8 hihat 8ths between them. Assert both occurrences decode to identical token sequences.

Keep total runtime under ~60 s (see the Phase 4 note on confidence cost; before Phase 4
these tiny cases are fine with the O(n²) flip method).

**Gate:** harness runs; `EXPECTED_FAIL` documents current reality; passing set matches the
"currently PASSES" notes above (if reality differs, record it in `EXPECTED_FAIL` with a
comment — do not "fix" anything yet).

---

## Phase 1 — hard physics enforcement

**Why:** the only inviolable-physics penalty (`3.0` at `ioi < floor*0.5`) applies solely to
same-surface pairs; cross-surface same-hand pairs at 20 ms pay only soft reach (~0.58).

**Change 1 — cross-surface speed floor.** In `decoder.py::_cell_feasibility`, the
`same_hand and not same_surf` branch currently charges only `_reach_travel(...)`. Add: if
`ioi < config.single_hand_min_ioi_ms / 1000.0`, add a hard `3.0 * dc.w_feasible` on top of
the reach cost. (One hand cannot strike two different surfaces faster than it can restrike
one — cross-surface is strictly harder than a same-surface rebound, which gets no hard
penalty until `floor*0.5`.)

**Change 2 — document the Stage-4 situation honestly.** `stage4_constraints.py` computes
`admissible` sets that `decode_stream` never reads, and its "spatial exclusivity" is a
comment. Do NOT build new machinery. Add one paragraph to the `stage4_constraints.py`
module docstring and one to `decoder.py`'s: in this project, hard physics is enforced by
the decoder's hard pair costs (kick→F by surface; the two speed floors above); Stage 4's
admissible sets are currently advisory/unused. Keep the code (shared backbone shape).

**Gate:** Case B passes; the <70 ms sub-assertion of Case A passes; D and E unchanged.
Remove B from `EXPECTED_FAIL`.

---

## Phase 2 — sign and reference fixes

**Why:** three terms have wrong sign/reference and together broke Case C and F.

**Change 1 — metric-start becomes a penalty for weak starts (kills the over-segmentation
subsidy).** In `decoder.py::_decode`, replace
`metric_bonus = -dc.w_metric_start * _metric_strength(...)` with
`metric_cost = dc.w_metric_start * (1.0 - _metric_strength(hands[i].quantized_pos))` and
set the default `w_metric_start = 0.3`. Now starting a cell on the beat costs 0, on a weak
position costs up to 0.3, and adding cells can never *reduce* total cost. Rename the local
variable; update the docstring line in the module header.

**Change 2 — stop rewarding slow same-surface alternation.** In
`scoring.py::pair_feasibility`, the different-hand same-surface branch returns
`weight * (frac - _NEUTRAL)`, which is **negative** (a reward) when slow — this is the term
that makes any slow one-surface stream alternate and (with Case A) alternates the hihat.
Replace with `weight * max(0.0, frac - _NEUTRAL)`: penalize fast singles, stay *neutral* on
slow singles. Alternation must win by the alternatives costing more (slow doubles still
pay `_NEUTRAL - frac > 0`), not by subsidy.

**Change 3 — speed-scale the run-length penalty.** Same function: the run≥3 surcharge
`(run_len - 2) * (0.5 + frac)` asserts "rebound can't sustain 3+" — true when fast, false
when slow (slow one-hand tapping is physically easy; it's *idiomatically* disfavored, which
the remaining base cost covers). Replace the factor `(0.5 + frac)` with
`(0.15 + 0.85 * frac)`. (Keep the base `_NEUTRAL - frac` same-hand cost as is — it is the
speed-independent-ish deterrent that stops side-affinity farming one-hand runs, a
previously-fought failure.)

**Change 4 — crossed-style convention replaces geometry for cymbals.** Two parts:

  a. In `types.py`, the `NATURAL_HAND` hihat entries are inverted for every setup
     ("crossed" *means* the lead hand crosses over to the hihat). Fix to:
     `right_crossed: hihat→"R"`, `right_open: hihat→"L"`, `left_crossed: hihat→"L"`,
     `left_open: hihat→"R"`. Leave snare/tom/cymbal entries unchanged. Add a comment noting
     this diverges from the vision project's copy deliberately (theirs is also inverted —
     flagged for that project separately; do not edit it).

  b. In `decoder.py::_side_affinity`, geometry is the wrong model for the hihat under a
     crossed setup (it penalizes exactly the conventional hand). New rule: normalize the
     surface (`"tom3" → "tom"` via `surface.rstrip("0123456789")`); if the normalized
     surface has a non-None entry in `NATURAL_HAND[f"{config.handedness}_{config.style}"]`,
     return `0.0` if `hand == natural` else `dc.w_side`; otherwise (snare → None entry,
     and all `tomN`) keep the existing geometric x rule. Import `NATURAL_HAND`. Note
     `NATURAL_HAND`'s generic `"tom"` entry must NOT capture sub-toms — the normalization
     above would send `tom0/1/2` to the `"tom"` entry, which defeats the geometry that
     places them left→right. So: apply the NATURAL_HAND rule **only** for `hihat` and
     `cymbal`; snare and all toms keep the geometric rule. (Implement it as an explicit
     `if base in ("hihat", "cymbal")` — simple beats clever.)

**Change 5 — local, per-surface dynamic reference.** `scoring.py::dynamic_reference` is
clip-global and surface-blind; the design (§2 C_accent) says *local*. Replace usage in
`decoder.py::decode_stream`: instead of one `(ref, spread)` pair, precompute per-strike
accent levels once, up front:
  - For each hand strike, its reference population = hand strikes on the **same normalized
    surface** (hihat vs snare vs tom…) within ±4 s (config: `dynamic_ref_window_s = 4.0`);
    if that population has < 4 strikes, fall back to all hand strikes in the window; if
    still < 4, the whole clip.
  - `ref = median(vels)`, `spread = max(IQR(vels), 0.1)` over that population;
    `accent_level_i = clip((v_i - ref)/spread, -2, 2)` (same math as `_accent_level`).
  Store as `accent_levels: list[float]` parallel to `hands`, thread it into
  `_span_coverings`/`score_accent` in place of `(ref, spread)` (change `score_accent` to
  accept precomputed observed levels for the cell — keep the old signature working for its
  `__main__` unit checks or update those checks too, your choice; keep the module runnable).

**Gate:** Cases C, F, G pass (remove from `EXPECTED_FAIL`); D and E still pass; Case A
still fails only on the "hihat all R" part (alternation subsidy is gone but nothing yet
*prefers* riding — that is Phase 3; if A's snare-hand or <70 ms parts regressed, stop and
fix). Also run `python -m src.scoring` and `python -m src.templates` — their self-checks
must still pass (update the scoring `__main__` if you changed signatures).

---

## Phase 3 — the timekeeping (ostinato) layer

**Why:** drummers dedicate one hand — the lead hand, crossed or not — to a repeating
cymbal/hihat timekeeping line. The model has no such concept, and neutrality (Phase 2)
alone doesn't produce riding: the run-length base cost still makes one-hand riding pay,
so alternation still wins on Case A. This phase makes riding the preferred reading of a
regular cymbal ostinato while leaving snare/tom material untouched.

**Detection (structural, pre-decode).** New helper in `decoder.py` (or a small
`src/ostinato.py` if you prefer; keep it < 60 lines):
`_ostinato_surfaces(hands, config) -> set[str]`. A surface qualifies iff:
  - its normalized name is in `config.timekeeping_surfaces` (default `["hihat", "cymbal"]`
    — timekeeping idiom lives on cymbals; deliberately NOT snare/toms, see Case E), and
  - it has ≥ `config.ostinato_min_hits` (default 6) hand strikes, and
  - its IOIs are regular: `IQR(iois) / median(iois) ≤ config.ostinato_ioi_cv` (default 0.3).
Return the set of *raw* surface names that qualify. Compute once in `decode_stream`, pass
into the DP (via `_cell_feasibility` args or a small context object).

**Cost changes, all inside `_cell_feasibility` / `pair_feasibility`, applied only when BOTH
strikes of an adjacency are on the SAME qualifying ostinato surface:**
  1. Same-hand, `ioi ≥ floor`: cost **0** (riding is free — waive both the base
     `_NEUTRAL - frac` cost and the run-length surcharge; do not let `run` grow past 2 for
     this purpose, or simpler: skip the `pair_feasibility` call and charge 0, setting
     `run = 1`). Below `floor`, normal double/floor logic applies unchanged (fast hihat
     16ths beyond one hand's ceiling still force alternation).
  2. Different-hand, `ioi ≥ floor`: add `config.w_ostinato_switch` (default 0.8) — changing
     the riding hand mid-ostinato is uneconomical. When `ioi < floor`, charge nothing extra
     (alternation is forced there, don't punish physics).
Implement as a wrapper/branch where `_cell_feasibility` currently dispatches — keep
`pair_feasibility` itself pure (it has external callers/tests); the ostinato branch lives in
the decoder.

Non-ostinato adjacencies (including hihat→snare crossings) are completely unaffected.
The snare backbeat then lands on `L` for free: with `R` riding, a same-hand
hihat→snare→hihat excursion pays reach twice, `L` pays nothing, and Phase 2's
NATURAL_HAND rule holds the ride on `R`.

**Config:** add `timekeeping_surfaces`, `ostinato_min_hits`, `ostinato_ioi_cv`,
`w_ostinato_switch` to `PipelineConfig` (Stage-5 section) with the defaults above and
one-line comments.

**Gate:** Case A passes fully (remove from `EXPECTED_FAIL`). Cases C–G unchanged — E
especially (snare must not qualify as ostinato). Re-run everything.

---## Phase 4 — forward–backward posteriors + factored confidence

**Why:** confidence currently costs 2 full re-decodes per hand strike (O(n²), minutes on
real clips), measures single-path margins rather than probability mass, and reports one
number that conflates two questions: *which relative pattern* (usually answerable) and
*which absolute hand leads* (often honestly ~0.5). The design doc promised a template
margin (§3) that was never built. This phase replaces the flip loop with one
forward–backward pass and splits the output.

**Step 1 — refactor `_decode` into a reusable lattice sweep.** The triple loop in
`_decode` (span `i→j` × covering × predecessor state) defines the lattice. Refactor so the
same enumeration serves two semirings:
  - **min-sum** (existing Viterbi; keep behavior identical — Cases C–G are the regression
    net), and
  - **log-sum-exp**: `alpha[j][state] = logsumexp over incoming (alpha[i][s'] − total/temp)`
    forward, and the mirror-image `beta` backward from the terminal. Work in log domain
    with `math` or numpy `logaddexp`; scores are `−cost/dc.temp`.
Keep the `forced` filter working in both (it just drops coverings).

**Step 2 — exact per-hit marginals.** For each lattice transition (span `i..j`, covering
tokens, prev-state `s'` at `i`, state `s` at `j`) compute its posterior log-mass
`alpha[i][s'] − cost/temp + beta[j][s] − logZ` (with `logZ` = logsumexp over terminal
states). For every hit `k` in `i..j`, accumulate `exp(mass)` into `P(hit_k = tokens[k−i])`.
Normalize per hit (should already sum to ~1; renormalize defensively). This replaces the
per-hit flip re-decodes in `decode_stream` entirely — total cost becomes ~3 decodes' worth
regardless of n. `limb_probs`/`confidence` now come from these marginals.

**Step 3 — parity confidence (one number per clip).** After the best decode, build
`forced_flip = {k: ("L" if best_limbs[k]=="R" else "R") for all k}` and run one min-sum
decode with it. `parity_confidence = 1/(1+exp((cost_flip − cost_best)/dc.temp))` — i.e. the
Gibbs weight of the best reading vs its full mirror. On symmetric material this is ~0.5
(honest); accented/kit-anchored material pushes it up.

**Step 4 — pattern confidence (per hit).** Condition on parity: run the forward–backward
of Step 2 once more with `forced = {k0: best_limbs[k0]}` where `k0` = the first hand
strike of the best path. `pattern_confidence_k = max(P_cond(R), P_cond(L))` from the
conditional marginals. Interpretation: "given the lead-hand choice, how pinned is this
hit's hand." Limitation (accept for v1, note in the docstring): a mid-clip parity shift
(e.g. the rll→lrr sample) is only captured by the unconditional marginals, not by the
single global parity number.

**Step 5 — output plumbing.**
  - `types.py::AssignedStrike`: add `pattern_confidence: float = 1.0` (default keeps feet
    valid). Keep `confidence` = unconditional marginal max (absolute, as before — meaning
    unchanged, now exact).
  - `flagged_ambiguous` now keys off **pattern** confidence:
    `flagged_ambiguous = pattern_confidence < config.ambiguity_threshold`. Parity
    uncertainty is reported once, not smeared over every hit.
  - `pipeline.py::PipelineResult`: add `parity_confidence: float`.
    `sticking_sequence(...)`: if `parity_confidence < config.ambiguity_threshold`, prefix
    the sequence with `"(lead hand uncertain — sticking shown up to L/R mirror) "`.
    `render_grid`: add a `pat` column next to `conf`.
  - `decode_stream` returns as before; put `parity_confidence` on the function as a second
    return value or a small result tuple — pick the least invasive shape and update
    `pipeline.py` accordingly (it is the only caller besides the harness).

**Step 6 — total decode count check.** `decode_stream` must now perform exactly: 1 min-sum
(best path) + 1 forward–backward (marginals) + 1 min-sum (flip, parity) + 1
forward–backward (conditioned, pattern). No per-hit loops over decodes remain.

**Gate:** add Case H; it passes. Cases A, C–G still pass — C's flag count may only
*improve* (pattern-flags ≤ old flags). Sanity: on Case D, unconditional per-hit confidence
stays ≤ ~0.75 (mass is genuinely split between mirrors), while pattern confidence is high —
that separation is the whole point. Runtime of the full harness drops or holds.

---

## Phase 5 — repetition coupling

**Why:** repeated figures should decode identically; today each occurrence is independent,
so mirrors can flip between occurrences and per-occurrence evidence never accumulates.

**Design (two-pass, bounded, honest):**
  1. Decode normally (Phase 4 machinery).
  2. Build a **signature** per decoded cell: `(tuple of normalized surfaces,
     tuple of round(qpos_i − qpos_start, 2), tuple of sign-quantized accent levels
     (+1 if > 0.75, −1 if < −0.75, else 0))`. Group cells by signature; keep groups with
     ≥ 2 occurrences.
  3. For each group, find the modal token-sequence among occurrences whose per-cell
     decision was non-trivial — count only occurrences where the cell's mean
     `pattern_confidence ≥ config.ambiguity_threshold` (don't let coin-flips vote). If the
     modal share among voters is ≥ `config.repeat_min_share` (default 0.6) and there was
     ≥ 1 voter, the group gets a preferred token-sequence.
  4. Re-decode **once** with a discount: in `_span_coverings`, a covering whose exact
     token tuple equals the preferred sequence for a span whose signature matches a
     preferred group gets `− config.w_repeat` (default 0.3). Exact tokens only — the
     mirror does NOT get the discount (parity consistency is the point).
  5. Recompute Phase-4 confidences on the re-decoded lattice (the discount is part of the
     model for that pass, so marginals stay self-consistent).
Implementation note: pass an optional `repeat_prefs: dict[signature → tokens]` down to
`_span_coverings`; computing a span's signature needs the same helper as step 2 — write it
once in the decoder module. Cap: `w_repeat` must stay ≤ `fallback_tax` (a repetition nudge
may break ties; it must never beat physics — same philosophy as `C_idiom`).

**Gate:** add Case I; it passes. All previous cases unchanged (w_repeat is too small to
flip any of them — verify, don't assume).

---

## Phase 6 — real-audio validation script (no tuning)

Recreate the lost end-to-end probe as a **committed** script `eval_samples.py` (project
root, next to `README.md`):
- Loads the ADTOF model once, then runs `pipeline.run` over every audio file in
  `../samples/sample_2/` (filenames encode rough GT, e.g. `rlkk`, `rll_lrr` — underscore
  means the sticking shifts mid-clip; letters are lowercase for aesthetics, not ghosts).
- Prints per clip: filename, `sequence`, `parity_confidence`, mean hand
  `pattern_confidence`, flag count. No assertions on GT (weak ground truth — the drummer
  played concurrent kicks the GT omits; sample_1 over-detects 23 vs 16 onsets). This is an
  eyeball artifact, not a scoreboard.
- Env guard: wrap imports in try/except and exit with a clear message if the audio stack
  is unavailable. Document the invocation at the top:
  `TF_USE_LEGACY_KERAS=1 PYTHONUTF8=1 ..\.venv\Scripts\python.exe eval_samples.py`.

Then update docs, minimally:
- `README.md` Status section: Stage 5 built; list the V2 fixes in one sentence each;
  point to this handoff and `src/synth_checks.py`.
- `DESIGN_STAGE5_SEGMENTAL.md`: append a short "V2 addendum (2026-07-04)" noting: metric
  start is a weak-start penalty; slow same-surface alternation is neutral not rewarded;
  timekeeping/ostinato layer added (cymbals only); NATURAL_HAND convention overrides
  geometry for hihat/cymbal; confidence = exact posteriors, factored into pattern × parity;
  repetition discount. Three-to-eight lines total; do not rewrite the document.

**Gate:** `eval_samples.py` runs end-to-end on sample_2 without error; sequences are
qualitatively no worse than the pre-fix state recorded in the sticking (`rll` groups still
emerge in `rll_lrr`, `R L K K…` shape holds in `rlkk`); `parity_confidence` behaves
sanely (higher on accented/kit-anchored clips than on symmetric snare-only ones).

---

## New config fields (summary)

| field | home | default | phase |
|---|---|---|---|
| `dynamic_ref_window_s` | PipelineConfig | 4.0 | 2 |
| `timekeeping_surfaces` | PipelineConfig | `["hihat","cymbal"]` | 3 |
| `ostinato_min_hits` | PipelineConfig | 6 | 3 |
| `ostinato_ioi_cv` | PipelineConfig | 0.3 | 3 |
| `w_ostinato_switch` | PipelineConfig | 0.8 | 3 |
| `repeat_min_share` | PipelineConfig | 0.6 | 5 |
| `w_repeat` | PipelineConfig | 0.3 | 5 |
| `w_metric_start` | DecoderConfig | 0.3 (semantics: weak-start **penalty**) | 2 |

`AssignedStrike.pattern_confidence` (Phase 4) and `PipelineResult.parity_confidence`
(Phase 4) are the only type changes.

## Definition of done

1. `python -m src.synth_checks` passes with an **empty** `EXPECTED_FAIL`.
2. `python -m src.scoring`, `python -m src.templates`, `python -m src.decoder` all still
   run their self-checks clean.
3. `decode_stream` performs a constant number of lattice sweeps (no per-hit re-decodes).
4. `eval_samples.py` committed and demonstrated on sample_2.
5. No file outside `ergonomic_sticking_estimator/` modified; no changes to stages 0–3.
6. Docs updated per Phase 6; every new weight appears in the table above and in code with
   a one-line comment.

**Out of scope (do not attempt):** weight calibration (waits on the ~10-clip inter-drummer
agreement set, still to be collected); mid-clip parity-shift segmentation; flam/drag
templates (blocked on Stage-3 grace clustering); left-handed/open-handed validation beyond
keeping the code paths intact; any vision-project work.
