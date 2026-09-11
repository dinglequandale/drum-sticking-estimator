# Handoff: Boosting L/R Hand-Assignment Confidence with Drummer-Grounded Priors

**Audience:** the implementing agent (Sonnet 4.6).
**Goal:** raise both the *accuracy* and the *honest confidence* of left-vs-right hand assignment in Stage 5, using priors that drummers actually use — without breaking the "honest ceiling" (genuinely ambiguous hits must stay flagged).

Read this whole document before editing. It is the single source of truth for this task. The relevant spec is `STICKING_ESTIMATOR_SPEC.md`; this handoff refines Stage 5 only.

---

## 0. TL;DR

The pipeline now produces **correct surfaces and 100%-correct kicks** (ADTOF backbone). The remaining weakness is the **left-vs-right binary for hand hits**: the model alternates but the *phase* floats, and almost every hand hit reports ~50/50 confidence and gets flagged `?`.

There are **two root causes**, and you must fix both:

1. **No phase anchor** — nothing tells the model *which* hand leads, so RLRL alternation can start on either hand and flips mid-phrase.
2. **The confidence meter is unplugged from the evidence** — `_compute_marginal` scores L/R using *isolated emission only*. It ignores metric position, velocity, and neighbors. So even when the Viterbi *path* is right, the displayed confidence stays ~0.5 and the hit gets flagged.

The fix is a **3-part package** (do these together; they reinforce):
- **#0** Rewire the marginal so confidence reflects the same context the path uses.
- **#1** Metric anchor: bias the lead hand onto metrically strong positions (uses `quantized_pos`, currently computed and unused).
- **#2** Velocity/accent: bias loud hits to the lead hand, ghost notes to the off-hand (uses `velocity`, currently barely used).

Follow-ons (#3 side-aware reach, #4 ostinato role-lock, #5 vocabulary) are specified but **out of scope for the first pass** unless the package lands and you have budget.

**Non-negotiable guardrail:** every prior is a *soft, defeasible log-odds term*. None becomes a hard constraint. A genuinely even mono-surface roll with no accent and no metric cue **must still flag** — that is correct behavior, not a bug to fix.

---

## 1. Orientation

### Codebase map (only the files you'll touch)
| File | What's in it |
|---|---|
| `src/stage5_hmm.py` | **The core. All your work is here.** Custom Viterbi HMM over `(L_loc, R_loc, last_hand)` states. |
| `src/types.py` | `PipelineConfig` (add new tunable weights here), `KIT_GEOMETRY`, `NATURAL_HAND`, `Strike`/`AssignedStrike` (carry `velocity`, `quantized_pos`). |
| `src/stage2_beats.py` | Produces `quantized_pos` per strike and `beat_info` (meter, subdivision). You *read* its output; don't change it. |
| `src/pipeline.py` | Orchestrator. You won't change it. |

### Key facts about the current HMM (`src/stage5_hmm.py`)
- **State** = `(L_loc, R_loc, last_hand)`. `last_hand ∈ {None,'L','R'}` was added recently so alternation can break ties on a single surface. **Do not revert to a bare `(L_loc,R_loc)` or `{L,R}` state** — there is a spec comment forbidding it; keep it.
- **`assign_limbs(...)`** runs the forward Viterbi + backtrack, then a *separate* per-hit confidence pass via `_compute_marginal`.
- **`_log_transition(prev_state, new_state, assignment, slot, ioi_sec, reach_costs, config)`** — reach cost, continuity bonus, and a speed-gated recency alternation penalty. `ioi_sec` is the real inter-onset gap (a recent fix; the old code used an assumed 32nd-note rate and mis-gated — don't reintroduce that).
- **`_log_emission(assignment, slot, prev_state, local_bpm, attack_cache, natural, config, rudiment_prior)`** — natural-hand affinity (−0.4 for crossing), double-stroke acoustic signature (capped by `double_sig_weight`), capped rudiment prior. **This is where #1 and #2 go.**
- **`_compute_marginal(...)`** (≈ line 462) — currently scores each admissible limb by calling `_log_emission([(strike_idx, limb)], slot, ("idle","idle",None), local_bpm, ...)` and softmaxes. **This is the "unplugged" marginal — #0 fixes it.**
- **`_speed_fraction(ioi_sec, config)`** — 0 = slow (alternation default), 1 = at the single-hand floor (doubles expected). Already correct; reuse it.

### The honest-ceiling comments
`src/stage5_hmm.py` has two prominent comment blocks (the module docstring "HONEST CEILING" and the one above `_double_stroke_log_odds`). **Do not delete or weaken them.** They encode the design contract. Add to the design; don't fight it.

---

## 2. The problem, with evidence

Ground truth sticking for all three test clips is the **same pattern**:
```
R L K K R L K K R K L K R L R L
```
- `sample_1.mp4` is played on **snare + kick**.
- `sample_2.mp4` and `sample_3.mp4` are the **same sticking orchestrated around the kit** (toms involved), faster. ADTOF surfaces for 2/3: `S S K K T T K K S K T K T T T T`.

Current output (post ADTOF + state-expansion + speed-gate fix) — **this is your baseline to beat**:
```
sample_3  GT : R L K K R L K K R K L K R L R L
          got: R L K K R R K K L K R K L R L R     kicks 6/6 ✓, hands ~3/10, 10 flagged
```
Diagnosis from the data: kicks are perfect. Hands **alternate but flip phase**: the two consecutive toms (GT positions 5–6 = `R L`) come out `R R` — a spurious double — and that inversion propagates, flipping the parity of the rest of the phrase. And **every** hand hit reads `{R:~0.5, L:~0.5}` → flagged. Click-through in the UI confirms the underlying leaning exists (e.g. `R 60% / L 40%`) but is too weak to clear the `ambiguity_threshold` (0.65).

So: the model is one good phase-anchor away from the right answer, and one marginal-rewire away from being able to *show* it.

---

## 3. Why the marginal is the linchpin (read this twice)

`_compute_marginal` decides what confidence the user sees and whether a hit is flagged. Today it uses **isolated emission with a dummy `("idle","idle",None)` previous state**. For a plain snare/tom hit, emission is symmetric between L and R (natural-hand for snare is `None`; the double-sig term only fires when `surface == prev_loc`, which the dummy state defeats). Result: softmax over two equal numbers = exactly 0.5.

**Consequence you must internalize:** if you add #1 and #2 only to the *transition* or only to the Viterbi path, the path may improve but **confidence will still print 0.5 and everything stays flagged.** The marginal must see the same evidence.

The clean way to get this for free:
- **Put #1 (metric) and #2 (velocity) inside `_log_emission`.** Both are *per-hit* priors (they depend only on the hit's grid position and its loudness, not on the path). Because `_compute_marginal` already calls `_log_emission`, the marginal picks them up automatically — and they break the L/R symmetry honestly.
- **Then add neighbor context to the marginal (#0b):** thread the *previous hand hit's assigned limb* into `_compute_marginal` and add one alternation term (favor the limb that continues alternation from the previous hand, speed-gated exactly like `_log_transition`). This is what lets "the previous hand was R, so this is likely L" raise confidence.

Honest ceiling is preserved automatically: if metric is neutral, velocity is mid, and the neighbor is ambiguous, the terms are ~equal and softmax still returns ~0.5 → still flagged. You are only adding confidence where real evidence exists.

---

## 4. The plan (ROI order)

| # | Change | Fixes | Scope |
|---|---|---|---|
| **0** | Rewire `_compute_marginal` to read emission (now metric+velocity-aware) **and** neighbor alternation context | confidence stuck at 0.5 | **first pass** |
| **1** | Metric anchor in `_log_emission` (uses `quantized_pos`) | floating phase | **first pass** |
| **2** | Velocity/accent term in `_log_emission` (uses `velocity`) | phase + confidence | **first pass** |
| 3 | Side-aware (crossover) reach cost in `_log_transition` | wrong hand on toms | follow-on |
| 4 | Ostinato role-lock (ride/hat = lead, snare backbeat = off-hand) | grooves (not these samples) | follow-on |
| 5 | Style-aware vocabulary / RLL-around-toms | fine detail | follow-on (overfit risk) |

Implement **0+1+2 together**, verify, commit. Then consider 3–5.

---

## 5. Detailed specs — the first-pass package

### Config additions (`src/types.py`, `PipelineConfig`)
Add modest, tunable weights (keep them small so they can't outvote real acoustic evidence):
```python
metric_anchor_weight: float = 1.0     # #1 strength of metric phase bias
velocity_accent_weight: float = 1.0   # #2 strength of accent/ghost bias
marginal_neighbor_weight: float = 1.5 # #0b alternation context in the marginal
```
`handedness` already exists (`"right"`/`"left"`). Derive `lead = "R" if config.handedness == "right" else "L"`; `off` is the other.

### #1 — Metric anchor (in `_log_emission`)

**Principle:** in single-stroke playing the lead hand lands on metrically *strong* positions; the off-hand on weak positions. Anchors phase to the barline.

**Helper** (add near the emission model):
```python
def _metric_strength(quantized_pos):
    """Return (strength, lead_bias) for a beat position.
    strength ∈ [0,1] = metrical weight. lead_bias ∈ {+1,0,-1}:
    +1 → position favors the lead hand, -1 → favors the off-hand."""
    if quantized_pos is None:
        return 0.0, 0
    frac = quantized_pos - math.floor(quantized_pos)
    # tolerance for float wobble
    if abs(frac) < 0.05 or abs(frac - 1.0) < 0.05:      # on the beat
        return 1.0, +1
    if abs(frac - 0.5) < 0.05:                           # the "&" (eighth offbeat)
        return 0.6, +1
    if abs(frac - 0.25) < 0.05 or abs(frac - 0.75) < 0.05:  # "e"/"a" (16ths)
        return 0.4, -1
    # triplet / other subdivisions: weak, no strong lead claim
    return 0.2, 0
```
**Apply in `_log_emission`**, per `(i, limb)` in the assignment:
```python
strength, lead_bias = _metric_strength(strike.quantized_pos)
if lead_bias != 0:
    favored = lead if lead_bias > 0 else off
    sign = +1.0 if limb == favored else -1.0
    log_p += sign * config.metric_anchor_weight * strength
```
**Nuances (must keep):**
- *Defeasible:* this is soft; velocity (#2) or a rudiment can override it. Never gate or hard-constrain on it.
- *Subdivision-aware:* the table above is the 16th-note common case. If `beat_info` exposes the detected subdivision (`stage2_beats._detect_subdivision`) and it's triplet-based, the "&"/"e"/"a" mapping doesn't apply — fall back to `strength` from the beat hierarchy with `lead_bias = +1` only on integer beats. Don't force a 16th model onto triplet music.
- *Handedness:* `lead`/`off` already encode it.

### #2 — Velocity / accent (in `_log_emission`)

**Principle:** accents are played by the lead hand; ghost notes by the off-hand.

**Critical subtlety — use *relative* dynamics, not absolute.** `velocity` is normalized to the clip peak, but "accent vs ghost" is relative to the *local* dynamic level. Compute a reference (median hand-hit velocity for the clip) once and pass it in, then bias by deviation.

In `assign_limbs`, before the Viterbi loop, compute:
```python
hand_vels = [s.velocity for cd in constraint_dicts for s in cd["slot"].strikes if s.surface != "kick"]
vel_ref = float(np.median(hand_vels)) if hand_vels else 0.5
vel_spread = (float(np.percentile(hand_vels, 75) - np.percentile(hand_vels, 25)) or 0.1) if hand_vels else 0.1
```
Thread `vel_ref, vel_spread` into `_log_emission` (and `_compute_marginal`). Then per hit:
```python
accent = (strike.velocity - vel_ref) / vel_spread     # >0 accent, <0 ghost
accent = float(np.clip(accent, -2.0, 2.0))
# accent favors lead hand, ghost favors off-hand
sign = +1.0 if limb == lead else -1.0
log_p += sign * config.velocity_accent_weight * (accent / 2.0)   # /2 keeps it modest
```
**Nuances:** mid-level hits (accent ≈ 0) get ≈ no bias → no false confidence. Clip to avoid one freak-loud hit dominating. This is the term most likely to lift confidence honestly because it is direct acoustic evidence.

### #0 — Rewire the marginal (`_compute_marginal`)

Two parts:

**#0a (free):** because #1 and #2 now live in `_log_emission`, and `_compute_marginal` already calls `_log_emission`, the marginal will incorporate metric + velocity automatically — **as long as you pass the real per-hit context** (it already passes the real `slot`, so `quantized_pos`/`velocity` are available; just also pass `vel_ref/vel_spread`). Verify the dummy `("idle","idle",None)` prev-state doesn't suppress your new terms — #1 and #2 don't depend on prev_state, so they survive. Good.

**#0b (the neighbor term):** give the marginal the previous hand hit's limb and reward alternation:
1. In `assign_limbs`' result-building loop, track the previous *hand* hit's assigned limb (`prev_hand_limb`) as you iterate `path_assignments` in time order (skip feet).
2. Pass `prev_hand_limb` and the slot's `ioi_sec` into `_compute_marginal`.
3. Inside, after computing the per-limb emission scores, add a neighbor term **mirroring `_log_transition`'s gate**:
```python
if prev_hand_limb is not None and _speed_fraction(ioi_sec, config) < config.double_prior_speed:
    for limb in adm:
        if limb != prev_hand_limb:          # alternation continues
            scores[limb] += config.marginal_neighbor_weight
```
Then softmax as today. This is what makes "prev was R → this is likely L" show as, say, 0.75 instead of 0.5.

**Sanity:** confirm that on a truly ambiguous hit (neutral metric, mid velocity, and — if you want to test the floor — set neighbor weight to 0) the marginal still returns ≈0.5. The honest ceiling must survive.

---

## 6. Follow-on leads (spec only — not first pass)

- **#3 Side-aware reach (`_log_transition` / `_precompute_reach_costs`):** today reach is symmetric Euclidean. Make it asymmetric: a hand crossing past kit center (L reaching a right-of-center surface, or R reaching left-of-center) pays a `crossover_penalty`. Use the X-coordinate sign in `KIT_GEOMETRY[setup]`. Biases which hand takes which tom in around-the-kit fills.
- **#4 Ostinato role-lock:** detect a surface repeating steadily on the grid (ride/hi-hat time-keeping) → lock that hand to the lead; snare on 2&4 → off-hand. Near-deterministic for grooves; little effect on these solo-chops samples but high value generally. Implement as a strong (not infinite) emission bias.
- **#5 Vocabulary / RLL-around-toms:** strengthen the Tier-1 rudiment prior and add the "groups of three moving across drums → RLL" tendency. **Highest overfit risk** — keep capped, and never enable the Tier-2 self-prior (it has a documented feedback-loop guard for a reason).

---

## 7. Drumming grounding (why these priors are legitimate)

These are not ML heuristics; they are documented drummer conventions:
- **Metric anchor:** "in natural sticking the strong pulse falls under the right/lead hand." — [The Drumslingers](http://thedrumslingers.blogspot.com/2021/10/sticking-methods-rudimental.html). (Note: alternating sticking *sometimes* puts the strong pulse on the left → keep it soft.)
- **Accents → lead hand, ghosts → off-hand:** [DRUM! Magazine "Your Left Hand Sucks"](https://drummagazine.com/your-left-hand-sucks-day-1/), [C&C Drums on accents/ghost notes](https://www.candccustomdrums.com/improving-dynamics-accents-and-ghost-notes/).
- **Groove roles (R = hats/ride, L = snare backbeat):** [eMastered basic beats](https://emastered.com/blog/basic-drum-beats), [The Drum Ninja](https://thedrumninja.com/common-drum-beats/).
- **Crossover economy around toms:** [Drummer Cafe on crossovers](https://www.drummercafe.com/education/articles/using-crossovers-in-drum-fills.html), [onlinedrummer "Moving Hands"](https://www.onlinedrummer.com/blogs/drum-lessons/moving-hands-simple-drum-fill-technique).
- **Linear/gospel fills use a small sticking vocabulary; "groups of three → RLL":** [drumstheword](https://www.drumstheword.com/free-drum-lesson-fa-de-la-dump-famous-classic-triplet-drum-fill-lick/), [drumspy paradiddles](https://drumspy.com/academy/how-to-play-paradiddles/).

---

## 8. Guardrails — do NOT

1. **Do not fabricate confidence.** Truly ambiguous hits (even mono-surface roll, no accent, neutral metric) must still flag. Verify the floor explicitly.
2. **Do not make any prior a hard constraint.** All soft log-odds. The kick→Foot rule is the only hard limb rule and it already works — don't touch it.
3. **Do not revert the `(L_loc, R_loc, last_hand)` state** or delete the HONEST-CEILING / "no bare {L,R} state" comments.
4. **Do not raise `double_sig_weight` above 1.0**, and do not enable the Tier-2 self-prior (`self_prior_enabled`).
5. **Do not overfit to the three samples.** They are a smoke test, not a training set. If a change only helps by encoding this exact phrase, it's wrong.
6. **Do not change Stage 0–4, the ADTOF integration, or `quantized_pos` generation.** Surfaces and kicks are correct; leave them.

---

## 9. Verification protocol

### Environment (Windows; required every run)
- Use the venv interpreter: `.venv/Scripts/python.exe`.
- Set env: `PYTHONUTF8=1` and `TF_USE_LEGACY_KERAS=1` (the latter is also set at import by `stage1_onsets.py`, but set it anyway).
- ADTOF model load is slow (~15–30s). **Load it once and reuse across all three samples** (pass `adtof_model=` into `pipeline.run`).

### Probe script (create as a temp file, delete after — don't commit it)
```python
import os; os.environ["TF_USE_LEGACY_KERAS"]="1"
import json, src._madmom_compat
from src.stage1_onsets import load_adtof_model
from src.pipeline import run
model = load_adtof_model()
GT = "R L K K R L K K R K L K R L R L"
for s in ["samples/sample_1.mp4","samples/sample_2.mp4","samples/sample_3.mp4"]:
    d = json.loads(run(s, adtof_model=model, separate=False).json_output)
    seq = " ".join("K" if h["limb"]=="F" else h["limb"] for h in d["hits"])
    hand = [h for h in d["hits"] if h["limb"]!="F"]
    flagged = sum(1 for h in d["hits"] if h["flagged_ambiguous"])
    avg_conf = sum(h["confidence"] for h in hand)/max(len(hand),1)
    print(f"\n{s}  hits={len(d['hits'])} flagged={flagged} avg_hand_conf={avg_conf:.2f}")
    print(" GT :", GT)
    print(" got:", seq)
```
Run: `PYTHONUTF8=1 TF_USE_LEGACY_KERAS=1 .venv/Scripts/python.exe probe.py` (filter TF noise with `2>&1 | grep -vE "WARNING|tensorflow|oneDNN|I0000|absl|deprecated"`).

### Success criteria
1. **Reproduce the baseline first** (numbers in §2) before changing anything, so you can measure your delta.
2. **Kicks stay 6/6 correct** on every sample (positions 3,4,7,8,10,12). Any regression here means you broke something upstream — stop.
3. **Phase locks:** the spurious tom double (`…R R…` at GT positions 5–6) should resolve toward alternation; the hand sequence should stop inverting mid-phrase.
4. **Confidence rises where evidence exists:** `avg_hand_conf` should climb above the baseline and `flagged` should drop — *but not to zero.* Some hits should remain flagged. If everything becomes confident, you've broken the honest ceiling — back off the weights.
5. **No overfitting:** the same weights should help all three samples (and not depend on the exact GT).
6. Report a before/after table (hits, flagged, avg_hand_conf, hand-accuracy-vs-GT excluding flagged) for all three.

### Tuning note
Start with the default weights in §5. If confidence over-saturates, lower `metric_anchor_weight` / `velocity_accent_weight` / `marginal_neighbor_weight`. If nothing moves, you likely added the priors to the path but not to `_log_emission`/the marginal — re-check §3.

---

## 10. Suggested commit sequence
1. Add config fields + `_metric_strength` helper + the velocity reference computation. (no behavior change yet)
2. Wire #1 and #2 into `_log_emission`. Verify path improves, note confidence may still be flat.
3. Wire #0b (neighbor term) into `_compute_marginal`; thread `prev_hand_limb`, `ioi_sec`, `vel_ref/vel_spread`. Verify confidence rises.
4. Tune weights on all three samples; confirm the honest-ceiling floor still flags ambiguous hits.
5. One commit with a clear message summarizing the three priors and the measured before/after.

When done, hand back a short report: the before/after table, the final weights, and any hit that *should* be confident but isn't (or vice-versa) so we can decide on follow-ons #3–#5.
