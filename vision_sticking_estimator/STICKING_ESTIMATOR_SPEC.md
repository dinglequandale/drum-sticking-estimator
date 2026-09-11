# Drum Sticking Estimator — Build Specification

**Goal.** Given a fixed-camera drum video (or its audio), produce a per-hit limb assignment — Left hand / Right hand / Foot — for full-kit grooves and fills, plus a list of recurring sticking patterns, with explicit confidence and explicit flags on hits that audio cannot resolve.

**Non-goal.** This is **not** a perfect transcriber. It is an *estimator* that outputs probabilities and flags its own ambiguity. Any step that would force a confident guess where the audio carries no signal is a bug, not a feature. The honest failure mode (a flagged "ambiguous" hit) is always preferred over a confident wrong one.

**Approach.** Audio-only. Pose/video was considered and rejected: pose estimation is too noisy for ghost notes and the kit self-occludes the hands during exactly the fast passages we care about. We extract everything from the audio and infer limb assignment from *context* (neighboring hits, surface, velocity, tempo, repetition), not from any single hit in isolation.

---

## Guiding principle: do not reinvent solved problems

Onset detection, drum classification, beat tracking, and source separation are **solved**. Use the proven libraries below. Write original code only for the **limb-assignment layer (Stage 5)** and the **self-similarity layer (Stage 6)** — that is where the actual novel work lives. Everything before Stage 5 is glue around existing tools.

| Task | Use this | Notes |
|------|----------|-------|
| Source separation | **Demucs** (`htdemucs` model) | Isolates a drum stem from mixed audio. Skip if input is already drum-only. |
| Onset + drum classification + velocity | **ADTOF** (CRNN, 5-class: kick / snare / toms / hi-hat / cymbals, with velocity) | Actively maintained (2025), non-synthetic training data, sub-10ms target alignment. This is the primary backbone. |
| Beat / downbeat / tempo / meter | **madmom** (`DBNDownBeatTrackingProcessor`) | Also provides the metrical grid for quantization. |
| Fallback onset detection | **librosa** (`onset.onset_detect`, spectral flux) | Sanity check / backup if ADTOF onsets look sparse. |
| Sequence decoding (HMM/Viterbi) | **madmom HMM module** or **pomegranate** | Don't hand-roll Viterbi unless the state space needs custom structure (it does — see Stage 5). |
| Eval metrics | **mir_eval** | Standard onset F-measure; reuse for assignment accuracy. |

All are Python, pip-installable, permissive licenses (BSD/MIT). Pin versions in `requirements.txt`; madmom + ADTOF have known dependency friction (numpy / tensorflow / cython), so build the environment first and freeze it before writing any logic.

---

## Pipeline overview

```
[audio in]
   |
(0) ingest + (optional) Demucs drum-stem separation
   |
(1) onset + drum-surface + velocity   ->  ADTOF        (event list)
   |
(2) beat / tempo / grid               ->  madmom       (metrical grid)
   |
(3) event consolidation               ->  CUSTOM       (chords, not just onsets)
   |
(4) hard physical constraints         ->  CUSTOM       (prune impossible assignments)
   |
(5) probabilistic limb assignment     ->  CUSTOM HMM   (the core; L/R/foot per event)
   |
(6) self-similarity propagation       ->  CUSTOM       (easy instances teach hard ones)
   |
(7) pattern extraction + output       ->  CUSTOM       (recurring stickings + confidence + flags)
```

---

## Stage 0 — Ingest

- Accept audio file or a video file (extract audio track with `ffmpeg` if video).
- Mono, resample to whatever ADTOF/madmom expect (typically 44.1 kHz).
- If the recording is mixed (music behind the kit), run **Demucs** and keep the `drums` stem. If it's already a clean kit recording, skip — separation adds artifacts that hurt onset precision.
- Output: a normalized WAV path + a flag `was_separated`.

## Stage 1 — Onset, surface, velocity (ADTOF)

- Run ADTOF to get a list of events: `{time, surface ∈ {kick, snare, tom, hihat, cymbal}, velocity}`.
- **Keep surface internally even though it's not in the final output.** Surface is the single strongest non-pose cue for hand assignment (surface changes force hand changes; reach cost depends on which drum). Discarding it would gut the model. The user doesn't want surface *displayed*; that is not the same as not *using* it.
- Velocity is required downstream (rebound decay is the RR-vs-RL tell).
- Validate against a librosa spectral-flux onset pass; if ADTOF misses a chunk of obvious onsets, log a warning. Don't silently trust one detector.

## Stage 2 — Beat, tempo, grid (madmom)

- Run madmom downbeat tracking to get beats, downbeats, tempo (possibly time-varying), and meter.
- Quantize each event to the nearest grid subdivision (16ths, triplets — detect which). Store both raw time and quantized position.
- **Tempo is a prior, not just metadata.** Local tempo sets where on the speed axis each passage sits, which gates the hard constraints (Stage 4) and tilts the singles-vs-doubles prior (Stage 5). Pass local tempo downstream per-event.
- Note: the kick/snare backbone from Stage 1 *helps* beat tracking. If madmom struggles, feed it the separated drum stem rather than the raw mix.

## Stage 3 — Event consolidation (chords vs. sequence) — CUSTOM

**This is a correction to the naive plan and must not be skipped.** An onset is not a limb. Two surfaces struck within a few ms (e.g., snare + crash on a downbeat) are a **chord** — two limbs at once — not two sequential single-limb events. If the assignment layer treats them as a sequence, the whole Viterbi state space is malformed.

- Cluster events whose onset times fall within a coincidence window (tunable, start ~15–25 ms) into a single **time-slot** that may contain multiple simultaneous strikes.
- A time-slot therefore holds 1–4 strikes (e.g., {kick} or {snare} or {snare+crash} or {kick+snare+crash}).
- The downstream sequence is a sequence of **time-slots**, and assignment operates on the strikes within each slot under the constraint that one limb fills at most one strike per slot.
- Output: ordered list of time-slots, each a set of `{surface, velocity, raw_time}`.

## Stage 4 — Hard physical constraints — CUSTOM

Apply inviolable physics to prune the hypothesis space *before* probabilistic ranking. These are near-certain; lean on them hard.

1. **Single-hand stroke-rate ceiling.** One hand cannot strike a surface twice faster than a physical minimum inter-onset interval (set a conservative ceiling, e.g. corresponding to ~14–16 Hz absolute, but make it a tunable parameter). Two same-surface strikes closer than this *must* be two different hands — **unless** the IOI sits in the narrow band characteristic of a controlled double/buzz (handle that as a flagged special case, see Stage 5).
2. **Spatial exclusivity.** A single limb cannot occupy two surfaces in the same time-slot. Within a chord, distinct strikes require distinct limbs.
3. **Foot is its own limb.** Kick (and hi-hat pedal, if separable) are assigned to feet and removed from the L/R hand problem entirely. This reduces the hand problem to **binary L/R** over hand strikes — a major simplification, so do it explicitly and early.
4. **Limb continuity.** A limb committed to a strike at time *t* is unavailable for another strike until a minimum recovery interval later. Encode as forbidden transitions.

Output: for each strike, a *reduced set* of admissible limb labels (often the constraint already collapses it to one).

## Stage 5 — Probabilistic limb assignment (the core) — CUSTOM HMM

This is the heart of the system. Frame it as finding the most probable hidden limb-state sequence given the observed time-slot sequence. Decode with Viterbi over the constraint-pruned space from Stage 4.

**State design — DO NOT use a bare {L, R} state.** The correct hidden state must encode *where each hand currently is*, because continuity and reach cost depend on hand position:

```
state = (left_hand_location, right_hand_location)
```

where location ∈ the kit's surfaces (plus "idle"). This is larger but it is what makes the continuity and reach priors actually computable — the cost of the next assignment depends on where each hand sits now. Prune aggressively with Stage 4 so the state space stays tractable.

**Emission / transition factors (the soft priors that rank survivors):**

- **Alternation default.** A run of same-surface strikes at moderate tempo defaults to hand-to-hand alternation (RLRL).
- **Doubles favored near the speed wall.** As IOI shrinks toward the single-hand ceiling, the prior shifts from singles toward doubles/paradiddle structures, because that's when human players physically switch technique. Tempo (Stage 2) drives this.
- **Double-stroke acoustic signature → double.** The second note of a natural double is not merely *quieter* — it is structurally different: the first stroke is wrist-driven (higher velocity, sharp attack, "staccato"), the second is a finger/rebound stroke (lower velocity, softer attack, "mushier"/legato, slightly different spectral character). So the emission model should compare the **pair's attack envelopes and timbre**, not just amplitude: a sharp-then-soft, articulate-then-legato pair signals RR/LL; two near-identical high-velocity "machine-gun" strikes signal RLRL singles. Single strokes also tend to read as more powerful/consistent (better suited to toms), doubles as smoother and easier at high speed on the snare — so surface interacts with this prior (weight the double-signature feature more on snare passages, less on tom power-hits).
  - **Critical limitation — do not over-trust this on hard cases.** This signature is precisely what a skilled player drills for years to *erase*: the explicit goal of a well-executed double-stroke roll is to make the rebound note as even as the wrist stroke, i.e. to sound like singles. So this feature is strongest on imperfect/amateur doubles (where you least need help) and weakest on fast, even, virtuoso passages (where you most do). It **raises the floor; it does not move the honest ceiling below.** Weight it modestly and let it be outvoted by evidence. Never let a clean-sounding double's *absence* of signature be read as positive evidence of singles — absence of the tell is not proof of alternation, it's just silence.
- **Reach / ergonomic cost.** Transition cost between hand locations encodes kit geometry: a hand crossing the kit is costlier than staying put. **This cost model is player-specific** (handedness, open vs. crossed setup). Expose it as a parameter set; default to right-handed crossed, but allow override (see Calibration below).
- **Hand continuity.** Penalize physically implausible jumps; reward a hand staying on or near its previous surface.

**Output per strike:** a probability distribution over limb labels (not a hard label), carrying the Viterbi best path *and* the marginal confidence. Where the distribution is near-uniform (e.g. an even double-stroke with no disambiguating context), that's the irreducible residue — **keep the uncertainty, do not collapse it to a guess.**

**Honest ceiling (state this in code comments so nobody "fixes" it later):** a genuinely even double-stroke from a skilled player, with no surrounding surface change or tempo cue, sits near a coin flip from audio alone. No emission model escapes this. All real accuracy comes from *context*, which is why Stage 6 exists.

**Optional calibration hook.** Since the user says videos are fixed-cam, optionally allow a one-time per-video calibration: a human labels handedness + cross-style (and optionally a few unambiguous hits). This anchors the reach-cost model to the actual player. Build the parameter interface for it even if the auto-default is used most of the time — a fixed geometry prior will confidently mislabel left-handed or open-handed players otherwise.

### Stage 5b — Sticking-vocabulary priors (use with discipline)

A prior over *which sticking sequences are idiomatic* can help break ties — but only as a **soft transition prior that evidence can override**, never as a template the decode snaps to. A hard "bank of known patterns" is forbidden: it manufactures false confidence by seeing common rudiments everywhere and mislabeling exactly the off-vocabulary playing this tool exists to capture. Two tiers, with very different risk profiles:

**Tier 1 — Global rudiment prior (DEFAULT ON, safe).**
- Encode the standard rudiments (paradiddles, double-stroke roll, flam taps, Swiss triplets, etc.) as sticking-token sequences; build an n-gram transition model over the tokens (RLRR is common, RLLR is rare). Smooth so no transition has zero probability.
- Inject as a transition prior into the Stage 5 HMM, **capped** so its log-weight can only break ties when audio evidence is near-silent. A real velocity/surface/speed signal must always outvote it.
- This is externally anchored and cheap (hours of work). It is safe precisely because it cannot learn the model's own biases.

**Tier 2 — Per-video self-prior (DEFAULT OFF, experimental, three hard guards).**
The most player-matched "bank" is the drummer in front of you — learn *this player's* sticking grammar from the video and feed it back. This is powerful but dangerous: it can become a **confirmation loop** (self-training collapse) where the model's own guesses are laundered back as evidence, inflating confidence without improving accuracy. Your concern that "a drummer won't repeat enough" is real but it is the *benign* failure (a sparse prior is merely weak). The real hazard is contamination. Therefore, admit this prior ONLY behind all three guards:

1. **Source restriction — train ONLY on `source: constraint` hits.** The self-prior may learn exclusively from hits that a *hard physical fact* forced (speed-gate proved two hands; surface change forced a hand switch) — see the `source` tag in Stage 7's output schema. Hits labeled `source: inference` (the soft-prior decode) are **forbidden** as training data, because feeding inference hits back into the prior is the exact laundering path that creates the loop. This single rule is what breaks the feedback: the prior can be taught by physics, never by its own opinions. Implement it so it is structurally impossible to wire inference hits into the prior's training set.
2. **Hard cap — prior breaks ties only.** Bound its contribution like Tier 1: decisive only in a genuine evidence vacuum, always outvoted by real acoustic evidence.
3. **Dual-run eval guard (see Evaluation).** Every test clip is scored twice — prior ON and prior OFF — reporting accuracy *and flag-rate* for both. The prior is admitted **only if it raises accuracy on a held-out off-vocabulary (deliberately non-repetitive) clip.** If it shrinks the flag-rate (raises confidence) without raising accuracy, it is corrupting and must be cut, regardless of how good the aggregate number looks.

Net guidance: ship Tier 1. Treat Tier 2 as off-by-default research that must earn its place through guard 3 on off-vocabulary material. Do not lean on repetition as a primary mechanism — the user is correct that real playing isn't that repetitive, so Stage 6 and Tier 2 are *secondary* lifts, not the backbone. The backbone remains audio evidence + hard constraints.

## Stage 6 — Self-similarity propagation — CUSTOM

**Secondary lift, not the backbone.** Drummers sometimes repeat figures; when they do, resolve the *unambiguous* instances first, then propagate their sticking to the *ambiguous* repeats. But real playing is not repetitive enough for this to carry the system — treat it as an opportunistic bonus on top of the audio-evidence backbone, not a primary mechanism. (Earlier drafts overcredited this stage; it helps when repetition exists and does nothing when it doesn't.)

- **Match on rhythm + surface skeleton ONLY.** The similarity metric must be defined over (quantized inter-onset pattern, surface sequence) and must **exclude the sticking itself** — otherwise the propagation is circular (you'd be matching on the thing you're trying to infer). An RR version and an RL version of the same figure have *identical* skeletons; that's exactly why the match is valid and the sticking transfer is informative.
- Find recurring figures via subsequence matching over the time-slot stream (suffix-array / n-gram clustering on the skeleton tokens).
- For each cluster of matching figures, identify the instance with the **highest Stage-5 confidence** (often one that appears at slower tempo, or one forced by a surface change). Treat its sticking as the cluster template.
- Propagate the template sticking to low-confidence instances in the same cluster, *raising* their confidence accordingly — but record that the assignment came from propagation, not direct inference.
- Iterate to a fixed point (a newly resolved instance may anchor another cluster).

## Stage 7 — Pattern extraction + output — CUSTOM

- Run motif discovery over the final limb-token sequence to surface the recurring stickings (e.g. "RLRR LRLL paradiddle appears at bars 5, 9, 13").
- Output a structured object per hit: `{time, quantized_position, surface (internal), limb, confidence, source ∈ {constraint, inference, propagation}, flagged_ambiguous: bool}`.
- Final user-facing rendering: limb sequence (L / R / F) aligned to the metrical grid, recurring patterns called out, and **every irreducibly ambiguous hit explicitly marked** rather than silently guessed. A drummer can apply their own judgment to flagged spots — that's the honest, useful product.
- Suggested output formats: a JSON event stream (source of truth) plus a human-readable grid (text tab or a simple piano-roll style render).

---

## Evaluation harness — BUILD THIS EARLY, NOT LAST

Without ground truth, every tuning decision is blind. This is non-optional.

- Hand-annotate a **small** test set: 5–10 short clips (8–16 bars each) spanning easy (slow groove), medium (moderate fill), and hard (fast even double-strokes). Label limb per hit by watching frame-by-frame where the camera *does* show the hands — use the easy-to-see passages as truth even if the model won't get pose.
- Metrics:
  - **Onset F-measure** (via mir_eval) — validates Stages 1–3 independently of assignment.
  - **Limb-assignment accuracy** on hits the model attempts (excludes flagged-ambiguous).
  - **Flag precision/recall** — when the model says "ambiguous," is it actually a hard case? A model that flags everything is useless; a model that flags nothing is dishonest. This metric guards the core design promise.
- Report assignment accuracy *separately* for easy / medium / hard tiers. A single aggregate number hides the only thing that matters (does it degrade gracefully on hard cases or fail silently).
- **Include at least one deliberately off-vocabulary clip** — a passage of non-repetitive, non-rudimental playing. This is the clip that catches a sticking prior (Stage 5b) lying: a prior that helps on idiomatic clips but hurts here is biasing toward "plausible" rather than "correct."
- **Dual-run guard for Stage 5b priors.** Score every clip twice — sticking-prior ON and OFF — reporting accuracy **and flag-rate** for each. Decision rule: a prior is admitted only if it *raises accuracy*, especially on the off-vocabulary clip. If it *shrinks the flag-rate without raising accuracy*, it is manufacturing false confidence and must be cut. Watch confidence-vs-accuracy, never accuracy alone.

---

## Build order (suggested)

1. Environment: pin Demucs + ADTOF + madmom + librosa + mir_eval; freeze `requirements.txt`. Resolve dependency friction *first*.
2. Stages 0–2 end to end on one clean clip; eyeball onsets/surfaces/tempo against the audio.
3. Build the eval harness + annotate 2–3 clips. Get onset F-measure passing before touching assignment.
4. Stages 3–4 (consolidation + hard constraints). Verify chords and speed-gates behave on hand-checked examples.
5. Stage 5 HMM with the (L-loc, R-loc) state and a *minimal* prior set (alternation + continuity only). Measure. Add the remaining priors one at a time, measuring each — don't add all priors blind. The double-stroke acoustic-signature feature and the reach/tempo priors each get their own measured A/B.
6. Stage 5b Tier 1 (global rudiment prior), capped. Measure on/off. Only then consider Tier 2 (per-video self-prior) behind its three guards — and only keep it if the dual-run eval earns it on the off-vocabulary clip.
7. Stage 6 self-similarity. A secondary lift, not the backbone — real playing isn't repetitive enough to lean on. If it doesn't help the medium tier, the matcher or confidence ranking is wrong.
8. Stage 7 output + pattern render.

## Open design decisions to resolve during build

- Coincidence window for chord clustering (Stage 3) — tune on real simultaneous hits.
- Single-hand IOI ceiling and the double/buzz band (Stage 4) — measure from fast single-surface passages in the test set rather than guessing.
- State-space size vs. tractability (Stage 5) — if (L-loc, R-loc) is too large, restrict locations to the surfaces actually present in a given video.
- Whether to ship per-video calibration in v1 or default-only (Stage 5).

---

### One-line reminders to keep taped to the monitor
- Surface is used internally, never shown. Using ≠ displaying.
- Onsets are not limbs; chords are not sequences (Stage 3).
- The HMM state is (left-loc, right-loc), not {L,R}.
- Self-similarity matches on skeleton only, never on sticking.
- Flag the residue. A flagged hit beats a confident wrong one, every time.
