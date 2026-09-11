# Design: Vision-First Refactor — Deterministic Sticking Recovery

**Audience:** the implementing agent. Read this whole document before editing anything.
**Normativity:** §§0–10 give the design and its rationale; **§11 is the implementation
appendix** — exact schemas, numeric defaults, code touch points, and stop conditions.
Where §11 gives a concrete value or procedure, §11 wins. Do not substitute your own
defaults for §11's; if a §11 value proves wrong, that is a measured Phase-gate finding to
report, not a silent change.
**Status:** this is the build contract for the final version of the vision sticking estimator.
It **supersedes the "Approach: audio-only, pose rejected" paragraph** of
`STICKING_ESTIMATOR_SPEC.md` and the priority ordering of `HANDOFF_LR_PRIORS.md`. Everything
else in those documents (stage contracts, honest-ceiling discipline, guardrails, eval
philosophy) remains in force and is referenced, not repeated, here.

---

## 0. The reframe (why this refactor exists)

The project's goal is unchanged: recover the sticking a drummer **actually used** from a
fixed-camera drum video. What changes is the **evidence hierarchy**:

| | Old (current code) | New (this contract) |
|---|---|---|
| Which-hand evidence | Audio-derived soft priors (metric anchor, accent, alternation, grammar) | **Vision kinematics, per hit** — deterministic where the camera can see |
| Audio's role | Everything | Event backbone (onset / surface / velocity / grid) + **fallback inference** for hits vision cannot resolve |
| Output character | Probabilistic everywhere, ~50% confidences, heavy flagging | **Deterministic anchors** where vision is confident; constrained interpolation between anchors; honest flags only on the true residue |

Two facts make this the right split:

1. **The feet are already solved.** ADTOF assigns kicks with 100% accuracy on the test
   clips (`HANDOFF_LR_PRIORS.md` §0). Removing feet leaves a **binary L/R problem** over
   hand hits — the cheapest possible question to ask a camera.
2. **Audio-only phase is structurally weak.** The documented failure mode
   (`HANDOFF_LR_PRIORS.md` §2) is not noise, it's structure: alternation is recoverable
   from audio but the **absolute phase** (which hand leads) floats, because audio is
   nearly symmetric under L↔R. The ergonomic sibling project independently converged on
   the same conclusion ("relative pattern confident, absolute hand ~0.5 is *honest*").
   Vision is the only evidence stream that breaks the L↔R symmetry directly. Even sparse
   vision resolves it globally — see §2.

**Non-goal (unchanged):** this is still an estimator, not an oracle. When the camera
cannot see (occlusion, motion blur beyond the frame rate, hands out of frame), the correct
output is a probabilistic fallback assignment with a flag — never a fabricated certainty.
"Deterministic" means *deterministic where the evidence is deterministic*, not everywhere.

---

## 1. Why the first vision attempt failed — and why this design is different

The 2026-06-22 probe (`vision_probe.py`) ran MediaPipe **HandLandmarker** on **single
frames** sampled at onsets, voting "lowest hand = striker." Result on real footage: any
hand detected 53% of onsets, **both hands 0%**, detections mislocated. Three root causes
were identified, and each dictates a design decision here:

| Failure cause | Design answer |
|---|---|
| **Stick decoupling** — the stick tip strikes 30–40 cm from the tracked hand, so hand *position* ≠ impact point | Score **motion, not position**: the striking hand is the wrist whose downward-velocity peak and rebound are **phase-locked to the audio onset** (the banked kinematic-onset insight). Position relative to the struck drum is a secondary, coarse cue. |
| **Occlusion / side camera angle** — the far hand invisible in 100% of probed frames | (a) Use **whole-body pose**, not hand-landmark models: a body-pose model localizes a wrist from arm/shoulder context even when the hand itself is hidden or blurred. (b) **Track continuously** across all frames, not per-onset snapshots — a wrist occluded at the strike instant is usually visible frames earlier/later, and a track can bridge the gap. (c) When a track is genuinely gone, **abstain** — that hit falls to audio fallback. |
| **Small, fast, motion-blurred hands at 30 fps** — HandLandmarker is trained on large, front-facing hands | Same two answers: body-pose backbones are trained on whole persons at distance (sports/surveillance footage), and temporal tracking integrates over blur. Plus an explicit **structural abstention** rule: when inter-onset interval < ~3 frames, vision does not vote (§4.5). |

**The prior probe tested the wrong tool in the wrong mode.** It does *not* condemn vision;
it condemns per-frame hand-landmarking. The go/no-go probe in Phase 1 (§7) re-tests with
body pose + continuous tracking before any large build — if that also fails on curated
footage, the fallback position is defined in §8.

---

## 2. The core architectural idea: sparse anchors + constrained interpolation

Vision does **not** need to resolve every hit. It needs to resolve *some* hits with
near-certainty. Here is why that is sufficient:

- Between two anchored hits, the HMM's freedom is tiny. Hard constraints (Stage 4
  speed-gate, chords) plus alternation/doubles structure mean that pinning the hand at
  hit *i* and hit *i+k* usually determines — or nearly determines — everything between.
  A single anchor **pins the parity of an entire alternating run**. The documented
  sample_3 failure (one spurious `R R` inverts the rest of the phrase) is exactly a
  parity error that any one anchor in the run would have prevented.
- Anchors are *most available* exactly where audio is *most blind*: slow-to-moderate
  passages with clear arm motion are trivial for the camera and ambiguous for audio
  (mono-surface alternation). Conversely audio's hard constraints are strongest at
  extreme speed, where vision is blind. The two evidence streams fail in opposite
  regimes — this remains the fusion thesis, now with vision as the senior partner.
- Therefore the target metric is **anchor precision** (a wrong anchor poisons the whole
  run between anchors), with coverage second. Design target: **anchor precision ≥ 0.98,
  coverage wherever it lands** (30% coverage with clean interpolation beats 90% coverage
  at 0.9 precision). This is THE number the eval guards (§6).

Pipeline shape:

```
                    [video in]
                        |
   AUDIO (unchanged)    |          VISION (new, src/vision/)
   Stage 0 ingest ------+--> V0 frame stream + cut/segment detection
   Stage 1 ADTOF             V1 per-frame body pose (wrists via arm chain)
   Stage 2 madmom grid       V2 wrist tracks: smooth, differentiate, identity
   Stage 3 chords            V3 kit-region calibration (surface -> image region)
   Stage 4 constraints       V4 per-onset kinematic strike scorer  <- onsets from Stage 3
                             V5 calibrated P(L/R) + abstention per hand hit
                        |
        Stage 5 FUSION DECODE (refactored):
          - vision anchors (calibrated conf >= anchor threshold) CLAMP the hit
          - softer vision votes enter as capped emission log-odds
          - audio HMM (alternation, reach, doubles, grammar) fills between anchors
        Stage 6 self-similarity: propagate from vision-anchored templates
        Stage 7 output: source in {constraint, vision, inference, propagation}
```

Audio Stages 0–4 are **not modified** (they are also shared with the ergonomic project —
CLAUDE.md forbids divergence without reason). All new code lives in `src/vision/` plus a
bounded refactor inside Stage 5.

---

## 3. Vision pipeline specification (`src/vision/`)

### V0 — Frame stream + segmentation (`v0_segments.py`)
- Decode video with OpenCV (`cv2.VideoCapture`); record fps, frame count, resolution.
- **Scene-cut detection** (found footage has cuts/zooms): mean-frame-difference threshold
  is sufficient; on a cut, close the current segment and start a new one. Kit calibration
  (V3) and wrist tracks (V2) are **per-segment** — never bridge a cut.
- Output: list of segments `{start_frame, end_frame}` + a frame iterator.

### V1 — Per-frame body pose (`v1_pose.py`)
- **Whole-body 2D pose**, one drummer (largest/most central person if several).
  Keypoints needed: both wrists, elbows, shoulders (+ pose confidence per keypoint).
- **Backbone is a Phase-1 bake-off decision, not a guess.** Candidates, in probe order:
  1. **MediaPipe Pose Landmarker** (Tasks API, `RunningMode.VIDEO` — the video mode does
     internal temporal tracking). Zero new dependencies (mediapipe already pinned).
  2. **rtmlib** (RTMPose via ONNX Runtime) — near-SOTA wrist accuracy, pip-installable on
     Windows without the mmcv toolchain, no torch/TF conflict.
  3. **Ultralytics YOLO-pose** — torch is already present (Demucs). Fastest to try if 1–2
     disappoint.
  Verify any addition against the root env pins (`numpy<2`, TF 2.x) before installing;
  install into the shared `.venv`.
- Wrap the chosen backbone behind a single interface:
  `pose(frame) -> {left_wrist: (x, y, conf), right_wrist: ..., elbows..., shoulders...}`
  so the backbone is swappable without touching V2+.
- **Anatomical identity, not screen position.** "Left wrist" means the drummer's left
  arm as resolved by the shoulder→elbow→wrist chain of the pose skeleton. The old probe's
  `wrist_x < 0.5 → L` heuristic is forbidden — it breaks on every crossover and cross-grip
  hi-hat pattern, which is precisely where we need vision most.

### V2 — Wrist tracks (`v2_tracks.py`)
- Build per-segment time series for each wrist: raw keypoints → outlier rejection
  (teleport gate: displacement > plausible max between consecutive frames ⇒ drop point) →
  gap interpolation for short gaps (≤ ~4 frames) → smoothing.
- **Smoothing + differentiation in one step:** Savitzky–Golay (window ~7 frames, order 2)
  or a constant-acceleration Kalman filter. Either yields vertical position, velocity,
  and acceleration per frame; Kalman additionally gives a principled per-frame
  uncertainty. Start with Savitzky–Golay (simpler, scipy already pinned); switch only if
  the eval shows it matters.
- Long gaps (> ~4 frames) are recorded as **occluded intervals** on the track. V4 must
  abstain for onsets inside or adjacent to them. Do not extrapolate through occlusion.
- Identity continuity check: left/right wrist tracks may cross in x (real crossovers) but
  may never swap identities; if the pose backbone flips labels mid-track (known failure
  mode), detect via track-continuity (nearest-neighbor association frame-to-frame beats
  trusting per-frame labels) and re-associate.

### V3 — Kit-region calibration (`v3_kitmap.py`)
- Purpose: map each ADTOF surface class present in the clip (snare, tom(s), hihat,
  cymbal) to an **image region** (center + radius is enough; no homography needed), so V4
  can use "which wrist was moving toward the struck drum" as a cue.
- **v1: one-time manual calibration.** Fixed camera ⇒ one click per surface on a
  reference frame (a tiny matplotlib/OpenCV click tool, or accept a JSON file of
  normalized coordinates alongside the video). Store per segment.
- **Follow-on (not v1): auto-calibration.** For each surface class, collect the wrist
  positions at that class's onsets across the whole clip, restricted to onsets where only
  ONE wrist was moving downward (unambiguous striker); the cluster centroid is the
  surface's region. Ship only if the manual path proves annoying in practice.
- Proximity is a **coarse, secondary** cue (stick decoupling means the wrist hovers
  ~30–40 cm from the impact); weight it accordingly in V4. It exists mainly to separate
  the hands when *both* are in motion (e.g. hat + snare simultaneously in a groove).

### V4 — Per-onset kinematic strike scorer (`v4_strike_scorer.py`) — the heart
For each **hand** hit (Stage 3 time-slot, kicks excluded) at audio time `t`, struck
surface `s` (from ADTOF), evaluate both wrist tracks over a window `[t − 150 ms, t + 100 ms]`:

Per-wrist features:
1. **Downstroke:** peak downward velocity in the pre-onset part of the window.
2. **Rebound:** vertical-velocity sign flip (down→up) near `t`. Localize the flip to
   **sub-frame precision** by linear interpolation of the velocity zero-crossing — at
   30 fps this is what makes phase-locking meaningful.
3. **Phase lock:** |zero-crossing time − t|. The striking wrist's flip sits within a few
   ms of the onset (audio is the ground-truth clock; that is the whole point of
   audio-anchoring). A wrist moving but out of phase is preparing, not striking. This
   also distinguishes a true strike from a feint/preparatory lift.
4. **Proximity:** distance from wrist (at `t`) to the calibrated region of surface `s`
   (coarse cue, low weight).
5. **Quality:** min pose confidence over the window; occlusion/gap overlap.

Combine 1–4 into a per-wrist strike score; the **vote** is the higher-scoring wrist and
the **raw confidence** derives from (score margin between wrists, quality). Keep the
combiner simple and inspectable — a hand-weighted logistic over the 4 features is right
for v1; do not reach for a learned model before the gold set exists (§6).

**Chords (both-hands slots):** a Stage-3 slot can contain two hand strikes (e.g.
snare+crash). Then the question is not "which wrist" but "which wrist on which surface" —
both wrists should show strike kinematics; assign by proximity (cue 4). If only one wrist
shows kinematics, vote that wrist for the acoustically louder strike and abstain on the
other. Never emit two anchors from one ambiguous chord.

### V4.5 — Abstention rules (as important as the votes)
Vision must emit **no vote** (not a weak vote — no vote) when:
- either wrist's track is occluded/gapped in the scoring window;
- the slot's IOI < ~3 frames at the video fps (e.g. < 100 ms at 30 fps): individual
  strokes are unresolvable; a fast roll is a single blur event to the camera. This is a
  *structural* limit — encode it as a hard rule keyed to fps, not a tunable score;
- both wrists (or neither) show strike kinematics on a single-strike slot and the margin
  is sub-threshold;
- the frame region is mid scene-cut / calibration is missing for surface `s`.

### V5 — Confidence calibration (`v5_calibrate.py`)
- The fusion contract (established in the fusion-architecture consensus): every prong's
  confidence must be **independently calibrated** — P = 0.9 must be right ~90% of the
  time — or its log-odds poison the decode.
- Fit a one-parameter temperature (or isotonic if the curve is ugly) mapping raw V4
  margin → calibrated P(vote correct), on the **gold set** (§6), using held-out clips.
- Output per hand hit: `VisionVote {vote: L/R, p_calibrated, basis, abstained: bool}`.

---

## 4. Fusion refactor (Stage 5)

Bounded changes to `src/stage5_hmm.py` + `src/types.py` + `src/pipeline.py`. The Viterbi
skeleton, the state `(L_loc, R_loc, last_hand, prev_hand)`, Stage 4 pruning, and the
HONEST-CEILING comments all stay.

### 4.1 Two-tier injection of vision votes
- **Anchor tier** — `p_calibrated ≥ anchor_threshold` (config; tuned on gold set for the
  ≥ 0.98 precision target, expect ~0.9–0.95):
  the hit's admissible limb set is **reduced to the voted hand** before the Viterbi runs
  (same mechanism Stage 4 already uses to clamp constraint-forced hits). Its output
  `source = "vision"`, `confidence = p_calibrated`, never flagged.
  **Precedence rule:** Stage-4 hard physics outranks an anchor. If an anchor contradicts
  a hard constraint (e.g. both strikes of a sub-70 ms same-surface pair voted to one
  hand), demote BOTH conflicting anchors to evidence tier and log a warning — this is the
  tripwire for wrist-identity flips, worth surfacing loudly.
- **Evidence tier** — abstain < p < anchor_threshold:
  add `log(p / (1 − p))`, **capped at ±3.0**, to the per-hit emission for the voted hand
  (in `_log_emission`, so `_compute_marginal` inherits it automatically — the same wiring
  lesson as HANDOFF §3). Capped so a miscalibrated mid-confidence vote can still be
  outvoted by strong context; ±3 (~95/5) is already near-decisive when unopposed.

### 4.2 What happens to the audio phase priors
The metric-anchor and velocity-accent priors (HANDOFF #1/#2) exist to guess the phase
vision now observes. They become **fallback-only**:
- Keep them implemented and on — they are what carries clips with zero vision coverage
  (audio-only input stays a supported mode; `pipeline.run` without video must work
  exactly as today).
- But **A/B them on the gold set with vision active**: if they change no decisions when
  ≥1 anchor per phrase exists (expected), reduce their weights to tie-breaker scale so
  they can never fight an anchor's interpolation. Do not delete code; retune.
- The grammar (order-2, capped) stays as-is: evidence showed it is a tie-breaker/fixer at
  realistic ambiguity, which is exactly the between-anchors role. Tier-2 self-prior gains
  one improvement: with vision, `source: vision` hits are **legitimate training data**
  for it (they are externally observed, not self-inferred — the laundering guard of spec
  Stage 5b concerned `inference` hits, and still does). Still default OFF; still behind
  the dual-run guard.

### 4.3 Plumbing
- `types.py`: add `"vision"` to `SOURCE_VALUES`; add `VisionVote`; extend
  `PipelineConfig` with `vision_enabled`, `anchor_threshold`, `vision_evidence_cap: 3.0`,
  path/backbone options, and the fps-keyed abstention constant.
- `pipeline.py`: accept video input (Stage 0 already extracts audio from video); run
  V0–V5 between Stage 4 and Stage 5; pass `dict[slot_index → VisionVote]` into
  `assign_limbs`. Vision absent ⇒ empty dict ⇒ current behavior, bit-for-bit.
- `stage6_similarity.py`: template selection for a figure cluster now prefers the
  instance with the most `source: vision` hits (a vision-verified template) over "highest
  Stage-5 confidence". Skeleton-only matching rule unchanged.
- `stage7_output.py` / `server.py` / `static/`: render the source per hit (e.g. solid =
  vision/constraint, hollow = inference, dotted = propagation). The user-facing promise
  becomes: *"hits marked solid were observed, not guessed."*

---

## 5. Keep / demote / retire (explicit)

| Component | Fate |
|---|---|
| Stages 0–4 (ADTOF, madmom, chords, constraints) | **Keep untouched** (shared backbone) |
| Viterbi HMM, state, alternation/reach/continuity | **Keep** — now the interpolator between anchors |
| Double-stroke acoustic signature | **Keep, modest weight** (fast rolls are vision-blind; this is the only per-hit evidence there) |
| Metric anchor + velocity accent priors | **Demote** to fallback/tie-breaker after gold-set A/B (§4.2) |
| Grammar order-2 Tier 1 (capped) | **Keep** as between-anchor tie-breaker |
| Tier 2 self-prior | **Keep OFF**; may later train on `source: vision` hits only, behind the dual-run guard |
| Stage 6 self-similarity | **Keep**, templates now vision-anchored |
| `vision_probe.py` + HandLandmarker (`hand_landmarker.task`) | **Retire** (superseded by `src/vision/`; keep file for the record, delete the 7.8 MB `.task` model once V1 lands) |
| `_compute_marginal` neighbor-context wiring | **Keep** — with anchors in the emission, marginals finally reflect real certainty |
| Honest-ceiling comments, flag discipline | **Keep, verbatim** |

---

## 6. Evaluation (build FIRST — Phase 0)

Everything in the spec's eval section stands. Additions and re-prioritization:

### Gold set
- **The blocking asset.** Current footage is insufficient: `samples/sample_1/ver_1-3.mp4`
  are three versions of ONE sticking (weak, partially wrong GT — known concurrent-kick
  omissions); `samples/sample_2/*.mp3` are audio-only (no vision possible). Collect
  **8–12 video clips** with trustworthy sticking labels:
  - The cheap label source (banked finding): **educational clips with on-screen sticking
    notation** — label by reading the notation + frame-stepping, no pose needed.
  - Span tiers: slow groove / moderate fill / fast doubles; ≥1 off-vocabulary
    (non-repetitive) clip; ≥1 clip with crossovers; ideally ≥1 at 60 fps.
  - Per-hit labels: `{time, surface, limb, tier}` — extend `samples/sample_1.gold.json`
    format; one JSON per clip beside the video.
- Split: calibration clips (fit V5 temperature, tune anchor_threshold) vs held-out
  (report). With ~10 clips, leave-one-out is fine.

### Metrics, in order of authority
1. **Anchor precision** — fraction of anchor-tier votes that match GT. **Gate: ≥ 0.98.**
   If unreachable at any useful coverage, raise the threshold until reached; coverage is
   the dependent variable, never precision.
2. **Run contamination** — fraction of inter-anchor runs containing a wrong anchor
   (measures blast radius of anchor errors; the reason precision outranks coverage).
3. **Anchor coverage** — % of hand hits at anchor tier, reported per tier (slow/med/fast).
   Expect: high on slow, structurally ~0 on fast. That is by design, not failure.
4. **End-to-end hand accuracy** (non-flagged hits) + **flag precision/recall**, per tier —
   unchanged from spec.
5. **Calibration curves** for V5 (vision prong) and final marginals (reliability plots).
6. **Dual-run audits** (unchanged discipline): vision ON vs OFF; each audio prior ON vs
   OFF with vision ON. Any component that shrinks flag-rate without raising accuracy is
   manufacturing confidence — cut or down-weight it.

---

## 7. Build order, with gates

Each phase ends with a measurable check; do not proceed through a failed gate.

- **Phase 0 — Gold set + eval harness.**
  Collect/label clips (§6); extend `eval.py`/`gold_eval.py` to compute the §6 metrics
  from a clip + gold JSON; **re-measure the current audio-only pipeline on the gold set**
  → this baseline is the number every later phase must beat.
  *Gate:* harness runs end-to-end; baseline table exists.
- **Phase 1 — Pose bake-off (go/no-go probe).**
  Implement V0 + V1 + minimal V2 for the candidate backbones (§3 V1, in order); on 3
  gold clips measure: % frames both wrists tracked, track continuity through strikes,
  wall-clock per minute of video.
  *Gate:* ≥ 80% of frames with both wrists tracked on slow/medium clips for at least one
  backbone. **If no backbone passes, stop and re-plan** (see §8 fallbacks) — do not build
  V4 on a failed V1.
- **Phase 2 — Kinematic scorer.**
  Full V2 + manual V3 + V4 + abstention. Evaluate votes directly against gold (no fusion
  yet): vote accuracy vs margin, abstention behavior.
  *Gate:* a margin threshold exists giving **precision ≥ 0.98 at coverage ≥ 30%** on
  slow/medium tiers.
- **Phase 3 — Calibration + fusion.**
  V5 temperature fit; Stage-5 two-tier injection (§4.1); plumbing (§4.3).
  *Gate:* end-to-end hand accuracy beats the Phase-0 audio-only baseline on every tier;
  flag-rate drops on slow/medium **without** accuracy loss; audio-only mode still
  reproduces the Phase-0 baseline exactly.
- **Phase 4 — Between-anchor tuning.**
  A/B the demotions (§4.2); Stage-6 vision-anchored templates; weight retune on
  calibration clips only.
  *Gate:* held-out clips improve or hold; nothing overfits (same weights help all clips).
- **Phase 5 — Output & product surface.**
  Stage-7/UI source rendering; calibration click-tool ergonomics; README + spec updates;
  delete retired probe assets.
  *Gate:* full run on a fresh (non-gold) video produces sane, source-tagged output.

---

## 8. Risks and pre-agreed fallback positions

| Risk | Likelihood | Fallback |
|---|---|---|
| Body pose also fails on found footage (Phase-1 gate) | Moderate — untested, but the probe's failure causes are Hands-specific | Narrow the product claim to **curated/controlled footage** (user films themself: framing + 60 fps solves V1) and/or add a stick-tip motion tracker (frame-difference + line fit along the forearm axis) as V1b. Do not silently ship audio-only under a vision banner. |
| Wrist identity flips (crossovers) | Known failure mode | V2 continuity association + §4.1 precedence tripwire; crossover clip in the gold set makes it measurable, not anecdotal. |
| 30 fps kills coverage on medium tier | Possible | Sub-frame interpolation (V4.2) is the mitigation; if insufficient, prefer 60 fps sources and say so in the README. Never lower `anchor_threshold` to buy coverage. |
| Anchor errors poison runs | The design's one real hazard | The ≥ 0.98 precision gate + run-contamination metric + physics-precedence demotion (§4.1). |
| Env friction (new vision dep vs `numpy<2`/TF pins) | Windows, so yes | Bake-off order starts with zero-new-deps (MediaPipe Pose); any addition validated in a scratch env before touching the shared `.venv`. |

---

## 9. Guardrails — do NOT

1. Do not resolve fast rolls with vision. Below the fps floor, abstention is correct;
   the double-signature + constraints + grammar carry those hits, flagged when weak.
2. Do not let coverage pressure erode `anchor_threshold`. Precision is gated, coverage
   floats.
3. Do not use screen-position for handedness. Skeleton chain only.
4. Do not modify Stages 0–4, and do not break audio-only mode (it is both the fallback
   product and the regression baseline).
5. Do not delete or weaken the HONEST-CEILING comments or the flagging discipline. The
   residue shrinks; it does not disappear.
6. Do not train anything (V4 combiner, Tier-2 prior) on the model's own inferences.
   Gold labels and `source: vision` hits only.
7. Do not tune on held-out clips. Calibration split only.

---

## 10. Open decisions for the implementer (resolve during build, in this order)

1. Pose backbone (Phase-1 bake-off — measured, not argued).
2. Savitzky–Golay vs Kalman for V2 (start S-G; switch only on evidence).
3. Exact anchor_threshold (tuned on calibration clips against the precision gate).
4. Manual vs auto kit calibration in v1 (ship manual; auto only if manual annoys).
5. Whether demoted audio phase priors keep nonzero weight in vision mode (Phase-4 A/B).

These five are the ONLY open decisions. Everything else is specified in §11.

---

## 11. Implementation appendix (normative)

### 11.0 Environment and process rules

- **Interpreter:** the shared venv at repo root — `..\..\.venv\Scripts\python.exe` is
  wrong; from this project dir it is `..\.venv\Scripts\python.exe`. Set
  `TF_USE_LEGACY_KERAS=1` and `PYTHONUTF8=1` for anything touching ADTOF. Always run
  project code as modules (`python -m src.x`) from this directory; `src/types.py` shadows
  stdlib `types` otherwise. Samples live at `../samples/`.
- **numpy<2 is load-bearing** (madmom/ADTOF compiled ABI). Before adding ANY vision
  dependency to the shared venv: create a throwaway venv, `pip install` the candidate
  there, confirm `import <pkg>; import numpy; assert numpy.__version__.startswith("1.")`
  after installing it *alongside* `numpy==1.26.4`. If the candidate force-upgrades numpy,
  it is disqualified — note it and move to the next backbone in the §3 V1 order.
- **ADTOF model:** load once per session via the existing loader and pass
  `adtof_model=` into `pipeline.run` (model load is slow).
- **Audio-only regression protocol (mandatory, every phase):** in Phase 0, run the
  current pipeline audio-only on `../samples/sample_1/ver_1.mp4` and serialize the
  assigned strikes (time, surface, limb, confidence rounded to 6 dp, source, flag) to
  `eval_out/baseline_sample1.json`. After every later phase, re-run with
  `vision_enabled=False` and diff against that file. **Any difference = the phase broke
  guardrail §9.4; fix before proceeding.**
- **Stop conditions — hand control back to the user instead of improvising when:**
  (a) gold clips are needed (Phase 0): you cannot download/scrape videos and you must
  NEVER fabricate labels or derive them from pose output (circularity). Deliverable when
  blocked: the eval harness + `../samples/GOLD_LABELING.md` (what to collect per §6, the
  JSON schema of §11.1, one worked example) — then stop and ask.
  (b) a phase gate fails: write `PHASE<N>_FINDINGS.md` with the measured numbers and
  stop. The §8 fallbacks are *pre-agreed positions for the user to choose from*, not
  license to pivot silently.
  (c) a §11 default seems wrong: report the measurement that says so; change it only as
  part of the phase whose gate measures it.

### 11.1 Interfaces and schemas (exact)

**Pose interface (V1)** — one class per backbone, all implementing:
```python
class PoseBackbone:
    def process(self, frame_bgr: np.ndarray, t_ms: int) -> dict:
        """Returns {name: (x, y, conf)} with x, y normalized to [0,1], y DOWNWARD.
        Keys (exactly): left_wrist, right_wrist, left_elbow, right_elbow,
        left_shoulder, right_shoulder. Missing keypoint -> conf 0.0, x=y=nan.
        'left' = the DRUMMER'S left (anatomical, from the skeleton), never screen side."""
```
Keypoints with `conf < 0.5` are treated as missing by V2.

**Track structure (V2)** — per segment, per wrist, parallel numpy arrays sampled at the
video frame rate: `t` (sec), `x`, `y` (normalized), `vy` (units: normalized-height/sec,
positive = downward), `ay`, `valid` (bool). Plus `occlusions: list[(t0, t1)]`. Store both
wrists in a `WristTracks` dataclass with `fps` and the segment bounds.

**Kit map (V3)** — JSON file next to the video, `<video_stem>.kitmap.json`:
```json
{"segments": [{"start_frame": 0, "end_frame": 5400,
  "surfaces": {"snare": {"x": 0.52, "y": 0.66, "r": 0.07},
               "hihat": {"x": 0.30, "y": 0.55, "r": 0.08}}}]}
```
Normalized image coords; `r` = region radius. Surfaces use ADTOF class names
(plus `tom0..tomN` if the clip's kit map distinguishes them; a plain `tom` entry is the
fallback for all sub-toms). A missing surface entry ⇒ V4 abstains on hits of that surface
(§4.5), with a logged count so it is visible.

**Gold labels (Phase 0)** — one JSON per clip, `<video_stem>.gold.json`, format matching
`gold_eval.py`: a list of `{"time": <sec>, "limb": "L"|"R"|"F", "tier":
"easy"|"medium"|"hard", "surface": <adtof class>}`. Matching rule for ALL metrics:
predicted hit ↔ gold hit paired greedily within **±50 ms**; limb accuracy is computed
over matched pairs only; unmatched counts (both directions) are reported separately as
transcription error, never folded into Stage-5 accuracy.

**VisionVote (V5 output / fusion input)**:
```python
@dataclass
class VisionVote:
    slot_index: int          # index into constraint_dicts
    strike_index: int        # index within slot.strikes (chords: one vote per hand strike)
    time: float
    vote: Optional[str]      # "L" / "R" / None when abstained
    p_raw: float             # combiner margin -> sigmoid, uncalibrated
    p_calibrated: float      # after V5; == p_raw before calibration exists
    abstained: bool
    basis: dict              # feature values (for audits); keys fixed in 11.2
```
Fusion consumes `votes: dict[tuple[int, int], VisionVote]` keyed `(slot_index,
strike_index)`, abstentions omitted.

### 11.2 Numeric defaults (all in config, these exact starting values)

| parameter | default | where |
|---|---|---|
| scene-cut: mean abs gray diff (frames downscaled to 64×36, 0–255) | > 25.0 ⇒ cut | V0 |
| pose keypoint min confidence | 0.5 | V1/V2 |
| teleport gate (per-frame displacement, fraction of frame diagonal) | > 0.12 ⇒ drop point | V2 |
| gap interpolation limit | ≤ 4 frames; longer ⇒ occlusion interval | V2 |
| Savitzky–Golay | window 7 (9 at ≥50 fps), polyorder 2 | V2 |
| V4 scoring window | [t − 150 ms, t + 100 ms] | V4 |
| phase-lock score | `max(0, 1 − |Δt| / 0.040)` (Δt in sec) | V4 |
| downstroke score | peak `vy` in pre-onset half ÷ clip-level 75th-pct of per-onset peak `vy` for that wrist, clipped to [0, 2] | V4 |
| rebound score | 1.0 if a vy down→up zero-crossing exists in the window, else 0.0 | V4 |
| proximity score | `max(0, 1 − dist_to_region_center / (2r))` | V4 |
| combiner (per wrist) | `2.0·phase_lock + 1.5·downstroke + 1.0·rebound + 0.5·proximity` | V4 |
| vote margin | `score_winner − score_loser`; **abstain if margin < 0.5** or any V4.5 rule fires | V4 |
| `p_raw` | `1 / (1 + exp(−margin))` | V4 |
| quality veto | any occlusion overlap of the window, or min keypoint conf < 0.5 in window ⇒ abstain | V4.5 |
| structural IOI floor | slot IOI < `3.0 / fps` sec ⇒ abstain | V4.5 |
| `anchor_threshold` | 0.93 (retuned in Phase 3 against the 0.98 precision gate) | fusion |
| `vision_evidence_cap` | 3.0 | fusion |
`basis` keys: `phase_lock, downstroke, rebound, proximity, margin, quality_min_conf`
(for both wrists, prefixed `L_`/`R_`).

### 11.3 Fusion touch points (exact code anchors, current line numbers)

1. **`types.py`**: append `"vision"` to `SOURCE_VALUES`; add the `VisionVote` dataclass
   (§11.1); add to `PipelineConfig`: `vision_enabled: bool = False`,
   `anchor_threshold: float = 0.93`, `vision_evidence_cap: float = 3.0`,
   `vision_abstain_ioi_frames: float = 3.0`, `pose_backbone: str = "mediapipe"`,
   `kitmap_path: Optional[str] = None`.
2. **Known bug — fix as part of this phase:** `types.py::NATURAL_HAND` hihat entries are
   inverted for all four setups ("crossed" means the LEAD hand crosses to the hihat).
   Correct values: `right_crossed: hihat→"R"`, `right_open: hihat→"L"`,
   `left_crossed: hihat→"L"`, `left_open: hihat→"R"`. This is live in the HMM
   (`_log_emission` charges −0.4 against the natural hand, `stage5_hmm.py:433`), so fixing
   it WILL change audio-only output: fix it **in Phase 0, before** snapshotting
   `baseline_sample1.json`, so the regression baseline is post-fix. (The identical fix
   already shipped in the ergonomic project on 2026-07-04.)
3. **`stage5_hmm.py::assign_limbs`** (line 49): add keyword param
   `vision_votes: Optional[dict] = None` (the §11.1 dict; `None`/empty ⇒ behavior
   identical to today).
   - **Anchor tier:** before the Viterbi loop, for each vote with
     `p_calibrated ≥ config.anchor_threshold`: check precedence — if the voted hand is
     not in `constraint_dicts[si]["admissible"][ki]`, or two anchors on consecutive
     slots with IOI < `single_hand_min_ioi_ms` and same surface vote the SAME hand,
     demote the offending vote(s) to evidence tier and
     `logger.warning("vision anchor demoted ...")` (the §4.1 tripwire). Surviving anchors:
     replace `constraint_dicts[si]["admissible"][ki]` with `frozenset({vote})` and record
     `(si, ki)` in an `anchored` set. This reuses `_enumerate_assignments` untouched.
   - **Source labels:** the existing line
     `source = "constraint" if cd["admissible"].get(i) == frozenset({limb}) else "inference"`
     (line 226) must check `anchored` first: `(t, i) in anchored ⇒ source = "vision"`,
     `confidence = p_calibrated`, `flagged_ambiguous = False`.
   - **Evidence tier:** add param `vision_vote` to `_log_emission` (line 416) and pass
     `vision_votes.get((t, i))` per strike at BOTH call sites — the Viterbi loop
     (line 159) and inside `_compute_marginal` (line 618; threading it here is exactly
     the HANDOFF §3 marginal-wiring lesson). In the per-strike loop of `_log_emission`,
     for a non-anchored vote: `log_p += clip(log(p/(1−p)), −cap, cap)` if
     `limb == vote` else `−=` the same quantity, with `cap = config.vision_evidence_cap`.
4. **`pipeline.py::run`** (line 23): treat `input_path` with suffix in
   `{.mp4,.mov,.mkv,.webm,.avi}` as video (Stage 0 already extracts audio from video).
   If `config.vision_enabled` and input is video: after Stage 4, run V0→V5 to produce
   `votes`, and pass `vision_votes=votes` into `assign_limbs`. All vision imports stay
   **inside** that branch (audio-only must not import cv2/mediapipe).
5. **`stage6_similarity.py`**: when choosing a cluster's template instance, sort by
   `(count of source=="vision" hits, mean confidence)` descending instead of confidence
   alone. One-line change; leave matching logic untouched.
6. **Stage 7 / `server.py` / `static/`**: expose `source` per hit; rendering per §4.3.
   Do not redesign the UI; add the source distinction minimally.

### 11.4 Per-phase deliverables (files, commands, artifacts)

| Phase | New files | Verify with |
|---|---|---|
| 0 | `eval.py` (harness: `python eval.py <video> <gold.json> [--vision off]` → §6 metrics table), `../samples/GOLD_LABELING.md`, `eval_out/baseline_sample1.json`, NATURAL_HAND fix | harness runs on sample_1; baseline written; findings on GT caveats printed, not hidden |
| 1 | `src/vision/v0_segments.py`, `v1_pose.py` (+ backbone impls), `probe_pose.py` (CLI: `--clip --backbone` → % both-wrist frames, continuity, sec/min-video) | table over 3 gold clips × candidate backbones; gate §7-P1 |
| 2 | `v2_tracks.py`, `v3_kitmap.py` (+ click tool `calibrate.py`), `v4_strike_scorer.py` with V4.5 rules | `probe_votes.py`: vote-vs-gold table, precision/coverage curve vs margin; gate §7-P2 |
| 3 | `v5_calibrate.py`, fusion edits (§11.3), `vision_types.py` if circular imports force it | `eval.py` full run, vision ON vs OFF, plus the audio-only diff protocol (§11.0) |
| 4 | A/B configs only (no new modules) | `eval.py` sweeps recorded in `PHASE4_FINDINGS.md` |
| 5 | UI/output edits, README + spec updates, delete `hand_landmarker.task` | fresh-video smoke run |

All probe/eval scripts live in the project root (pattern: `gold_eval.py`), import via
`src.`, and print tables — no notebooks, no hidden state.

### 11.5 Conduct rules for the implementing model

1. A failed gate is a **result**, not an obstacle: record it and stop (§11.0). Do not
   lower a threshold, shrink a window, or re-define a metric to pass a gate.
2. Never label, pseudo-label, or augment gold data (§9.6). If the gold set feels small,
   that is the user's call to expand — ask.
3. Every tunable you introduce must appear in `PipelineConfig` (or a `VisionConfig`
   dataclass in `types.py`) with the §11.2 default and a one-line comment. No literals
   buried in function bodies.
4. Keep diffs surgical (CLAUDE.md): do not reformat, rename, or "improve" the HMM while
   threading votes through it. The §11.3 edits are the complete intended footprint in
   existing files.
5. When §§1–10 and your instinct disagree, follow the document; when the document is
   genuinely silent, choose the smallest reversible step and note the choice in the
   phase findings file.
