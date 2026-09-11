---
name: project-vision-first-refactor
description: "2026-07-04 decision: vision_sticking_estimator reactivated with a vision-PRIMARY refactor — DESIGN_VISION_FIRST.md is the build contract (sparse high-precision vision anchors, audio HMM demoted to between-anchor interpolator/fallback)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d48d524-79cf-452a-97d6-48e185b7b14a
---

On 2026-07-04 Konrad reframed the [[project-two-project-split]] vision project: the estimator
must become **deterministic-where-visible via clever vision**, with the audio/probabilistic
machinery as *fallback* for occluded/uncertain hits — the current code leans the wrong way
(audio-heavy soft priors). Kick is trivially solved by ADTOF, leaving a binary L/R problem.

**The build contract is `vision_sticking_estimator/DESIGN_VISION_FIRST.md`** (written this
session, for another model to implement). Core commitments:

- **Sparse anchors + constrained interpolation:** vision need not resolve every hit; anchor
  precision ≥ 0.98 is THE gated metric (one anchor pins the parity of a whole alternating
  run — fixes the documented floating-phase failure); coverage floats.
- **Why the old probe verdict doesn't condemn vision:** [[project-vision-probe-findings]]
  tested MediaPipe *Hands* on single frames. The refactor uses whole-body pose (wrist via
  arm chain, robust to blur/distance), continuous per-frame tracking, and the
  [[project-kinematic-onset-emission]] scorer (downstroke + rebound zero-crossing
  phase-locked to the audio onset, sub-frame interpolated). Phase-1 pose bake-off
  (MediaPipe Pose VIDEO mode → rtmlib/RTMPose → YOLO-pose) is a go/no-go gate.
- **Fusion:** anchor tier clamps admissible limbs pre-Viterbi (Stage-4 physics outranks
  anchors — conflict demotes anchor and flags identity-flip); evidence tier = capped ±3
  log-odds in `_log_emission`. Metric-anchor/velocity-accent priors demoted to
  fallback/tie-breakers; grammar order-2 stays as between-anchor tie-breaker; audio-only
  mode must stay bit-for-bit intact.
- **Structural abstention:** IOI < ~3 frames at clip fps ⇒ vision never votes (fast rolls
  stay flagged/audio territory). `source: "vision"` added; vision hits are legal Tier-2
  training data (externally observed, not laundered inference).
- **Blocking asset: gold set** (Phase 0) — 8–12 sticking-labeled *video* clips, labeled
  cheaply from on-screen-notation educational footage; current samples are insufficient
  (sample_1 = one sticking ×3, sample_2 = audio-only mp3s).

**2026-07-05: normative implementation appendix (§11) added to DESIGN_VISION_FIRST.md**
because the implementer will be a weaker model (likely Sonnet 4.6). §11 pins: env/process
rules (audio-only bit-for-bit regression via `eval_out/baseline_sample1.json`; scratch-venv
numpy<2 check before any dep; STOP conditions — gold clips are the USER's to provide, never
scrape/fabricate/pose-derive labels; failed gate ⇒ findings file + stop), exact schemas
(PoseBackbone.process, WristTracks, kitmap JSON, gold JSON ±50 ms matching, VisionVote keyed
(slot_index, strike_index)), all numeric defaults (scene-cut 25.0, teleport 0.12, gap ≤4,
S-G 7/2, V4 window −150/+100 ms, combiner 2.0·phase+1.5·down+1.0·rebound+0.5·prox, abstain
margin <0.5, anchor_threshold 0.93, cap ±3), and exact fusion code anchors (assign_limbs:49,
_log_emission:416 + call sites 159/618, source label line 226, pipeline.run:23).
**Found live bug:** vision `types.py::NATURAL_HAND` hihat entries inverted for all 4 setups
(same as the ergonomic bug fixed 2026-07-04) and ACTIVE via the −0.4 natural-hand penalty at
stage5_hmm.py:433 — §11.3 mandates fixing it in Phase 0 BEFORE snapshotting the regression
baseline.

Next chat per Konrad: the ergonomic/optimizer subproject ("sticking optimizer", the
mathematically interesting one) — review its cost metrics and their effectiveness.
[Done 2026-07-04/05: see [[project-ergo-diagnosis-2026-07-04]] + HANDOFF_V2_FIXES.md.]
