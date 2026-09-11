---
name: project-fusion-architecture
description: "Agreed long-term architecture — audio+vision fusion with a learned, calibrated sticking grammar over the HMM; supersedes the audio-only-only framing"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

Consensus reached with Konrad (2026-06-22) on the target architecture for sticking estimation, replacing the pure audio-only mental model. The deployment target will likely have video as well as audio.

**Core principle:** audio and vision fail in *opposite* places, so the answer is fusion, not either/or. Audio is reliable for onset timing / velocity / surface but structurally cannot resolve which-hand on many hits (the honest ceiling). Vision directly observes which-hand but is flimsy (occlusion, camera angle, wrist-vs-arm motion, low temporal resolution on fast rolls). Neither is "primary."

**Three-legged stool:**
1. Audio front-end (already built, Stages 0-5) = the event backbone: onsets, velocity, surface, IOI, metric position. Defines the timeline.
2. Audio-anchored vision module: at each audio onset, emit P(L/R) + a self-confidence. Vision is NEVER asked to detect strokes (audio does that) — only "which hand was down at this known timestamp," which is a far easier, bounded question. Used in two roles: (a) offline labeler to build a real-audio + GT-sticking training corpus from curated clips, and (b) inference-time soft vote. Confidence-gated: when vision can't see, its vote is down-weighted and audio+grammar carry the hit, so vision is strictly additive (floor = audio-only performance).
3. Learned sticking grammar = "an LLM for drum chops": a *conditional* sequence model P(next sticking label | history, observation). The current HMM is the degenerate version (hand-coded bigram transition = the alternation_weight knob). Replace that hand-set prior with a learned, higher-order, calibrated one. Keeps the same Viterbi/Bayes skeleton: learned-grammar prior × per-hit likelihoods (audio ⊕ vision) = calibrated posterior.

**Calibration is the contract that makes fusion valid:** fusion combines log-likelihoods, so a systematically overconfident prong poisons the posterior even when wrong. Each prong must be independently calibrated (high confidence ⟺ high correctness; uncertain ONLY at genuine capability limits). Measurable because we have GT — reliability curves + temperature-scaling per prong. The "confidently wrong" failures the synth harness found are a *calibration* bug, not an accuracy bug. See [[project-synth-eval-harness]].

**Why training audio-from-vision isn't circular:** we keep vision at inference (fusion), we don't discard it — vision is both teacher (labeler) and teammate (evidence stream). Synthetic generation is demoted to a logic/regression guardrail only; it cannot capture real acoustic priors (e.g. double-vs-single-stroke timbre), so real training labels must come from vision-on-real-audio.

**Hard parts to expect:** crossed-grip handedness flips in off-the-shelf hand trackers (need frame-to-frame identity tracking + kit-geometry priors); vision temporal resolution on fast passages (covered by the grammar's limited fast-roll vocabulary + graceful flagging).

**Agreed next concrete step (not yet started):** an audio-anchored MediaPipe Hands probe on `samples/sample_1.mp4` — at each onset, get a which-hand vote + confidence, and check whether confidence tracks correctness on one curated clip. This single cheap experiment validates or kills the vision prong before any larger build.
