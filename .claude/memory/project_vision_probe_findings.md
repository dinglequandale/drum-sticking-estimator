---
name: project-vision-probe-findings
description: Result of the first audio-anchored MediaPipe probe — off-the-shelf hand pose is a poor labeler on found drum footage; on-screen sticking notation is a better corpus source
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

Ran `vision_probe.py` (audio-anchored MediaPipe HandLandmarker, Tasks API) on `samples/sample_1.mp4` (2026-06-22). Outcome: off-the-shelf hand pose is **not** a viable vision labeler/prong on this kind of footage as-is.

**Numbers (17 onsets):** any hand detected only 53% of the time; **both hands detected 0%**; every detection was a single low-confidence hand mislocated near the snare, not on the drummer's actual hands.

**Why it fails (visible in probe_out/ overlays — these are the durable lessons):**
1. **Drumsticks decouple the hand from the impact point** — the stick tip hits the drum ~30-40 cm from the tracked hand, so "which hand is lowest" ≠ "which stick struck." Hand pose alone can't localize the hit.
2. **Side / 3-quarter camera angle** → the far hand is routinely occluded by body/near-arm/cymbals (hence 0% two-hand frames). The whole two-hand-margin heuristic never fires.
3. **Small, fast, motion-blurred hands at 30 fps** → poor detection. MediaPipe Hands is trained on large, clear, front-facing hands (gesture/selfie), not distant drummers.

**Serendipitous finding that reframes labeling:** these sample videos are *educational* clips with the sticking **printed on-screen as notation** (staff + per-note R/L/K captions, e.g. "Breakdown with snare only"). For building the real-audio + GT-sticking training corpus, **reading the on-screen notation (OCR/template-match + audio alignment) is far cheaper and more reliable than pose estimation.** This labels real audio without solving the hard vision problem.

**Re-sequencing implication for [[project-fusion-architecture]]:** separate the two vision roles sharply. (a) *Labeling/corpus* — use on-screen-notation educational videos, not pose; this unblocks training the learned grammar (leg 3) now. (b) *Inference-time vision prong* (leg 2) is the genuinely hard part: off-the-shelf pose is insufficient; would need dedicated stick/tip tracking, better angles, or higher fps, and should be treated as an optional confidence-gated bonus, deferred. The deployed product still targets videos WITHOUT notation; notation videos are purely a training-label source.

Artifacts kept at repo root: `vision_probe.py`, `hand_landmarker.task` (7.8 MB model), `probe_out/` overlays.
