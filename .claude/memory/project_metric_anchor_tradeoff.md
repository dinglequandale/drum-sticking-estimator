---
name: project-metric-anchor-tradeoff
description: Open question — metric_anchor_weight=1.0 (default) sometimes locks in a confidently-wrong hand on fast single-stroke fills that break the lead-on-beat convention
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

The Stage 5 L/R confidence package (HANDOFF_LR_PRIORS.md) is implemented in [src/stage5_hmm.py](src/stage5_hmm.py): #0 marginal rewire, #1 metric anchor, #2 velocity/accent, all at the handoff's default weights (`metric_anchor_weight=1.0`, `velocity_accent_weight=1.0`, `marginal_neighbor_weight=1.5` in `PipelineConfig`).

On the three smoke-test samples: sample_2's hand path now matches ground truth exactly; sample_3's hand path is unchanged from the pre-package baseline (still wrong on the same hits) but now reports much higher confidence on those wrong hits. Ablation showed velocity/accent alone fixes sample_3 to 9/10 hand accuracy, but metric_anchor at any weight >= 0.05 reverts to the wrong phase — traced to a real sticking pattern (a fill that keeps alternating straight through the barline rather than realigning the lead hand to the beat), not a calibration bug. quantized_pos's fractional part encodes position-within-bar, not within-beat — `_metric_strength` in stage5_hmm.py rescales by beats_per_bar before applying the on-beat/&/e-a table; this was already corrected during implementation, not part of the open question.

**Why:** Konrad decided to keep the metric-anchor weight at the handoff's spec default (1.0) rather than lowering it, given only 3 test samples — wants more samples before concluding the convention is unreliable enough to deweight.

**How to apply:** Before changing `metric_anchor_weight` away from 1.0, or before declaring the metric-anchor prior settled either way, test against additional sample clips beyond sample_1/2/3 (more fills, more grooves, varied tempos) to see whether the lead-on-beat convention holds up or whether sample_3's failure is the common case rather than the exception. If it keeps losing to velocity evidence on fast rolls, that's a real signal to dial it down — but one fast fill isn't enough evidence yet.
