---
name: project-synth-eval-harness
description: "src/synth_eval.py — audio-free symbolic Stage-5 eval harness; a logic/calibration guardrail, NOT training data"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

`src/synth_eval.py` (built 2026-06-22) generates symbolic sticking patterns with known ground-truth L/R (rolls, paradiddle, rock groove, tom fill), builds the exact structures Stage 5 consumes (`constraint_dicts` + `beat_info`, with `quantized_pos` computed via stage2's bar-fraction formula), runs `assign_limbs` with `wav_path=None`, and scores against GT pooled by tier. Run: `python -m src.synth_eval`. Matching is surface-aware (keyed on rounded-time + surface) because chords put kick+hihat at the same instant and time-only matching cross-matches them.

**Role:** a fast logic/regression gate and the **calibration instrument** for the grammar prong — never training data. It cannot capture real acoustic priors (double-vs-single timbre), and `wav_path=None` strips the attack-envelope cue, so it *overstates* double-stroke failure. See [[project-fusion-architecture]].

**Findings on the current hand-tuned HMM (default config):**
- Single-stroke rolls + tom fill: 100% (alternation + surface-change logic are solid).
- Double rolls (RRLL) and paradiddle collapse to pure alternation (RLRL), ~50%, because alternation_weight=2.0 dominates and nothing reconstructs doubles. Partly an audio-free artifact — confirm magnitude with a future audio harness.
- Rock groove: hi-hat ostinato gets alternated (should be one hand); continuity_bonus=0.5 loses to alternation_weight=2.0. NOT an artifact — a real missing structural prior.
- **Meta-finding:** both failures have flag_rate ~0 = confidently wrong. The honest-ceiling flagging catches local even-double coin-flips but is blind to *structural* sticking misreads. This is a calibration failure, and it motivates the learned-grammar + calibration direction.
- `metric_anchor_weight` sweep is flat over the current bank (patterns don't discriminate it). To finally settle [[project-metric-anchor-tradeoff]], add a phase-discriminating fill (the sample_3 scenario) to the bank.
