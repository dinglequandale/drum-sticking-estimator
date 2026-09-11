---
name: project-grammar-order1-redundant
description: "Evidence that the sticking grammar integrated at order-1 only re-derives stage5's existing alternation/lead-hand heuristics, so weight/scale tuning cannot change decisions — order-2 is the only live lead for decision gains"
metadata:
  node_type: memory
  type: project
  originSessionId: 4ed1ae29-55c6-4d2e-8c23-38cc60869cbe
---

The learned sticking grammar ([[project-fusion-architecture]], `src/grammar.py`) was integrated into `stage5b_priors.RudimentPrior` as a bounded log-odds nudge in `_log_emission`. Audio-free dual-run (`src/synth_eval.grammar_dual_run`, with `vel_noise` + a `_decision_accuracy` metric I added) shows:

- **Decision accuracy delta = +0.000 at every weight, scale (1–12x), and noise level.** The grammar changes ZERO decisions. Its only effect is calibration — flagging confidently-wrong off-vocab hits (nonflag_acc rises, flags rise). Verdict "ADMIT +0.030" was entirely flagging, not fixing.
- **Wiring is fine** (forcing a constant ±5 nudge tanks/moves decision_acc as expected), so it's not a bug.
- **Root cause: redundancy at order-1.** At inference the Viterbi state carries only `last_hand` (order-1). The grammar's order-1 L/R log-odds are sizable but are pure alternation + lead-hand default (empty→favor R −0.98; after L→favor R −0.77; after R→favor L +0.67) — exactly what `alternation_weight` and the natural-hand/lead bias already encode. Amplifying a redundant signal just reinforces the same argmax.
- The grammar's NON-redundant signal is at **order-2** (e.g. after R,L: +0.10, which diverges from the order-1 L→−0.77), and that is unreachable until the Viterbi state is extended to carry 2-deep hand history (~3x state space).

**Leads, with evidence:** weight/scale tuning = inert *at order-1* (nothing non-redundant to amplify). Order-2 state extension = the live lead. Corpus growth = premature until the grammar can move decisions.

**UPDATE — order-2 built and validated.** Extended the Viterbi state from `(L_loc,R_loc,last_hand)` to `(L_loc,R_loc,last_hand,prev_hand)` (~2.3x states); `_grammar_context` now feeds order-2 history. Result on synth_eval decision_accuracy: clean data +0.000 (no ties); **vel_noise=0.1: +0.015 ALL, +0.083 off-vocab (a real FIXER)**; vel_noise=0.2: -0.015/-0.083 (overreach). So order-2 converts the grammar from flagger to fixer at realistic ambiguity but overrides at heavy noise. Cap sweep with order-2: <=0.15 inert, >=0.3 saturates — the cap CANNOT separate help@0.1 from hurt@0.2, so the overreach is a **confidence-gating** problem (defer to flagging when the acoustic margin is near-zero), not a weight one. n is tiny (off-vocab ~36 hits), and the effect sign flips with noise regime → synth_eval has hit its limit; the gold set is now the critical path for tuning cap/gating and the real verdict.

**Eval caveat (the bigger one):** synth_eval uses idealized symbolic observables, so acoustic terms are rarely ambiguous → few ties for a tie-breaker to flip → it likely *understates* the grammar's real-audio value. And it tests partly in-vocabulary patterns. The trustworthy test is real audio + ground-truth sticking; no gold set exists yet, but `samples/sample_1.mp4` has on-screen R/L/K notation ([[project-vision-probe-findings]]) that can label real audio cheaply.
