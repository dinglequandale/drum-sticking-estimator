# Ergonomic Sticking Estimator (audio-only)

**Problem:** given a drum audio track (mp3 upload, no video required), output the **most
economical / ergonomic / idiomatic sticking** a competent right-handed drummer would most
likely play, with **calibrated uncertainty**. Genuinely ambiguous groupings are flagged and
treated as don't-cares, not fabricated.

This is deliberately **not** the same problem as `../vision_sticking_estimator/`, which tries
to recover the sticking a drummer *actually* used (needs vision + a large sticking-labeled
corpus). Reframing the goal to "optimal ergonomic sticking" removes that data bottleneck: the
backbone is parametric biomechanics, needing no labeled corpus.

## Status
- **Backbone (Stages 0–4) copied and working** — ingest, ADTOF onsets/surfaces/velocity,
  madmom beats/grid, chord consolidation, hard constraints. Identical to the vision project;
  do not diverge without reason.
- **Stage 5 (the core) not yet built.** Design is in `DESIGN_STAGE5_SEGMENTAL.md` — a
  segmental (grouping-based) model: cover the note stream with metric cells, assign each a
  sticking *template* (with accents), minimize total ergonomic cost, decode with semi-Markov
  Viterbi, and flag low-margin (don't-care) regions.

## Read before touching Stage 5
1. `DESIGN_STAGE5_SEGMENTAL.md` (this project's contract).
2. `../vision_sticking_estimator/STICKING_ESTIMATOR_SPEC.md` for the Stage 0–4 backbone the
   two projects share.

## Env (Windows)
Shared `../.venv/Scripts/python.exe`; set `TF_USE_LEGACY_KERAS=1` and `PYTHONUTF8=1`. ADTOF
model load is slow — load once and reuse. Test clips are in `../samples/`.
