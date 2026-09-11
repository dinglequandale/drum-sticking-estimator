# Vision Sticking Estimator (safekept)

**Problem:** recover the sticking a drummer *actually* used, for short clips where a drummer
shows off their chops — using audio as the event backbone and **vision** (audio-anchored
which-hand detection) plus a **learned sticking grammar** to resolve the hands audio cannot.

**Status: reactivating (2026-07-04) — vision-first refactor planned.** The build contract for
the final version is **`DESIGN_VISION_FIRST.md`**: vision kinematics become the primary,
near-deterministic which-hand evidence (sparse high-precision anchors), and the audio HMM is
demoted to the constrained interpolator/fallback between anchors. Read that document first; it
supersedes the "audio-only, pose rejected" framing of `STICKING_ESTIMATOR_SPEC.md`.
The sibling `../ergonomic_sticking_estimator/` (most-ergonomic sticking from audio alone)
remains a separate, active project.

This directory is a complete, self-contained snapshot — it was relocated wholesale, not gutted.

## What's here
- `src/` — the full 7-stage pipeline: Stages 0–4 (shared transcription backbone), the
  note-wise Viterbi HMM (`stage5_hmm.py`, state `(L_loc, R_loc, last_hand, prev_hand)`),
  Stage 5b rudiment/self priors, learned grammar (`grammar.py`), self-similarity, output,
  and `synth_eval.py` (audio-free Stage-5 logic/calibration guardrail).
- `STICKING_ESTIMATOR_SPEC.md` — the original 7-stage build contract.
- `HANDOFF_LR_PRIORS.md` — the note-wise L/R-priors plan (metric anchor, velocity accent,
  marginal rewire). **Note:** the ergonomic project supersedes this with a segmental model;
  this handoff applies to the *note-wise* HMM kept here.
- `DRUM_GRAMMAR.md`, `corpus/` — the sticking-grammar corpus seed + schema (the
  circularity/provenance guards live in `corpus/schema.md`).
- `vision_probe.py`, `gold_eval.py`, `hand_landmarker.task`, `probe_out/` — MediaPipe Hands
  audio-anchored which-hand probe and gold-set eval.
- `cli.py`, `server.py`, `static/` — CLI + web front-end.

## Env (Windows)
Shared `../.venv/Scripts/python.exe`; `TF_USE_LEGACY_KERAS=1`, `PYTHONUTF8=1`. Scripts here
reference `../samples/` and the shared venv at the parent root; when reactivating, run from
this directory and adjust any hard-coded `samples/` paths to `../samples/` if needed.
