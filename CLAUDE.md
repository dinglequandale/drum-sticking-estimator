### Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project pointers (drum sticking estimator)

The repo now holds **two separate projects** with different goals. Shared env/data
(`.venv/`, `requirements.txt`, `samples/`) stay at root.

- **`ergonomic_sticking_estimator/` — ACTIVE.** Audio-only: output the most ergonomic /
  idiomatic sticking with calibrated uncertainty (no vision, no labeled corpus). Read
  `ergonomic_sticking_estimator/DESIGN_STAGE5_SEGMENTAL.md` before touching Stage 5 — it's a
  segmental (grouping/template) model that supersedes the note-wise HMM. Stages 0–4 backbone
  is copied here and shared with the vision project; don't diverge it without reason.
- **`vision_sticking_estimator/` — PARKED (safe keeping).** The original problem: recover the
  *actual* sticking using vision + a learned grammar. Full 7-stage pipeline, note-wise HMM,
  `STICKING_ESTIMATOR_SPEC.md`, `HANDOFF_LR_PRIORS.md`, corpus. Don't disturb unless working
  the vision problem specifically.
- **Env quirk (Windows):** ADTOF (transcription backbone) needs `TF_USE_LEGACY_KERAS=1` and
  the venv interpreter `.venv/Scripts/python.exe`; model load is slow, so load once and pass
  `adtof_model=` through `pipeline.run`. Install notes are in `requirements.txt`.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


<!-- curated-memory-pin -->
## Curated memory (durable, project-local)

Curated notes for this project live in `.claude/memory/` — instrumental facts,
preferences, and feedback built up over time. **Before starting work here, read the
memory index below and open any entries relevant to the task.**

@.claude/memory/MEMORY.md

This copy is stored inside the project on purpose: it travels with the folder and
survives switching Claude accounts (main ↔ supp) and any future merge to a single
account. The active account also keeps a live recall-based copy of these notes; when
they diverge, treat this in-project copy as the durable record and re-sync if needed.
