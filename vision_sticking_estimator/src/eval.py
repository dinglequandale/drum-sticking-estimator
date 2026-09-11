"""Evaluation harness — build early, not last.

Metrics:
  1. Onset F-measure (validates Stages 1-3 independently of assignment).
  2. Limb-assignment accuracy on non-flagged hits.
  3. Flag precision/recall — when the model flags 'ambiguous', is it actually hard?
  4. Tier breakdown: easy / medium / hard.
  5. Dual-run guard for Stage 5b priors (prior ON vs OFF, accuracy AND flag-rate).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ONSET_TOLERANCE_SEC = 0.025  # 25 ms window for onset matching


@dataclass
class EvalMetrics:
    onset_precision: float = 0.0
    onset_recall: float = 0.0
    onset_f1: float = 0.0
    assignment_accuracy: float = 0.0  # on non-flagged hits only
    flag_precision: float = 0.0       # flagged hits that were truly hard
    flag_recall: float = 0.0          # truly hard hits that were flagged
    n_evaluated: int = 0
    n_flagged: int = 0
    tier: str = "unknown"             # easy / medium / hard


@dataclass
class Annotation:
    time: float
    limb: str    # L / R / F
    tier: str    # easy / medium / hard
    surface: Optional[str] = None


def load_annotations(path: str) -> list[Annotation]:
    """Load hand-annotated ground-truth from a JSON file.

    Expected format:
    [{"time": 1.234, "limb": "L", "tier": "easy", "surface": "snare"}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [Annotation(**d) for d in data]


def evaluate(
    result,                          # PipelineResult
    annotations: list[Annotation],
    tier: Optional[str] = None,      # filter to easy/medium/hard if set
) -> EvalMetrics:
    """Compute all metrics for a single clip."""
    assigned = result.assigned_hits
    if tier:
        annotations = [a for a in annotations if a.tier == tier]
        assigned = [h for h in assigned if _nearest_annotation(h.time, annotations) is not None]

    try:
        import mir_eval
        ref_times = np.array([a.time for a in annotations])
        est_times = np.array([h.time for h in assigned])
        p, r, f = mir_eval.onset.f_measure(ref_times, est_times, window=ONSET_TOLERANCE_SEC)
    except ImportError:
        logger.warning("mir_eval not available; skipping onset F-measure")
        p, r, f = 0.0, 0.0, 0.0

    # Assignment accuracy — exclude flagged hits
    non_flagged = [h for h in assigned if not h.flagged_ambiguous]
    correct = 0
    evaluated = 0
    for hit in non_flagged:
        ann = _nearest_annotation(hit.time, annotations)
        if ann is None:
            continue
        evaluated += 1
        if hit.limb == ann.limb:
            correct += 1

    accuracy = correct / evaluated if evaluated > 0 else 0.0

    # Flag precision/recall — "truly hard" = two annotations within 50ms differ on limb
    truly_hard = _find_truly_hard(annotations)
    flagged_times = {h.time for h in assigned if h.flagged_ambiguous}
    n_flagged = len(flagged_times)

    flag_tp = sum(1 for t in truly_hard if any(abs(t - ft) < ONSET_TOLERANCE_SEC for ft in flagged_times))
    flag_precision = flag_tp / n_flagged if n_flagged > 0 else 1.0
    flag_recall = flag_tp / len(truly_hard) if truly_hard else 1.0

    return EvalMetrics(
        onset_precision=p,
        onset_recall=r,
        onset_f1=f,
        assignment_accuracy=accuracy,
        flag_precision=flag_precision,
        flag_recall=flag_recall,
        n_evaluated=evaluated,
        n_flagged=n_flagged,
        tier=tier or "all",
    )


def dual_run_eval(
    input_path: str,
    annotations: list[Annotation],
    config=None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run the pipeline twice — with and without Stage 5b priors — and compare.

    Decision rule: a prior is admitted only if it raises accuracy, especially
    on off-vocabulary material. If it shrinks flag_rate without raising accuracy,
    it is manufacturing false confidence and must be rejected.
    """
    from .pipeline import run
    from .types import PipelineConfig

    if config is None:
        config = PipelineConfig()

    # Run 1: Tier-1 prior on, Tier-2 off (default)
    config_on = PipelineConfig(**config.__dict__)
    config_on.self_prior_enabled = False
    result_on = run(input_path, config=config_on, output_dir=output_dir)
    metrics_on = evaluate(result_on, annotations)

    # Run 2: all priors off
    config_off = PipelineConfig(**config.__dict__)
    config_off.rudiment_prior_weight = 0.0
    config_off.self_prior_enabled = False
    result_off = run(input_path, config=config_off, output_dir=output_dir)
    metrics_off = evaluate(result_off, annotations)

    decision = _admit_prior(metrics_on, metrics_off)
    logger.info(
        "Dual-run: prior_on acc=%.3f flags=%d | prior_off acc=%.3f flags=%d → %s",
        metrics_on.assignment_accuracy, metrics_on.n_flagged,
        metrics_off.assignment_accuracy, metrics_off.n_flagged,
        decision,
    )
    return {
        "prior_on": metrics_on,
        "prior_off": metrics_off,
        "decision": decision,
    }


def _admit_prior(on: EvalMetrics, off: EvalMetrics) -> str:
    """Return 'admit' or 'reject' with reason."""
    acc_delta = on.assignment_accuracy - off.assignment_accuracy
    flag_delta = on.n_flagged - off.n_flagged  # negative = fewer flags with prior on

    if acc_delta > 0.01:
        return f"admit (accuracy +{acc_delta:.3f})"
    if acc_delta <= 0 and flag_delta < -2:
        return f"reject (shrinks flag_rate by {-flag_delta} without accuracy gain — false confidence)"
    return f"neutral (accuracy delta={acc_delta:.3f}); keep off by default"


def report_all_tiers(
    result,
    annotations: list[Annotation],
) -> str:
    lines = ["Evaluation report"]
    lines.append(f"  Total hits: {len(result.assigned_hits)}")
    for tier in ["all", "easy", "medium", "hard"]:
        m = evaluate(result, annotations, tier=tier if tier != "all" else None)
        lines.append(
            f"  [{m.tier:6}] onset_F1={m.onset_f1:.3f}  "
            f"assign_acc={m.assignment_accuracy:.3f} ({m.n_evaluated} hits)  "
            f"flag_prec={m.flag_precision:.3f} flag_rec={m.flag_recall:.3f} ({m.n_flagged} flagged)"
        )
    return "\n".join(lines)


def _nearest_annotation(time: float, annotations: list[Annotation]) -> Optional[Annotation]:
    if not annotations:
        return None
    times = np.array([a.time for a in annotations])
    idx = int(np.argmin(np.abs(times - time)))
    if abs(times[idx] - time) <= ONSET_TOLERANCE_SEC:
        return annotations[idx]
    return None


def _find_truly_hard(annotations: list[Annotation]) -> list[float]:
    """Times where manual annotators would plausibly disagree (placeholder heuristic)."""
    # A real implementation would use inter-annotator agreement on the test set.
    # Here: flag as hard any annotation marked tier='hard'.
    return [a.time for a in annotations if a.tier == "hard"]
