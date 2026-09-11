"""Top-level pipeline: wires all stages together."""

import logging
from dataclasses import dataclass
from typing import Optional

from .types import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    assigned_hits: list
    patterns: list
    beat_info: dict
    json_output: str
    grid_output: str
    pattern_summary: str
    was_separated: bool


def run(
    input_path: str,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[str] = None,
    separate: object = "auto",
    adtof_model=None,
) -> PipelineResult:
    if config is None:
        config = PipelineConfig()

    logger.info("=== Stage 0: Ingest ===")
    from .stage0_ingest import ingest
    wav_path, was_separated = ingest(input_path, separate=separate, output_dir=output_dir)

    logger.info("=== Stage 1: Onsets ===")
    from .stage1_onsets import detect_events
    strikes = detect_events(wav_path, adtof_model=adtof_model)

    logger.info("=== Stage 2: Beat tracking ===")
    from .stage2_beats import add_grid
    strikes, beat_info = add_grid(wav_path, strikes)

    logger.info("=== Stage 3: Event consolidation ===")
    from .stage3_consolidate import consolidate
    slots = consolidate(strikes, config)

    logger.info("=== Stage 4: Hard constraints ===")
    from .stage4_constraints import apply_constraints
    constraint_dicts = apply_constraints(slots, config, beat_info)

    logger.info("=== Stage 5: Limb assignment ===")
    rudiment_prior = None
    if True:  # Tier 1 always on
        from .stage5b_priors import build_tier1_prior
        rudiment_prior = build_tier1_prior(weight_cap=config.rudiment_prior_weight)

    from .stage5_hmm import assign_limbs
    assigned = assign_limbs(constraint_dicts, beat_info, wav_path, config, rudiment_prior)

    if config.self_prior_enabled:
        logger.info("Stage 5b Tier 2: building self-prior (experimental)")
        from .stage5b_priors import build_tier2_prior
        self_prior = build_tier2_prior()
        self_prior.train(assigned)
        # Re-run assignment with self-prior
        assigned = assign_limbs(constraint_dicts, beat_info, wav_path, config,
                                 rudiment_prior=self_prior)

    logger.info("=== Stage 6: Self-similarity ===")
    from .stage6_similarity import propagate
    assigned = propagate(assigned, config)

    logger.info("=== Stage 7: Output ===")
    from .stage7_output import extract_patterns, to_json, to_grid, pattern_summary
    patterns = extract_patterns(assigned)
    json_out = to_json(assigned, patterns, beat_info)
    grid_out = to_grid(assigned, beat_info)
    summary = pattern_summary(patterns)

    return PipelineResult(
        assigned_hits=assigned,
        patterns=patterns,
        beat_info=beat_info,
        json_output=json_out,
        grid_output=grid_out,
        pattern_summary=summary,
        was_separated=was_separated,
    )
