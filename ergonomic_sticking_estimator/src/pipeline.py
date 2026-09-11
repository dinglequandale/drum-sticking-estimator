"""End-to-end pipeline: Stages 0-4 backbone + the segmental Stage 5 decoder.

Audio in -> ingest -> ADTOF onsets/surfaces/velocity -> madmom grid -> consolidate ->
hard constraints -> segmental sticking decode -> readable sticking + flags.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .types import PipelineConfig, AssignedStrike
from .decoder import DecoderConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    assigned: list          # list[AssignedStrike], time-sorted
    beat_info: dict
    sequence: str           # e.g. "R L K K R L? K K"
    grid: str               # per-hit detail table
    was_separated: bool


def run(input_path: str, config: Optional[PipelineConfig] = None,
        dc: Optional[DecoderConfig] = None, separate: object = "auto",
        adtof_model=None, output_dir: Optional[str] = None) -> PipelineResult:
    config = config or PipelineConfig()
    dc = dc or DecoderConfig()

    logger.info("=== Stage 0: Ingest ===")
    from .stage0_ingest import ingest
    wav_path, was_separated = ingest(input_path, separate=separate, output_dir=output_dir)

    logger.info("=== Stage 1: Onsets (ADTOF) ===")
    from .stage1_onsets import detect_events
    strikes = detect_events(wav_path, adtof_model=adtof_model)

    logger.info("=== Stage 1.5: Tom sub-classification ===")
    from .stage1p5_toms import subclassify_toms
    strikes = subclassify_toms(strikes, wav_path, config)

    logger.info("=== Stage 2: Beat grid (madmom) ===")
    from .stage2_beats import add_grid
    strikes, beat_info = add_grid(wav_path, strikes)

    logger.info("=== Stage 3: Consolidation ===")
    from .stage3_consolidate import consolidate
    slots = consolidate(strikes, config)

    logger.info("=== Stage 4: Hard constraints ===")
    from .stage4_constraints import apply_constraints
    constraint_dicts = apply_constraints(slots, config, beat_info)

    logger.info("=== Stage 5: Segmental sticking decode ===")
    from .decoder import decode_stream
    assigned = decode_stream(constraint_dicts, beat_info, config, dc)

    return PipelineResult(
        assigned=assigned, beat_info=beat_info,
        sequence=sticking_sequence(assigned), grid=render_grid(assigned),
        was_separated=was_separated,
    )


def sticking_sequence(assigned: list) -> str:
    """Compact one-line sticking; feet shown as K, ambiguous hits marked '?'."""
    return " ".join(("K" if a.limb == "F" else a.limb) + ("?" if a.flagged_ambiguous else "")
                    for a in assigned)


def render_grid(assigned: list) -> str:
    """Per-hit detail table."""
    lines = [f"{'#':>3} {'time':>7} {'surface':8} {'limb':4} {'conf':>5} flag"]
    for i, a in enumerate(assigned):
        flag = "?" if a.flagged_ambiguous else ""
        lines.append(f"{i:>3} {a.time:7.3f} {a.surface:8} {a.limb:4} {a.confidence:5.2f}  {flag}")
    return "\n".join(lines)
