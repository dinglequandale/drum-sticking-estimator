"""Stage 3 — Event consolidation: cluster simultaneous strikes into TimeSlots.

An onset is not a limb. Two surfaces hit within the coincidence window are a
chord — two limbs at once — not a sequence. Treating them as sequential corrupts
the Viterbi state space downstream. This stage must not be skipped.
"""

import logging

import numpy as np

from .types import Strike, TimeSlot, PipelineConfig

logger = logging.getLogger(__name__)


def consolidate(strikes: list[Strike], config: PipelineConfig) -> list[TimeSlot]:
    """Cluster strikes within coincidence_window_ms into TimeSlots."""
    if not strikes:
        return []

    window_sec = config.coincidence_window_ms / 1000.0
    slots: list[TimeSlot] = []
    current_group: list[Strike] = [strikes[0]]

    for strike in strikes[1:]:
        # Compare to the earliest strike in the current group
        if strike.time - current_group[0].time <= window_sec:
            current_group.append(strike)
        else:
            slots.append(_make_slot(current_group))
            current_group = [strike]

    slots.append(_make_slot(current_group))

    n_chords = sum(1 for s in slots if len(s.strikes) > 1)
    logger.info("Consolidated %d strikes → %d time-slots (%d chords)",
                len(strikes), len(slots), n_chords)
    return slots


def _make_slot(group: list[Strike]) -> TimeSlot:
    mean_time = float(np.mean([s.time for s in group]))
    return TimeSlot(time=mean_time, strikes=list(group))
