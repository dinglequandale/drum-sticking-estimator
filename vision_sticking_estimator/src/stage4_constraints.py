"""Stage 4 — Hard physical constraints: prune the assignment hypothesis space.

These are inviolable physics. Apply them before probabilistic ranking.
The output per strike is a reduced set of admissible limb labels (often
a constraint already collapses it to one).
"""

import logging
from collections import defaultdict

from .types import Strike, TimeSlot, PipelineConfig, LIMBS

logger = logging.getLogger(__name__)

# Admissible labels shrink here; the HMM never considers pruned labels.
AdmissibleMap = dict  # str(strike_id) -> frozenset of admissible limbs


def apply_constraints(
    slots: list[TimeSlot],
    config: PipelineConfig,
    beat_info: dict,
) -> list[dict]:
    """Return one constraint-dict per TimeSlot.

    Each dict maps strike index (within slot) to frozenset of admissible limb labels.
    Also annotates strikes that the constraint alone resolved (source='constraint').
    """
    from .stage2_beats import get_local_tempo

    result = []
    # Track per-limb last-use time for the recovery-interval constraint
    last_used: dict[str, float] = {"L": -999.0, "R": -999.0, "F": -999.0}

    for slot_idx, slot in enumerate(slots):
        slot_admissible: dict[int, frozenset] = {}
        local_bpm = get_local_tempo(beat_info, slot.time)
        local_ioi_min = 60.0 / (local_bpm * 4) * 1000  # 16th note duration in ms

        for i, strike in enumerate(slot.strikes):
            admissible = _initial_admissible(strike)

            # Constraint 3: kick → foot
            if strike.surface == "kick":
                admissible = frozenset({"F"})
                slot_admissible[i] = admissible
                continue

            # Constraint 2: spatial exclusivity within chord
            # (handled at assignment-enumeration time in Stage 5 — mark here as reminder)

            # Constraint 1: single-hand stroke-rate ceiling
            # Check preceding same-surface strikes for IOI violations
            admissible = _apply_speed_gate(
                strike, slot, slots, slot_idx, admissible, config, last_used
            )

            # Constraint 4: limb continuity (recovery interval)
            admissible = _apply_recovery_constraint(
                strike, admissible, last_used, config
            )

            slot_admissible[i] = admissible

        result.append({
            "admissible": slot_admissible,
            "slot": slot,
            "local_bpm": local_bpm,
        })

    return result


def _initial_admissible(strike: Strike) -> frozenset:
    """Every hand strike starts with {L, R} as candidates."""
    return frozenset({"L", "R"})


def _apply_speed_gate(
    strike: Strike,
    current_slot: TimeSlot,
    all_slots: list[TimeSlot],
    slot_idx: int,
    admissible: frozenset,
    config: PipelineConfig,
    last_used: dict,
) -> frozenset:
    """If two same-surface hits are closer than single_hand_min_ioi, they must differ."""
    min_ioi = config.single_hand_min_ioi_ms / 1000.0
    double_band = config.double_buzz_band_ms / 1000.0

    # Find the most recent prior same-surface strike
    for prev_slot in reversed(all_slots[:slot_idx]):
        for prev_strike in prev_slot.strikes:
            if prev_strike.surface != strike.surface:
                continue
            ioi = current_slot.time - prev_slot.time
            if ioi > double_band:
                break  # far enough apart; no constraint
            if ioi < min_ioi:
                # Must be two different hands: if we know the previous hand, we can constrain this one.
                # (Previous hand assignment happens in Stage 5; here we just flag the situation.)
                # Mark as "must differ from prior" — Stage 5 will enforce via state transition.
                logger.debug(
                    "Speed gate: %.1f ms IOI on %s at %.3fs — must be two hands",
                    ioi * 1000, strike.surface, current_slot.time,
                )
            # In the double-band: flag as possible double/buzz — don't collapse to two-hands.
            break

    return admissible  # pruning of specific labels happens in Stage 5 given the prev assignment


def _apply_recovery_constraint(
    strike: Strike,
    admissible: frozenset,
    last_used: dict,
    config: PipelineConfig,
) -> frozenset:
    """Remove limbs that haven't recovered yet — very conservative; mainly for dense chords."""
    min_recovery = config.single_hand_min_ioi_ms / 1000.0
    # This is a soft gate; actual enforcement happens in Stage 5's forbidden-transition table.
    return admissible


def build_forbidden_transitions(
    constraint_dicts: list[dict],
    config: PipelineConfig,
) -> set[tuple]:
    """Return a set of (slot_idx, prev_limb, limb) triples that are physically forbidden.

    Used by the HMM to zero out impossible Viterbi transitions.
    """
    forbidden = set()
    min_ioi = config.single_hand_min_ioi_ms / 1000.0

    for i in range(1, len(constraint_dicts)):
        curr = constraint_dicts[i]
        prev = constraint_dicts[i - 1]
        ioi = curr["slot"].time - prev["slot"].time

        if ioi >= min_ioi:
            continue

        # If same-surface consecutive hit within speed gate → same limb is forbidden
        for ci, cs in enumerate(curr["slot"].strikes):
            for pi, ps in enumerate(prev["slot"].strikes):
                if cs.surface == ps.surface:
                    for limb in ("L", "R"):
                        forbidden.add((i, limb, limb))

    return forbidden
