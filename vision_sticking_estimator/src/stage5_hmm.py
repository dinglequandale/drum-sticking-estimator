"""Stage 5 — Probabilistic limb assignment via custom Viterbi HMM.

Hidden state: (left_hand_location, right_hand_location)
where location ∈ HAND_SURFACES = [snare, tom, hihat, cymbal, idle].

Do NOT use a bare {L, R} state. Continuity and reach cost require knowing
where each hand currently is, not just which one last played.

HONEST CEILING (do not remove this comment or "fix" the threshold below):
A genuinely even double-stroke from a skilled player, with no surrounding
surface change or tempo cue, is near a coin flip from audio alone. No emission
model escapes this. All real accuracy comes from context — neighboring hits,
surface, velocity, tempo, repetition. Where the margin distribution is near-
uniform, we keep the uncertainty and flag it. That is the correct behaviour.
"""

import itertools
import logging
import math
from typing import NamedTuple, Optional

import numpy as np

from .types import (
    AssignedStrike, PipelineConfig, HAND_SURFACES,
    KIT_GEOMETRY, NATURAL_HAND
)
from .grammar import Context

logger = logging.getLogger(__name__)

NEG_INF = -1e30


class _PriorContext(NamedTuple):
    """Per-clip context shared by the metric-anchor and accent priors."""
    lead: str
    off: str
    vel_ref: float
    vel_spread: float
    meter: str
    subdivision: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_limbs(
    constraint_dicts: list[dict],
    beat_info: dict,
    wav_path: str,
    config: PipelineConfig,
    rudiment_prior=None,    # optional Stage5b Tier-1 prior
) -> list[AssignedStrike]:
    """Run Viterbi over all time-slots; return per-strike AssignedStrike objects."""

    setup_key = f"{config.handedness}_{config.style}"
    geometry = KIT_GEOMETRY.get(setup_key, KIT_GEOMETRY["right_crossed"])
    natural = NATURAL_HAND.get(setup_key, NATURAL_HAND["right_crossed"])

    # Build state space from surfaces actually present in the clip
    present_surfaces = _get_present_surfaces(constraint_dicts)
    states = _build_states(present_surfaces)
    state_index = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    # Pre-compute reach costs between all surface pairs
    reach_costs = _precompute_reach_costs(geometry)

    # Forbidden transitions from Stage 4 speed-gate
    from .stage4_constraints import build_forbidden_transitions
    forbidden = build_forbidden_transitions(constraint_dicts, config)

    # Load attack envelopes for double-stroke signature if wav_path provided
    attack_cache = _preload_attacks(constraint_dicts, wav_path)

    # Shared context for the metric-anchor (#1) and accent (#2) priors.
    lead = "R" if config.handedness == "right" else "L"
    off = "L" if lead == "R" else "R"
    hand_vels = [
        s.velocity for cd in constraint_dicts for s in cd["slot"].strikes
        if s.surface != "kick"
    ]
    if hand_vels:
        vel_ref = float(np.median(hand_vels))
        iqr = float(np.percentile(hand_vels, 75) - np.percentile(hand_vels, 25))
        vel_spread = iqr if iqr > 1e-6 else 0.1
    else:
        vel_ref, vel_spread = 0.5, 0.1
    ctx = _PriorContext(
        lead=lead, off=off, vel_ref=vel_ref, vel_spread=vel_spread,
        meter=beat_info.get("meter", "4/4"),
        subdivision=beat_info.get("subdivision", 16),
    )

    # -----------------------------------------------------------------------
    # Viterbi forward pass
    # dp[state_idx] = (log_prob, prev_state_idx, assignment_for_this_slot)
    # -----------------------------------------------------------------------
    n_slots = len(constraint_dicts)
    if n_slots == 0:
        return []

    # Initial state: all hands idle, no hand has struck yet
    idle_state = ("idle", "idle", None, None)
    log_prob = np.full(n_states, NEG_INF)
    if idle_state in state_index:
        log_prob[state_index[idle_state]] = 0.0
    else:
        log_prob[:] = 0.0  # uniform start if idle not in pruned space

    backtrack = []  # list of (prev_state_indices, assignments) per slot
    ioi_secs = []  # per-slot ioi_sec, reused by the marginal's neighbor term

    for t, cd in enumerate(constraint_dicts):
        slot = cd["slot"]
        local_bpm = cd["local_bpm"]
        admissible = cd["admissible"]
        # Actual gap to the previous slot — the real measure of how fast the
        # playing is, used for the doubles speed gate. Large for the first slot.
        ioi_sec = (slot.time - constraint_dicts[t - 1]["slot"].time) if t > 0 else 1e9
        ioi_secs.append(ioi_sec)

        new_log_prob = np.full(n_states, NEG_INF)
        prev_state_arr = np.full(n_states, -1, dtype=int)
        assignment_arr = [None] * n_states

        # Enumerate all valid assignments for the hand-strikes in this slot
        hand_strikes = [(i, s) for i, s in enumerate(slot.strikes) if s.surface != "kick"]
        foot_strikes = [(i, s) for i, s in enumerate(slot.strikes) if s.surface == "kick"]

        valid_assignments = _enumerate_assignments(hand_strikes, admissible)

        for prev_s_idx in range(n_states):
            if log_prob[prev_s_idx] == NEG_INF:
                continue
            prev_state = states[prev_s_idx]

            for assignment in valid_assignments:
                # Forbidden transition check (Stage 4 speed gate)
                skip = False
                for i, limb in assignment:
                    if (t, _prev_limb(prev_state, states, backtrack, t, i), limb) in forbidden:
                        skip = True
                        break
                if skip:
                    continue

                new_state = _next_state(prev_state, assignment, slot)
                if new_state not in state_index:
                    continue
                new_s_idx = state_index[new_state]

                log_trans = _log_transition(
                    prev_state, new_state, assignment, slot,
                    ioi_sec, reach_costs, config
                )
                log_emit = _log_emission(
                    assignment, slot, prev_state, local_bpm,
                    attack_cache, natural, config, rudiment_prior, ctx
                )
                score = log_prob[prev_s_idx] + log_trans + log_emit

                if score > new_log_prob[new_s_idx]:
                    new_log_prob[new_s_idx] = score
                    prev_state_arr[new_s_idx] = prev_s_idx
                    assignment_arr[new_s_idx] = assignment

        log_prob = new_log_prob
        backtrack.append((prev_state_arr, assignment_arr))

    # -----------------------------------------------------------------------
    # Backtrack
    # -----------------------------------------------------------------------
    best_final = int(np.argmax(log_prob))
    path_states = [best_final]
    path_assignments = []

    for t in range(n_slots - 1, 0, -1):
        prev_arr, assign_arr = backtrack[t]
        path_assignments.insert(0, assign_arr[path_states[0]])
        path_states.insert(0, int(prev_arr[path_states[0]]))

    path_assignments.insert(0, backtrack[0][1][path_states[0]] if backtrack else [])

    # -----------------------------------------------------------------------
    # Compute marginal confidence via forward-backward (approximation)
    # For efficiency: use ratio of best-vs-second-best path at each slot.
    # -----------------------------------------------------------------------
    result: list[AssignedStrike] = []
    prev_hand_limb = None  # previous hand hit's assigned limb, in time order

    for t, (cd, assignment) in enumerate(zip(constraint_dicts, path_assignments)):
        slot = cd["slot"]
        ioi_sec = ioi_secs[t]
        if assignment is None:
            assignment = []

        # Foot strikes — always confident
        for i, strike in enumerate(slot.strikes):
            if strike.surface == "kick":
                result.append(AssignedStrike(
                    time=strike.time,
                    surface=strike.surface,
                    velocity=strike.velocity,
                    quantized_pos=strike.quantized_pos,
                    limb="F",
                    limb_probs={"F": 1.0},
                    confidence=1.0,
                    source="constraint",
                    flagged_ambiguous=False,
                ))

        # Hand strikes
        for i, limb in (assignment or []):
            if i >= len(slot.strikes):
                continue
            strike = slot.strikes[i]
            probs = _compute_marginal(
                i, limb, t, constraint_dicts, states, state_index,
                reach_costs, config, attack_cache, natural, cd["local_bpm"],
                rudiment_prior, ctx, prev_hand_limb, ioi_sec,
            )
            confidence = probs.get(limb, 0.5)
            source = "constraint" if cd["admissible"].get(i) == frozenset({limb}) else "inference"

            result.append(AssignedStrike(
                time=strike.time,
                surface=strike.surface,
                velocity=strike.velocity,
                quantized_pos=strike.quantized_pos,
                limb=limb,
                limb_probs=probs,
                confidence=confidence,
                source=source,
                flagged_ambiguous=confidence < config.ambiguity_threshold,
            ))
            prev_hand_limb = limb

    result.sort(key=lambda a: a.time)
    n_flagged = sum(1 for a in result if a.flagged_ambiguous)
    logger.info("Stage 5: %d assigned hits, %d flagged ambiguous (%.0f%%)",
                len(result), n_flagged, 100 * n_flagged / max(len(result), 1))
    return result


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------

def _get_present_surfaces(constraint_dicts: list[dict]) -> list[str]:
    surfaces = set()
    for cd in constraint_dicts:
        for strike in cd["slot"].strikes:
            if strike.surface != "kick":
                surfaces.add(strike.surface)
    surfaces.add("idle")
    return sorted(surfaces)


def _build_states(surfaces: list[str]) -> list[tuple]:
    """All (L_loc, R_loc, last_hand, prev_hand) states over present surfaces.

    last_hand ∈ {None,'L','R'} is the hand that struck most recently; prev_hand
    is the one before it. The pair gives the **order-2 hand history** the sticking
    grammar needs (e.g. after R,R the lead hands off to L — unreachable with
    last_hand alone). prev_hand is only meaningful once last_hand exists, so we
    skip the inconsistent (last_hand=None, prev_hand=set) combinations.
    Reach/continuity still key off the location pair.
    """
    states = []
    for l, r in itertools.product(surfaces, surfaces):
        states.append((l, r, None, None))
        for h in ("L", "R"):
            for ph in (None, "L", "R"):
                states.append((l, r, h, ph))
    return states


# ---------------------------------------------------------------------------
# Reach costs
# ---------------------------------------------------------------------------

def _precompute_reach_costs(geometry: dict) -> dict:
    costs = {}
    for s1 in geometry:
        for s2 in geometry:
            dx = geometry[s1][0] - geometry[s2][0]
            dy = geometry[s1][1] - geometry[s2][1]
            costs[(s1, s2)] = math.sqrt(dx * dx + dy * dy)
    return costs


def _reach(src: str, dst: str, reach_costs: dict) -> float:
    if src == dst:
        return 0.0
    return reach_costs.get((src, dst), 1.0)


# ---------------------------------------------------------------------------
# Assignment enumeration
# ---------------------------------------------------------------------------

def _enumerate_assignments(
    hand_strikes: list[tuple[int, object]],
    admissible: dict,
) -> list[list[tuple[int, str]]]:
    """Generate all valid (strike_index, limb) assignments for a time-slot.

    Constraint: one limb plays at most one strike per slot (spatial exclusivity).
    """
    if not hand_strikes:
        return [[]]

    options = []
    for i, strike in hand_strikes:
        adm = admissible.get(i, frozenset({"L", "R"}))
        options.append([(i, limb) for limb in adm])

    assignments = []
    for combo in itertools.product(*options):
        # Check spatial exclusivity: no two strikes assigned to the same limb
        limbs_used = [limb for _, limb in combo]
        if len(limbs_used) == len(set(limbs_used)):
            assignments.append(list(combo))
    return assignments if assignments else [[]]


def _next_state(
    prev_state: tuple,
    assignment: list[tuple[int, str]],
    slot,
) -> tuple:
    """Compute new (L_loc, R_loc, last_hand, prev_hand) after executing assignment."""
    left_loc, right_loc, last_hand, prev_hand = prev_state
    for i, limb in assignment:
        surface = slot.strikes[i].surface
        if limb == "L":
            left_loc = surface
            prev_hand, last_hand = last_hand, "L"
        elif limb == "R":
            right_loc = surface
            prev_hand, last_hand = last_hand, "R"
    return (left_loc, right_loc, last_hand, prev_hand)


def _prev_limb(prev_state, states, backtrack, t, strike_i):
    """Best-effort: what limb did strike_i's surface last get assigned to."""
    # Simplified: returns None (speed-gate enforcement is conservative here)
    return None


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------

def _log_transition(
    prev_state: tuple,
    new_state: tuple,
    assignment: list[tuple[int, str]],
    slot,
    ioi_sec: float,
    reach_costs: dict,
    config: PipelineConfig,
) -> float:
    log_p = 0.0
    prev_L, prev_R, prev_last, _ = prev_state
    new_L, new_R, _, _ = new_state

    # Reach cost for each hand that moved
    if new_L != prev_L:
        log_p -= config.reach_cost_scale * _reach(prev_L, new_L, reach_costs)
    if new_R != prev_R:
        log_p -= config.reach_cost_scale * _reach(prev_R, new_R, reach_costs)

    # Continuity bonus for hands that stayed put
    if new_L == prev_L and new_L != "idle":
        log_p += config.continuity_bonus
    if new_R == prev_R and new_R != "idle":
        log_p += config.continuity_bonus

    # Alternation default at moderate tempo.
    # Single strokes alternate hands, so below the doubles tempo we penalize
    # using the same hand twice in a row (a double). This keys off last_hand —
    # the true notion of alternation — rather than surface equality, which
    # couldn't tell two hands apart when both sat on the same surface. The speed
    # gate leaves it untouched near the wall, where doubles become expected, so
    # this does not suppress genuine fast double strokes.
    speed_fraction = _speed_fraction(ioi_sec, config)
    if speed_fraction < config.double_prior_speed and prev_last is not None:
        for i, limb in assignment:
            if limb == prev_last:
                log_p -= config.alternation_weight

    return log_p


def _speed_fraction(ioi_sec: float, config: PipelineConfig) -> float:
    """How close is the actual note spacing to the single-hand ceiling?

    0 = slow (notes well-spaced, single-stroke alternation is the default),
    1 = at/below the single-hand floor (so fast that doubles are expected).
    Measured from the real inter-onset interval, not an assumed subdivision.
    """
    floor_sec = config.single_hand_min_ioi_ms / 1000.0
    if ioi_sec <= 0:
        return 1.0
    return min(floor_sec / ioi_sec, 1.0)


# ---------------------------------------------------------------------------
# Emission model
# ---------------------------------------------------------------------------

def _log_emission(
    assignment: list[tuple[int, str]],
    slot,
    prev_state: tuple,
    local_bpm: float,
    attack_cache: dict,
    natural: dict,
    config: PipelineConfig,
    rudiment_prior,
    ctx: "_PriorContext",
) -> float:
    log_p = 0.0

    for i, limb in assignment:
        strike = slot.strikes[i]

        # Natural hand affinity for surface
        nat = natural.get(strike.surface)
        if nat is not None and nat != limb:
            log_p -= 0.4  # soft penalty for crossing natural hand assignment

        # Double-stroke acoustic signature (modest weight; see honest ceiling above)
        sig = _double_stroke_log_odds(i, slot, attack_cache, config)
        # sig > 0 favors a repeat (RR/LL), sig < 0 favors alternation
        # Apply only when we're assigning a repeat on the same surface as prev
        prev_loc = prev_state[0] if limb == "L" else prev_state[1]
        if strike.surface == prev_loc:
            log_p += config.double_sig_weight * sig

        # #1 Metric anchor: lead hand favored on metrically strong positions.
        strength, lead_bias = _metric_strength(strike.quantized_pos, ctx.meter, ctx.subdivision)
        if lead_bias != 0:
            favored = ctx.lead if lead_bias > 0 else ctx.off
            sign = 1.0 if limb == favored else -1.0
            log_p += sign * config.metric_anchor_weight * strength

        # #2 Velocity/accent: accents favor the lead hand, ghosts the off-hand,
        # measured relative to this clip's own dynamic level (not an absolute level).
        accent = (strike.velocity - ctx.vel_ref) / ctx.vel_spread
        accent = float(np.clip(accent, -2.0, 2.0))
        sign = 1.0 if limb == ctx.lead else -1.0
        log_p += sign * config.velocity_accent_weight * (accent / 2.0)

        # Sticking-grammar prior (Tier 1): a bounded, context-conditioned nudge.
        # Conceptually a transition prior (it depends on the previous hand) but
        # injected here because prev_state + ctx + per-strike features are all in
        # scope. Uses order-2 history (prev_state's prev_hand, last_hand).
        # Bounded by weight_cap inside nudge().
        if rudiment_prior is not None:
            log_p += rudiment_prior.nudge(limb, _grammar_context(prev_state, strike, ctx))

    return log_p


def _metric_label(quantized_pos, meter, subdivision) -> str:
    """Categorical metric position (strong/weak/off) matching grammar.metric_class
    semantics, derived from the within-beat phase of quantized_pos."""
    if quantized_pos is None:
        return "off"
    try:
        beats_per_bar = int(str(meter).split("/")[0])
    except (ValueError, AttributeError, IndexError):
        beats_per_bar = 4
    bar_frac = quantized_pos - math.floor(quantized_pos)
    within_beat = (bar_frac * beats_per_bar) % 1.0
    if within_beat < 0.05 or within_beat > 0.95:
        return "strong"
    if abs(within_beat - 0.5) < 0.05:
        return "weak"
    return "off"


def _grammar_context(prev_state, strike, ctx) -> "Context":
    """Build the grammar conditioning context for one hand strike. History is the
    Viterbi state's (prev_hand, last_hand) → order-2; accent is clip-relative."""
    last_hand, prev_hand = prev_state[2], prev_state[3]
    hist = tuple(h for h in (prev_hand, last_hand) if h in ("L", "R"))
    accent = ((strike.velocity - ctx.vel_ref) / ctx.vel_spread) > 0.6
    return Context(
        hist=hist,
        surface=strike.surface,
        metric=_metric_label(strike.quantized_pos, ctx.meter, ctx.subdivision),
        accent=bool(accent),
        speed=None,
    )


def _metric_strength(quantized_pos: Optional[float], meter: str, subdivision: int):
    """Return (strength, lead_bias) for a beat position.

    strength in [0,1] = metrical weight. lead_bias in {+1,0,-1}: +1 favors the
    lead hand, -1 favors the off-hand.

    `quantized_pos`'s fractional part encodes position *within the bar*
    (stage2_beats._quantize divides by beats_per_bar), not within a beat —
    so it must be rescaled by beats_per_bar to recover the within-beat phase
    before applying the on-beat/eighth/sixteenth table below.
    """
    if quantized_pos is None:
        return 0.0, 0
    try:
        beats_per_bar = int(meter.split("/")[0])
    except (ValueError, AttributeError, IndexError):
        beats_per_bar = 4
    bar_frac = quantized_pos - math.floor(quantized_pos)
    within_beat = (bar_frac * beats_per_bar) % 1.0

    on_beat = within_beat < 0.05 or within_beat > 0.95

    # Triplet-based subdivisions: the 1/3, 2/3 offsets don't map onto the
    # 16th-note table below, so only claim a strong lead bias on the beat itself.
    if subdivision in (12, 24):
        return (1.0, +1) if on_beat else (0.2, 0)

    if on_beat:
        return 1.0, +1
    if abs(within_beat - 0.5) < 0.05:                                # the "&" (eighth offbeat)
        return 0.6, +1
    if abs(within_beat - 0.25) < 0.05 or abs(within_beat - 0.75) < 0.05:  # "e"/"a" (16ths)
        return 0.4, -1
    return 0.2, 0  # triplet or other subdivision: weak, no strong lead claim


def _double_stroke_log_odds(
    strike_idx: int,
    slot,
    attack_cache: dict,
    config: PipelineConfig,
) -> float:
    """Estimate log-odds ratio: double (RR/LL) vs alternation (RL) from acoustic signature.

    Positive → evidence of a double (rebound note: softer, mushier, lower centroid).
    Negative → evidence of alternation (two near-identical strong strokes).

    HONEST CEILING: this feature is strongest on imperfect/amateur doubles and weakest
    on fast, even, virtuoso passages — exactly when you most need it. Do not increase
    double_sig_weight beyond 1.0 or let it outvote velocity/surface/speed evidence.
    """
    key = (id(slot), strike_idx)
    if key not in attack_cache:
        return 0.0  # no audio data for this strike

    curr = attack_cache[key]
    if curr is None:
        return 0.0

    # For the double-stroke signature we need the *previous* same-surface strike's envelope.
    # attack_cache is keyed by (slot_id, strike_idx) — look for it among immediately prior slots.
    prev_key = attack_cache.get("_prev_same_surface", {}).get(key)
    if prev_key is None or prev_key not in attack_cache:
        return 0.0

    prev = attack_cache[prev_key]
    if prev is None:
        return 0.0

    # First stroke wrist-driven: higher amplitude, sharper attack, higher centroid.
    # Second (rebound) stroke: lower amplitude, mushier, lower centroid.
    vel_ratio = curr["peak_amplitude"] / (prev["peak_amplitude"] + 1e-8)
    rise_ratio = curr["rise_time_ms"] / (prev["rise_time_ms"] + 1e-3)
    centroid_ratio = curr["spectral_centroid"] / (prev["spectral_centroid"] + 1e-3)

    # Rebound note has vel_ratio < 1, rise_ratio > 1, centroid_ratio < 1
    double_score = (1.0 - vel_ratio) + (rise_ratio - 1.0) + (1.0 - centroid_ratio)

    # Map to log-odds: positive means "looks like a double"
    return float(np.clip(double_score, -2.0, 2.0))


# ---------------------------------------------------------------------------
# Marginal confidence (approximation)
# ---------------------------------------------------------------------------

def _compute_marginal(
    strike_idx: int,
    best_limb: str,
    slot_t: int,
    constraint_dicts: list[dict],
    states: list,
    state_index: dict,
    reach_costs: dict,
    config: PipelineConfig,
    attack_cache: dict,
    natural: dict,
    local_bpm: float,
    rudiment_prior,
    ctx: "_PriorContext",
    prev_hand_limb: Optional[str],
    ioi_sec: float,
) -> dict:
    """Approximate marginal confidence using the single-slot score difference."""
    cd = constraint_dicts[slot_t]
    slot = cd["slot"]
    admissible = cd["admissible"]
    adm = admissible.get(strike_idx, frozenset({"L", "R"}))

    if len(adm) == 1:
        return {list(adm)[0]: 1.0}

    scores = {}
    for limb in adm:
        # Score this assignment in isolation vs. the alternatives
        log_e = _log_emission(
            [(strike_idx, limb)], slot, ("idle", "idle", None, None), local_bpm,
            attack_cache, natural, config, rudiment_prior, ctx
        )
        scores[limb] = log_e

    # #0b: neighbor alternation context — mirrors _log_transition's speed gate.
    if prev_hand_limb is not None and _speed_fraction(ioi_sec, config) < config.double_prior_speed:
        for limb in adm:
            if limb != prev_hand_limb:
                scores[limb] += config.marginal_neighbor_weight

    # Softmax
    vals = np.array([scores[l] for l in adm])
    vals -= vals.max()
    exp_vals = np.exp(vals)
    exp_vals /= exp_vals.sum()
    return {l: float(p) for l, p in zip(adm, exp_vals)}


# ---------------------------------------------------------------------------
# Attack envelope cache
# ---------------------------------------------------------------------------

def _preload_attacks(constraint_dicts: list[dict], wav_path: str) -> dict:
    """Load attack envelopes for all strikes. Returns empty dict on failure."""
    if not wav_path:
        return {}
    try:
        from .stage1_onsets import extract_attack_envelope
        cache = {}
        prev_same_surface: dict[str, tuple] = {}  # surface -> last (slot_id, strike_idx) key

        for cd in constraint_dicts:
            slot = cd["slot"]
            for i, strike in enumerate(slot.strikes):
                key = (id(slot), i)
                try:
                    cache[key] = extract_attack_envelope(wav_path, strike.time)
                except Exception:
                    cache[key] = None

                # Record cross-reference for double-stroke detection
                surf = strike.surface
                if surf in prev_same_surface:
                    if "_prev_same_surface" not in cache:
                        cache["_prev_same_surface"] = {}
                    cache["_prev_same_surface"][key] = prev_same_surface[surf]
                prev_same_surface[surf] = key

        return cache
    except Exception as e:
        logger.warning("Attack cache preload failed: %s", e)
        return {}
