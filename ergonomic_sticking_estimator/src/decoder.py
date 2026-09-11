"""Stage 5 v2 — semi-Markov (segmental) sticking decoder.

Covers the hand-strike stream with metric cells, dresses each cell in a sticking template
(or the physics-only fallback), and minimizes total cost via a semi-Markov Viterbi:

    Cost = Σ_cells [ C_accent + C_feasible + C_idiom + metric-start ]  +  Σ_seams C_seam

See DESIGN_STAGE5_SEGMENTAL.md. This step adds the decoder + margin/flag on top of the
`scoring.py` terms. `C_reach` and the surface-aware `C_seam` are the step-4 follow-on; here
`C_seam` is a light alternation-continuity term and a metric-start bonus (marked below).

Confidence is honest: per hit, P(hand) comes from comparing the best decode to the best
decode with that hit's hand *forced* to the other value (a constrained re-decode). That
includes seam + segmentation context — so a hit whose hand is pinned by accents/continuity
reads confident, while a genuinely symmetric one (template ties its mirror) reads ~0.5 and
flags. Feet are hard-assigned upstream and pass through as anchors.

Runnable: `python -m src.decoder` runs end-to-end checks on synthetic streams.
"""

import math
from dataclasses import dataclass
from itertools import product

from .types import Strike, AssignedStrike, PipelineConfig, KIT_GEOMETRY
from . import templates as T
from .scoring import dynamic_reference, score_accent, pair_feasibility, speed_fraction


@dataclass
class DecoderConfig:
    w_accent: float = 1.0
    w_feasible: float = 1.0
    w_metric_start: float = 0.5  # bonus for starting a cell on a strong metric position
    w_metric_anchor: float = 0.6 # ABSOLUTE lead=R bias on strong beats — breaks global L<->R symmetry
    w_reach: float = 1.0         # cost of a hand traveling between surfaces (step 4)
    w_side: float = 0.3          # nearest-hand affinity: R favors kit-right, L kit-left (step 4)
    fallback_tax: float = 0.4    # how much better physics-alone must be before abandoning a named cell
    temp: float = 1.0            # softmax temperature for margin -> probability
    max_cell_len: int = 6        # longest span the decoder considers (= longest named template)


_INF = float("inf")


# --- metric position -------------------------------------------------------------------
def _metric_strength(quantized_pos) -> float:
    """Metrical weight of a beat position in [0,1]; strong beats anchor cell starts."""
    if quantized_pos is None:
        return 0.0
    frac = quantized_pos - math.floor(quantized_pos)
    if frac < 0.05 or frac > 0.95:
        return 1.0          # on the beat
    if abs(frac - 0.5) < 0.05:
        return 0.6          # the "&"
    if abs(frac - 0.25) < 0.05 or abs(frac - 0.75) < 0.05:
        return 0.4          # "e"/"a"
    return 0.2


def _lead_bias(quantized_pos) -> int:
    """Whether a position favors the lead hand (+1) under the right-handed convention.

    Only the strong pulse (the integer beat) is anchored to the lead hand — that pins the
    phase without prescribing sticking. Deliberately NEUTRAL (0) on inner subdivisions: an
    'e/a → off-hand' rule would hardcode RLRL single strokes and fight the doubles-at-speed
    physics. Feasibility, not the anchor, decides singles vs. doubles off the beat."""
    if quantized_pos is None:
        return 0
    frac = quantized_pos - math.floor(quantized_pos)
    if frac < 0.05 or frac > 0.95:
        return +1
    return 0


# --- kit geometry / reach (step 4) -----------------------------------------------------
def _pos(surface: str, geom: dict):
    """2D kit position of a surface; unmapped toms fall back to the generic tom, else snare."""
    return geom.get(surface) or geom.get("tom") or (0.0, 0.0)


def _dist(surf_a: str, surf_b: str, geom: dict) -> float:
    ax, ay = _pos(surf_a, geom)
    bx, by = _pos(surf_b, geom)
    return math.hypot(ax - bx, ay - by)


def _reach_travel(dist: float, ioi_sec: float, config: PipelineConfig, dc: DecoderConfig) -> float:
    """Cost of ONE hand moving between two surfaces (a same-hand, different-surface adjacency).
    This is a hand traveling, NOT a rebound double, so the double/run-length logic doesn't apply.
    Two parts: a small speed-independent BASE (using one hand while the other is idle is mildly
    uneconomical — keeps alternation the default and stops one-hand runs across right-side toms)
    plus a speed term (a far move is infeasible when fast). Still cheap enough that a deliberate
    LL-across-two-toms survives when an idiom/accent supports it."""
    return dc.w_reach * dist * (0.25 + 0.75 * speed_fraction(ioi_sec, config))


def _side_affinity(surface: str, hand: str, geom: dict, config: PipelineConfig,
                   dc: DecoderConfig) -> float:
    """Nearest-hand rule: the lead hand favors its own side of the kit, the off-hand the other.
    Per-hit, absolute (like the metric anchor) — needs no hand-position state and both breaks
    L<->R symmetry and encodes crossover. Neutral on center surfaces (snare), so those stay
    honestly ambiguous. Reach is the missing phase cue around the kit."""
    x = _pos(surface, geom)[0]
    lead = "R" if config.handedness == "right" else "L"
    right_hand = lead                     # lead hand lives on kit-right for a right-hander
    if hand == right_hand:
        return dc.w_side * max(0.0, -x)   # penalize the right hand reaching kit-left
    return dc.w_side * max(0.0, x)        # penalize the off hand reaching kit-right


def _anchor_cost(cell: list[Strike], tokens, config: PipelineConfig, dc: DecoderConfig) -> float:
    """Absolute lead=R (right-handed) metric anchor per hit. Breaks the global L<->R symmetry
    that otherwise pins every hit at exactly 0.5. Soft: scaled by metric strength, defeasible."""
    if dc.w_metric_anchor == 0.0:
        return 0.0
    lead = "R" if config.handedness == "right" else "L"
    off = "L" if lead == "R" else "R"
    cost = 0.0
    for strike, hand in zip(cell, tokens):
        bias = _lead_bias(strike.quantized_pos)
        if bias == 0:
            continue
        favored = lead if bias > 0 else off
        strength = _metric_strength(strike.quantized_pos)
        cost += (-dc.w_metric_anchor * strength) if hand == favored else (dc.w_metric_anchor * strength)
    return cost


# --- coverings of a span ---------------------------------------------------------------
def _absolute_cost(cell: list[Strike], tokens, geom: dict, config: PipelineConfig,
                   dc: DecoderConfig) -> float:
    """Per-hit absolute priors that break L<->R symmetry: metric anchor + nearest-hand side
    affinity. Both need no path/segmentation context, so they live in the covering cost."""
    cost = _anchor_cost(cell, tokens, config, dc)
    for strike, hand in zip(cell, tokens):
        cost += _side_affinity(strike.surface, hand, geom, config, dc)
    return cost


def _span_coverings(cell: list[Strike], ref: float, spread: float, geom: dict,
                    config: PipelineConfig, dc: DecoderConfig, forced: dict, offset: int):
    """All (tokens, intrinsic_cost, source) ways to stick this span.

    `intrinsic_cost` = accent + metric-anchor + side-affinity + idiom (all segmentation-internal
    and run-independent). Feasibility + reach are charged per-adjacency by the DP (threaded
    run-length), NOT here — that keeps the score segmentation-invariant. `forced` maps a *global*
    hand-strike index to a required hand; violating coverings are dropped (margin re-decode)."""
    L = len(cell)
    out = []

    def ok(tokens):
        return all(tokens[k - offset] == h for k, h in forced.items() if offset <= k < offset + L)

    # Named templates that exactly cover the span.
    for t in T.templates_of_length(L):
        if not ok(t.tokens):
            continue
        cost = (score_accent(cell, t, ref, spread, dc.w_accent)
                + _absolute_cost(cell, t.tokens, geom, config, dc)
                + t.idiom_cost)
        out.append((t.tokens, cost, t.name))

    # Fallback: any hand sequence, no accent claim, + tax (also discourages trivial over-segmentation).
    if L <= dc.max_cell_len:
        for seq in product("RL", repeat=L):
            if not ok(seq):
                continue
            cost = _absolute_cost(cell, seq, geom, config, dc) + dc.fallback_tax
            out.append((seq, cost, "fallback"))
    return out


def _cell_feasibility(hands, i, j, tokens, prev_hand, prev_run, geom, config, dc):
    """Feasibility + reach over the cell's adjacencies, INCLUDING the entry adjacency from the
    previous cell's exit hand — same per-pair physics whether within-cell or across a seam,
    threading the same-hand run globally. Returns (cost, exit_run).

    Same-hand adjacency splits by surface: SAME surface = rebound double (pair_feasibility);
    DIFFERENT surface = a hand traveling (reach), not a double — so run resets and the double/
    run-length logic does not apply."""
    cost = 0.0
    run = prev_run
    prev_h = prev_hand
    for k in range(i, j):
        h = tokens[k - i]
        if prev_h is not None:
            same_hand = (h == prev_h)
            same_surf = hands[k].surface == hands[k - 1].surface
            ioi = hands[k].time - hands[k - 1].time
            if same_hand and not same_surf:
                run = 1   # a move, not a rebound run
                cost += _reach_travel(_dist(hands[k - 1].surface, hands[k].surface, geom),
                                      ioi, config, dc)
            else:
                run = run + 1 if same_hand else 1
                cost += pair_feasibility(ioi, same_hand, same_surf, run, config, dc.w_feasible)
        else:
            run = 1
        prev_h = h
    return cost, min(run, 3)


# --- the semi-Markov decode ------------------------------------------------------------
def _decode(hands: list[Strike], ref: float, spread: float, config: PipelineConfig,
            dc: DecoderConfig, forced: dict = None):
    """Return (total_cost, path) covering the hand stream. path = list of (start, tokens, source).

    `forced`: optional {global_index: 'R'/'L'} pinning specific hits (for margin re-decodes)."""
    forced = forced or {}
    n = len(hands)
    if n == 0:
        return 0.0, []

    geom = KIT_GEOMETRY.get(f"{config.handedness}_{config.style}", KIT_GEOMETRY["right_crossed"])

    # best[j] : (exit_hand, exit_run) -> (cost, backptr) ; backptr = (i, prev_state, tokens, source)
    best = [dict() for _ in range(n + 1)]
    best[0][(None, 0)] = (0.0, None)

    for j in range(1, n + 1):
        for i in range(max(0, j - dc.max_cell_len), j):
            if not best[i]:
                continue
            cell = hands[i:j]
            metric_bonus = -dc.w_metric_start * _metric_strength(hands[i].quantized_pos)
            coverings = _span_coverings(cell, ref, spread, geom, config, dc, forced, i)
            if not coverings:
                continue
            for tokens, cov_cost, source in coverings:
                exit_h = tokens[-1]
                for (prev_hand, prev_run), (pcost, _) in best[i].items():
                    feas, exit_run = _cell_feasibility(
                        hands, i, j, tokens, prev_hand, prev_run, geom, config, dc)
                    total = pcost + cov_cost + metric_bonus + feas
                    state = (exit_h, exit_run)
                    cur = best[j].get(state)
                    if cur is None or total < cur[0]:
                        best[j][state] = (total, (i, (prev_hand, prev_run), tokens, source))

    # backtrack from the cheapest terminal state
    end_state = min(best[n], key=lambda s: best[n][s][0])
    total_cost = best[n][end_state][0]
    path = []
    j, state = n, end_state
    while j > 0:
        _, back = best[j][state]
        i, prev_state, tokens, source = back
        path.append((i, tokens, source))
        j, state = i, prev_state
    path.reverse()
    return total_cost, path


def _assignment_from_path(n: int, path) -> list[str]:
    """Flatten a path into a per-hand-strike limb list."""
    limbs = [None] * n
    for start, tokens, _ in path:
        for k, tok in enumerate(tokens):
            limbs[start + k] = tok
    return limbs


# --- public entry ----------------------------------------------------------------------
def decode_stream(constraint_dicts: list[dict], beat_info: dict, config: PipelineConfig,
                  dc: DecoderConfig = None) -> list[AssignedStrike]:
    """Assign L/R to every hand strike and F to every kick, returning time-sorted
    AssignedStrikes with honest per-hit confidence and ambiguity flags."""
    dc = dc or DecoderConfig()

    hands: list[Strike] = []
    feet: list[Strike] = []
    for cdict in constraint_dicts:
        for strike in cdict["slot"].strikes:
            (feet if strike.surface == "kick" else hands).append(strike)
    hands.sort(key=lambda s: s.time)

    ref, spread = dynamic_reference(hands)
    _, best_path = _decode(hands, ref, spread, config, dc)
    best_limbs = _assignment_from_path(len(hands), best_path)

    # Honest per-hit confidence: cost of forcing the opposite hand vs. the best decode.
    assigned: list[AssignedStrike] = []
    for k, (strike, limb) in enumerate(zip(hands, best_limbs)):
        cost_R, _ = _decode(hands, ref, spread, config, dc, forced={k: "R"})
        cost_L, _ = _decode(hands, ref, spread, config, dc, forced={k: "L"})
        # P(R) from the cost gap (lower cost = preferred)
        p_r = 1.0 / (1.0 + math.exp((cost_R - cost_L) / dc.temp))
        probs = {"R": p_r, "L": 1.0 - p_r}
        conf = max(probs.values())
        assigned.append(AssignedStrike(
            time=strike.time, surface=strike.surface, velocity=strike.velocity,
            quantized_pos=strike.quantized_pos, limb=limb, limb_probs=probs,
            confidence=conf, source="inference",
            flagged_ambiguous=conf < config.ambiguity_threshold,
        ))

    for strike in feet:
        assigned.append(AssignedStrike(
            time=strike.time, surface=strike.surface, velocity=strike.velocity,
            quantized_pos=strike.quantized_pos, limb="F", limb_probs={"F": 1.0},
            confidence=1.0, source="constraint", flagged_ambiguous=False,
        ))

    assigned.sort(key=lambda a: a.time)
    return assigned


# --- end-to-end smoke checks -----------------------------------------------------------
if __name__ == "__main__":
    cfg = PipelineConfig()

    def stream(spec):
        """spec: list of (surface, velocity, ioi_ms, quantized_pos). Builds constraint_dicts."""
        t = 0.0
        cds = []
        for surf, vel, ioi, qpos in spec:
            t += ioi / 1000.0
            slot = type("S", (), {})()
            slot.time = t
            slot.strikes = [Strike(time=t, surface=surf, velocity=vel, quantized_pos=qpos)]
            cds.append({"slot": slot, "admissible": {}, "local_bpm": 120.0})
        return cds

    def show(name, assigned):
        seq = " ".join(("K" if a.limb == "F" else a.limb) + ("?" if a.flagged_ambiguous else "")
                       for a in assigned)
        hands = [a for a in assigned if a.limb != "F"]
        avg = sum(a.confidence for a in hands) / max(len(hands), 1)
        flg = sum(a.flagged_ambiguous for a in assigned)
        print(f"{name:22} flags={flg} avg_conf={avg:.2f}\n  {seq}")

    # 1) Accented paradiddle on snare: loud on beats 1,2,3,4 -> should read as paradiddles, confident.
    para = []
    for beat in range(4):
        for k, v in enumerate([0.95, 0.4, 0.4, 0.4]):        # accent then three taps
            para.append(("snare", v, 120, beat + k * 0.25))
    show("accented paradiddle", decode_stream(stream(para), {}, cfg))

    # 2) Fast even mono roll (60 ms, flat): ergonomic reading is doubles; phase symmetric -> some flags.
    fast = [("snare", 0.6, 60, (i % 4) * 0.25) for i in range(8)]
    show("fast even roll", decode_stream(stream(fast), {}, cfg))

    # 3) Slow flat alternation on alternating surfaces (a fill around the kit).
    fill = []
    surfs = ["snare", "tom", "snare", "tom", "snare", "tom", "snare", "tom"]
    for i, s in enumerate(surfs):
        fill.append((s, 0.6, 180, (i % 4) * 0.25))
    show("slow surface-fill", decode_stream(stream(fill), {}, cfg))

    # 4) Linear phrase with kicks interleaved (the sample-style skeleton), snare hands.
    #    surfaces: S S K K  S S K K  (hands should alternate across the kicks)
    lin_spec = [("snare", 0.7, 120, 0.0), ("snare", 0.6, 120, 0.25),
                ("kick", 0.9, 120, 0.5), ("kick", 0.9, 120, 0.75),
                ("snare", 0.7, 120, 1.0), ("snare", 0.6, 120, 1.25),
                ("kick", 0.9, 120, 1.5), ("kick", 0.9, 120, 1.75)]
    show("linear w/ kicks", decode_stream(stream(lin_spec), {}, cfg))

    print("\ndecoder smoke checks ran ✓")
