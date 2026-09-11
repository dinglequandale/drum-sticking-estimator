"""Stage 5 v2 — per-cell physical cost terms (corpus-free).

Scores how well a Template fits a candidate cell (a contiguous span of *hand* strikes).
Lower cost = better fit. These are the two backbone terms from DESIGN_STAGE5_SEGMENTAL.md
§2; C_reach / C_seam / C_idiom are added later at the decoder level.

    C_accent   — does the template's accent pattern match the observed dynamics?
                 (accents ride the lead hand → this selects the sticking AND its phase)
    C_feasible — is each same-/different-hand pair physically economical at this speed?
                 (fast same-surface → doubles cheap, singles costly → "buzzy = RRLL")

Runnable: `python -m src.scoring` from the project dir runs the unit checks.
"""

import numpy as np

from .types import Strike, PipelineConfig

# Accent codes (mirror templates.py without importing it — keeps this module light)
ACCENT, NORMAL, GHOST = 1, 0, -1


# --- dynamics -------------------------------------------------------------------------
def dynamic_reference(strikes: list[Strike]) -> tuple[float, float]:
    """Local dynamic level (median) and spread (IQR) over hand strikes.
    Accent-vs-ghost is *relative* to the local level, not the clip peak — so we bias by
    deviation from this reference. Returns (ref, spread); spread floored to avoid blowup."""
    vels = [s.velocity for s in strikes]
    if not vels:
        return 0.5, 0.1
    ref = float(np.median(vels))
    spread = float(np.percentile(vels, 75) - np.percentile(vels, 25))
    return ref, (spread or 0.1)


def _accent_level(velocity: float, ref: float, spread: float) -> float:
    """Signed, clipped deviation from the local level. >0 accent, <0 ghost, ~0 normal."""
    return float(np.clip((velocity - ref) / spread, -2.0, 2.0))


# --- C_accent -------------------------------------------------------------------------
def score_accent(cell: list[Strike], template, ref: float, spread: float,
                 weight: float = 1.0) -> float:
    """Cost of dressing `cell` in `template` given the accent pattern mismatch.

    Compares each template slot's target level τ ∈ {+1,0,-1} to the observed local accent
    level. A cleanly accented cell locks onto the matching template (and its phase); a flat
    cell leaves accent-bearing templates no better than the plain single/double → the tie
    that must survive as a flag."""
    if template.is_fallback:
        return 0.0  # fallback makes no dynamic claim
    obs = [_accent_level(s.velocity, ref, spread) for s in cell]
    cost = sum(abs(tau - a) for tau, a in zip(template.accents, obs))
    return weight * cost


# --- C_feasible -----------------------------------------------------------------------
def speed_fraction(ioi_sec: float, config: PipelineConfig) -> float:
    """0 = slow (comfortable singles), 1 = at the single-hand floor (doubles expected).

    Below the floor we clamp to 1; above a 'comfortable singles' reference (2.5x the floor)
    we clamp to 0; linear between. Drives the singles<->doubles tradeoff."""
    floor = config.single_hand_min_ioi_ms / 1000.0
    slow = floor * 2.5
    if ioi_sec <= floor:
        return 1.0
    if ioi_sec >= slow:
        return 0.0
    return (slow - ioi_sec) / (slow - floor)


# Absolute floor for a rebound double: even a diddle can't be arbitrarily fast.
_DOUBLE_FLOOR_FRAC = 0.5   # fraction of single_hand_min_ioi below which a same-hand pair is implausible
_NEUTRAL = 0.5             # speed_fraction at which singles and doubles are equally economical


def score_feasible(cell: list[Strike], template, config: PipelineConfig,
                   weight: float = 1.0) -> float:
    """Cost from the economy of each adjacent hand pair, given IOI and surface.

    - same-hand pair (a diddle): cheap when fast, mildly costly when gratuitously slow;
      implausibly-fast same-hand pairs pay a hard penalty.
    - different-hand pair (a single) on the SAME surface: costly when fast (near the floor,
      humans switch to doubles), mildly rewarded when slow. On a DIFFERENT surface it's free
      — cross-kit alternation is natural and not a speed signal."""
    if template.is_fallback or len(cell) < 2:
        return 0.0
    cost = 0.0
    run = 1   # length of the current same-hand run (a rebound gives a double, not an 8-run)
    for i in range(len(cell) - 1):
        ioi = cell[i + 1].time - cell[i].time
        same_hand = template.tokens[i] == template.tokens[i + 1]
        run = run + 1 if same_hand else 1
        cost += pair_feasibility(
            ioi, same_hand, cell[i + 1].surface == cell[i].surface, run, config, weight)
    return cost


def pair_feasibility(ioi_sec: float, same_hand: bool, same_surface: bool, run_len: int,
                     config: PipelineConfig, weight: float = 1.0) -> float:
    """Cost of ONE hand-strike adjacency — the single source of truth for singles-vs-doubles
    physics, used both within a cell and across a cell seam so the score is segmentation-
    invariant. `run_len` = length of the same-hand run ending at the second stroke (>=2 for a
    same-hand pair, 1 otherwise); threaded globally by the decoder so a rebound can't be farmed
    into an infinite run by re-segmenting."""
    frac = speed_fraction(ioi_sec, config)
    if same_hand:
        floor = config.single_hand_min_ioi_ms / 1000.0
        if ioi_sec < floor * _DOUBLE_FLOOR_FRAC:
            return weight * 3.0                       # too fast even for a rebound double
        cost = weight * (_NEUTRAL - frac)             # <0 reward a double when fast, >0 when slow
        if run_len >= 3:
            cost += weight * (run_len - 2) * (0.5 + frac)   # rebound can't sustain 3+ on one hand
        return cost
    # different hand
    if same_surface:
        return weight * (frac - _NEUTRAL)             # >0 penalty singles when fast, <0 when slow
    return 0.0                                        # different surface → no speed signal (reach is step 4)


def cell_physical_cost(cell: list[Strike], template, ref: float, spread: float,
                       config: PipelineConfig, w_accent: float = 1.0,
                       w_feasible: float = 1.0) -> float:
    """The two backbone terms combined. Reach/seam/idiom are added by the decoder."""
    return (score_accent(cell, template, ref, spread, w_accent)
            + score_feasible(cell, template, config, w_feasible))


if __name__ == "__main__":
    from . import templates as T

    cfg = PipelineConfig()

    def cell(times, surfs, vels):
        return [Strike(time=t, surface=s, velocity=v) for t, s, v in zip(times, surfs, vels)]

    para = next(t for t in T.VOCABULARY if t.name == "paradiddle")          # R! L R R
    single4 = next(t for t in T.VOCABULARY if t.name == "single_stroke")    # R L R L
    double4 = next(t for t in T.VOCABULARY if t.name == "double_stroke")    # R R L L

    # --- C_accent: a loud first note should prefer the accented paradiddle over flat singles
    loud_first = cell([0.0, 0.12, 0.24, 0.36], ["snare"] * 4, [0.95, 0.4, 0.4, 0.4])
    ref, spr = dynamic_reference(loud_first)
    a_para = score_accent(loud_first, para, ref, spr)
    a_single = score_accent(loud_first, single4, ref, spr)
    print(f"loud-first cell:  accent(paradiddle)={a_para:.2f}  accent(single)={a_single:.2f}")
    assert a_para < a_single, "accented template should win on a clearly accented cell"

    # --- and a FLAT cell must NOT prefer the accented template (honest ceiling for accents)
    flat = cell([0.0, 0.12, 0.24, 0.36], ["snare"] * 4, [0.6, 0.6, 0.6, 0.6])
    ref, spr = dynamic_reference(flat)
    print(f"flat cell:        accent(paradiddle)={score_accent(flat, para, ref, spr):.2f}  "
          f"accent(single)={score_accent(flat, single4, ref, spr):.2f}")
    assert score_accent(flat, single4, ref, spr) <= score_accent(flat, para, ref, spr)

    # --- C_feasible: fast same-surface run should prefer doubles over singles
    fast = cell([0.0, 0.05, 0.10, 0.15], ["snare"] * 4, [0.6] * 4)   # 50 ms IOI (< 70 ms floor)
    f_double = score_feasible(fast, double4, cfg)
    f_single = score_feasible(fast, single4, cfg)
    print(f"fast mono cell:   feasible(double)={f_double:.2f}  feasible(single)={f_single:.2f}")
    assert f_double < f_single, "fast mono-surface should favor doubles (RRLL) over singles"

    # --- slow same-surface run should NOT be pushed into gratuitous doubles
    slow = cell([0.0, 0.30, 0.60, 0.90], ["snare"] * 4, [0.6] * 4)   # 300 ms IOI (comfortable)
    print(f"slow mono cell:   feasible(double)={score_feasible(slow, double4, cfg):.2f}  "
          f"feasible(single)={score_feasible(slow, single4, cfg):.2f}")
    assert score_feasible(slow, single4, cfg) <= score_feasible(slow, double4, cfg)

    print("\nall scoring unit checks passed ✓")
