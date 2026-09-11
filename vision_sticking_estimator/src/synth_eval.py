"""Synthetic Stage-5 evaluation harness — symbolic, audio-free.

We generate sticking patterns whose ground-truth L/R is known by construction
(rudiments, grooves, fills), build the exact structures Stage 5 consumes
(`constraint_dicts` + `beat_info`), run `assign_limbs`, and score the result
against ground truth using the existing eval matching contract.

Why audio-free: rendering to audio would route every score through ADTOF
(onset/velocity/instrument detection), which is trained on real drums and would
dominate the error on synthesized one-shots. To stress and tune the *sticking
logic* (Stage 5 priors, alternation, the honest-ceiling flagging) we inject
known onsets/velocities/surfaces directly. An audio renderer with domain
randomisation is a separate, later phase, needed only for end-to-end validation
or for training a learned model.

Vision-ready seam: each generated hit carries `gt_limb` separately from the
acoustic observables (surface/velocity/time). A future pose/stick estimator
would supply a noisy per-hit L/R observation at exactly this point — the harness
already isolates that injection site.

Run:  python -m src.synth_eval
"""

import random
from dataclasses import dataclass

from .types import Strike, TimeSlot, PipelineConfig
from .stage4_constraints import apply_constraints
from .stage5_hmm import assign_limbs
from .eval import Annotation

START_OFFSET = 0.5  # lead-in seconds so all times are positive/realistic


@dataclass
class GenHit:
    """One generated strike: acoustic observables + known ground truth."""
    time: float
    surface: str
    velocity: float
    quantized_pos: float
    gt_limb: str       # L / R / F  — known by construction
    tier: str          # easy / medium / hard (audio-resolvability of the L/R call)


@dataclass
class Pattern:
    name: str
    bpm: float
    meter: str
    subdivision: int
    hits: list  # list[GenHit]
    vocab: str = "in"   # "in" = pattern type is in the grammar corpus; "off" = held-out


# ---------------------------------------------------------------------------
# Grid helper — places a hit and computes the production-identical quantized_pos
# ---------------------------------------------------------------------------

def _grid(bpm: float, meter: str, bar: int, beat_in_bar: int, frac: float):
    """Return (time_sec, quantized_pos) for a hit at (bar, beat, within-beat frac).

    Mirrors stage2_beats._quantize: the fractional part of quantized_pos encodes
    position *within the bar* (frac scaled by beats_per_bar), not within a beat.
    """
    beats_per_bar = int(meter.split("/")[0])
    beat_dur = 60.0 / bpm
    t = START_OFFSET + (bar * beats_per_bar + (beat_in_bar - 1) + frac) * beat_dur
    qpos = (bar + 1) + (beat_in_bar - 1 + frac) / beats_per_bar
    return t, qpos


# ---------------------------------------------------------------------------
# Pattern generators (right-handed conventions; lead = R)
# ---------------------------------------------------------------------------

# Sixteenth-note offsets within a beat → within-beat fraction.
_S16 = [0.0, 0.25, 0.5, 0.75]


def _roll_on_snare(name, bpm, sticking, accented_lead, bars=2):
    """Continuous 16th-note roll on the snare with a fixed sticking cycle.

    sticking: e.g. "RL" (single) or "RRLL" (double), repeated to fill the bars.
    accented_lead: if True, the first note of every beat is loud (an accent the
    velocity prior can latch onto → phase is resolvable). If False, dead-even
    dynamics → the L/R phase is a genuine coin flip from audio = honest ceiling.
    """
    meter, subdivision = "4/4", 16
    hits = []
    n = bars * 4 * 4  # bars * beats * 16ths-per-beat
    for k in range(n):
        bar, rem = divmod(k, 16)
        beat_in_bar, sub = divmod(rem, 4)
        beat_in_bar += 1
        t, q = _grid(bpm, meter, bar, beat_in_bar, _S16[sub])
        limb = sticking[k % len(sticking)]
        on_beat = (sub == 0)
        vel = 0.95 if (accented_lead and on_beat) else 0.6
        # Resolvability: the trailing note of a same-hand double with no accent
        # is the hard case; an accented anchor or a plain alternation is easier.
        is_double_tail = sticking[k % len(sticking)] == sticking[(k - 1) % len(sticking)]
        if accented_lead:
            tier = "medium" if is_double_tail else "easy"
        else:
            tier = "hard" if is_double_tail else "medium"
        hits.append(GenHit(t, "snare", vel, q, limb, tier))
    return Pattern(name, bpm, meter, subdivision, hits)


def _paradiddle(bpm=110):
    """RLRR LRLL on snare, 16ths, with a natural accent on each group's lead."""
    meter, subdivision = "4/4", 16
    sticking = "RLRRLRLL"
    hits = []
    n = 2 * 4 * 4
    for k in range(n):
        bar, rem = divmod(k, 16)
        beat_in_bar, sub = divmod(rem, 4)
        beat_in_bar += 1
        t, q = _grid(bpm, meter, bar, beat_in_bar, _S16[sub])
        limb = sticking[k % 8]
        vel = 0.9 if (k % 8) in (0, 4) else 0.6  # accent the diddle leads
        is_double_tail = sticking[k % 8] == sticking[(k - 1) % 8]
        tier = "medium" if is_double_tail else "easy"
        hits.append(GenHit(t, "snare", vel, q, limb, tier))
    return Pattern("paradiddle", bpm, meter, subdivision, hits)


def _rock_groove(bpm=100):
    """Eighth-note groove: hi-hat on every 8th (R), snare backbeat (L), kick (F).

    GT uses the standard right-handed *crossed* sticking: right hand rides the
    hi-hat, left hand plays the snare backbeat. (If NATURAL_HAND disagrees, the
    harness will surface it as systematically low groove accuracy — by design we
    encode the physically conventional sticking, not the model's assumption.)
    """
    meter, subdivision = "4/4", 8
    hits = []
    for bar in range(2):
        for beat in range(1, 5):
            for frac in (0.0, 0.5):
                t, q = _grid(bpm, meter, bar, beat, frac)
                hits.append(GenHit(t, "hihat", 0.7, q, "R", "easy"))
            if beat in (2, 4):  # snare backbeat on 2 & 4
                t, q = _grid(bpm, meter, bar, beat, 0.0)
                hits.append(GenHit(t, "snare", 0.9, q, "L", "easy"))
            if beat in (1, 3):  # kick on 1 & 3
                t, q = _grid(bpm, meter, bar, beat, 0.0)
                hits.append(GenHit(t, "kick", 0.85, q, "F", "easy"))
    hits.sort(key=lambda h: h.time)
    return Pattern("rock_groove", bpm, meter, subdivision, hits)


def _tom_fill(bpm=110):
    """Single strokes (RLRL) descending snare→tom→tom, 16ths — surface changes
    make most calls resolvable; same-surface fast pairs are the harder ones."""
    meter, subdivision = "4/4", 16
    surfaces = ["snare", "snare", "tom", "tom", "tom", "tom", "snare", "snare"]
    hits = []
    for k in range(16):
        bar, rem = divmod(k, 16)
        beat_in_bar, sub = divmod(rem, 4)
        beat_in_bar += 1
        t, q = _grid(bpm, meter, bar, beat_in_bar, _S16[sub])
        limb = "R" if k % 2 == 0 else "L"
        surf = surfaces[k % len(surfaces)]
        hits.append(GenHit(t, surf, 0.8, q, limb, "medium"))
    return Pattern("tom_fill", bpm, meter, subdivision, hits)


def build_bank():
    """The synthetic test bank. Grow this — more tempos, grooves, fills.

    `vocab` marks whether the pattern TYPE is in the grammar corpus. Reporting
    grammar gains separately on held-out ("off") patterns guards against the
    train/test circularity of crediting a prior for predicting what it memorized.
    """
    bank = [
        _roll_on_snare("single_roll_slow",   90,  "RL",   accented_lead=True),
        _roll_on_snare("single_roll_fast",   150, "RL",   accented_lead=True),
        _roll_on_snare("double_roll_accent", 110, "RRLL", accented_lead=True),
        _roll_on_snare("double_roll_even",   110, "RRLL", accented_lead=False),
        _paradiddle(),
        _rock_groove(),
        _tom_fill(),
    ]
    held_out = {"rock_groove", "tom_fill"}  # types not present in corpus/*.jsonl
    for p in bank:
        p.vocab = "off" if p.name in held_out else "in"
    return bank


# ---------------------------------------------------------------------------
# Build Stage-5 input from a symbolic pattern
# ---------------------------------------------------------------------------

def _to_stage5_input(pat: Pattern, config: PipelineConfig, vel_noise=0.0, rng=None):
    """Return (constraint_dicts, beat_info) — the real Stage-4 output shape.

    vel_noise > 0 perturbs each hit's velocity with Gaussian noise (clipped to
    [0,1]), blurring the accent cue so the L/R call becomes a genuine tie in more
    places — i.e. the regime where a tie-breaking prior can actually change a
    DECISION, not just the flag. Ground truth (gt_limb) is untouched."""
    # Group simultaneous hits (same rounded time) into TimeSlots.
    by_time: dict[int, list] = {}
    for h in pat.hits:
        key = round(h.time * 1000)  # 1 ms grouping
        by_time.setdefault(key, []).append(h)

    def vel(v):
        if vel_noise > 0 and rng is not None:
            return min(1.0, max(0.0, v + rng.gauss(0.0, vel_noise)))
        return v

    slots = []
    for key in sorted(by_time):
        group = by_time[key]
        t = sum(h.time for h in group) / len(group)
        strikes = [Strike(h.time, h.surface, vel(h.velocity), h.quantized_pos) for h in group]
        slots.append(TimeSlot(t, strikes))

    end = pat.hits[-1].time + 1.0
    beat_info = {
        "tempo_curve": [(0.0, pat.bpm), (end, pat.bpm)],
        "meter": pat.meter,
        "subdivision": pat.subdivision,
    }
    constraint_dicts = apply_constraints(slots, config, beat_info)
    return constraint_dicts, beat_info


# ---------------------------------------------------------------------------
# Scoring — pool per-hit records across the whole bank, by tier
# ---------------------------------------------------------------------------

@dataclass
class _Rec:
    tier: str
    gt: str
    pred: str
    flagged: bool
    vocab: str = "in"


def _score_pattern(pat: Pattern, config: PipelineConfig, rudiment_prior=None,
                   vel_noise=0.0, rng=None):
    cds, beat_info = _to_stage5_input(pat, config, vel_noise, rng)
    assigned = assign_limbs(cds, beat_info, None, config, rudiment_prior)

    # Surface-aware match: synthetic times are exact, and a chord can hold two
    # hits at the same instant (e.g. kick + hi-hat on beat 1), so keying on time
    # alone cross-matches them. Key on (rounded time, surface) instead.
    gt = {(round(h.time * 1000), h.surface): h for h in pat.hits}
    recs = []
    for hit in assigned:
        h = gt.get((round(hit.time * 1000), hit.surface))
        if h is None:
            continue
        recs.append(_Rec(h.tier, h.gt_limb, hit.limb, hit.flagged_ambiguous, pat.vocab))
    return recs


def _summarize(recs: list, label: str) -> str:
    tiers = ["easy", "medium", "hard"]
    lines = [f"[{label}]"]
    for tier in tiers + ["ALL"]:
        sub = recs if tier == "ALL" else [r for r in recs if r.tier == tier]
        hand = [r for r in sub if r.gt in ("L", "R")]  # L/R only; kicks are trivial
        if not hand:
            continue
        nonflag = [r for r in hand if not r.flagged]
        correct = sum(1 for r in nonflag if r.pred == r.gt)
        acc = correct / len(nonflag) if nonflag else float("nan")
        # On hard hits the *right* behaviour is to flag, not to guess — report both.
        flag_rate = sum(1 for r in hand if r.flagged) / len(hand)
        lines.append(
            f"  {tier:7} n={len(hand):3d}  "
            f"acc(non-flagged)={acc:.3f} ({correct}/{len(nonflag)})  "
            f"flag_rate={flag_rate:.2f}"
        )
    return "\n".join(lines)


def run_bank(config: PipelineConfig = None, rudiment_prior=None,
             vel_noise=0.0, seed=0) -> list:
    config = config or PipelineConfig()
    rng = random.Random(seed)  # fixed seed => ON/OFF runs see identical perturbed inputs
    recs = []
    for pat in build_bank():
        recs.extend(_score_pattern(pat, config, rudiment_prior, vel_noise, rng))
    return recs


def _bank_accuracy(recs: list) -> tuple:
    """(accuracy over non-flagged L/R hits, n flagged) across the whole bank."""
    hand = [r for r in recs if r.gt in ("L", "R")]
    nonflag = [r for r in hand if not r.flagged]
    correct = sum(1 for r in nonflag if r.pred == r.gt)
    acc = correct / len(nonflag) if nonflag else float("nan")
    return acc, sum(1 for r in hand if r.flagged)


def _decision_accuracy(recs: list) -> float:
    """Accuracy over ALL L/R hits, ignoring flags — the DECISION metric. Distinct
    from _bank_accuracy: a prior that only moves hits into the flagged bucket
    raises non-flagged accuracy but leaves this unchanged. Only genuine flips of
    the argmax toward ground truth move this number."""
    hand = [r for r in recs if r.gt in ("L", "R")]
    if not hand:
        return float("nan")
    return sum(1 for r in hand if r.pred == r.gt) / len(hand)


def grammar_dual_run(config: PipelineConfig = None, vel_noise=0.0, seed=0):
    """Audio-free dual run: sticking-grammar prior ON vs OFF over the whole bank.
    Reports DECISION accuracy (did the prior flip argmaxes toward truth?) and
    non-flagged accuracy + flag count (calibration), split by in/off vocabulary.
    vel_noise injects the ambiguity that clean synthetic data lacks."""
    from .stage5b_priors import build_tier1_prior

    config = config or PipelineConfig()
    prior = build_tier1_prior(weight_cap=config.rudiment_prior_weight)

    recs_off = run_bank(config, None, vel_noise, seed)
    recs_on = run_bank(config, prior, vel_noise, seed)

    print("=" * 70)
    print(f"GRAMMAR DUAL RUN  (vel_noise={vel_noise}, cap={config.rudiment_prior_weight})")
    print("=" * 70)
    print(f"{'subset':12} {'decision_acc':>22} {'nonflag_acc':>20} {'flags':>10}")
    for label, sub_off, sub_on in (
        ("ALL", recs_off, recs_on),
        ("in-vocab", [r for r in recs_off if r.vocab == "in"], [r for r in recs_on if r.vocab == "in"]),
        ("off-vocab", [r for r in recs_off if r.vocab == "off"], [r for r in recs_on if r.vocab == "off"]),
    ):
        dec_off, dec_on = _decision_accuracy(sub_off), _decision_accuracy(sub_on)
        nf_off, fl_off = _bank_accuracy(sub_off)
        nf_on, fl_on = _bank_accuracy(sub_on)
        print(f"{label:12} {dec_off:6.3f}->{dec_on:6.3f} ({dec_on-dec_off:+.3f}) "
              f"  {nf_off:5.3f}->{nf_on:5.3f}   {fl_off:2d}->{fl_on:2d}")
    print("\nDECISION acc is the one that matters for the model; nonflag/flags are "
          "calibration. A prior that only changes the last two is a flagger, not a fixer.")


def metric_anchor_sweep():
    """Settle the open metric_anchor_weight question across the whole bank."""
    print("=" * 64)
    print("METRIC ANCHOR WEIGHT SWEEP (whole bank)")
    print("=" * 64)
    for w in (0.0, 0.05, 0.25, 0.5, 1.0):
        cfg = PipelineConfig()
        cfg.metric_anchor_weight = w
        recs = run_bank(cfg)
        print(_summarize(recs, f"metric_anchor_weight={w}"))
        print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    for nz in (0.0, 0.10, 0.20):
        grammar_dual_run(vel_noise=nz)
        print()
