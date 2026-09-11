from dataclasses import dataclass, field
from typing import Optional

# Surfaces as ADTOF classifies them
SURFACES = ["kick", "snare", "tom", "hihat", "cymbal"]
# Surfaces reachable by hands (kick/hihat-pedal are feet)
HAND_SURFACES = ["snare", "tom", "hihat", "cymbal", "idle"]
LIMBS = ["L", "R", "F"]
SOURCE_VALUES = ["constraint", "inference", "propagation"]


@dataclass
class Strike:
    time: float           # raw onset time in seconds
    surface: str          # kick/snare/tom/hihat/cymbal
    velocity: float       # 0-1 normalized
    quantized_pos: Optional[float] = None  # beat position (e.g. 1.25 = beat 1 + one 16th)


@dataclass
class TimeSlot:
    time: float           # representative time (mean onset of constituent strikes)
    strikes: list = field(default_factory=list)  # list[Strike], 1-4 simultaneous


@dataclass
class AssignedStrike:
    time: float
    surface: str
    velocity: float
    quantized_pos: Optional[float]
    limb: str             # L / R / F
    limb_probs: dict      # e.g. {"L": 0.7, "R": 0.3} or {"F": 1.0}
    confidence: float     # max(limb_probs.values())
    source: str           # constraint / inference / propagation
    flagged_ambiguous: bool


@dataclass
class PipelineConfig:
    # Stage 1.5 — tom sub-classification (per-clip, adaptive; no fixed pitch cutoffs)
    subclassify_toms: bool = True
    tom_pitch_fmin: float = 60.0        # physical pitch SEARCH range for toms (Hz) — not a
    tom_pitch_fmax: float = 400.0       #   classification threshold; boundaries are learned per clip
    tom_pitch_window_ms: float = 120.0  # window after each tom onset for pitch estimation
    tom_cluster_max: int = 4            # most toms we'll split a clip into
    tom_cluster_sil_floor: float = 0.5  # dimensionless cluster-quality floor (tuning-invariant)

    # Stage 3 — chord coincidence window
    coincidence_window_ms: float = 20.0

    # Stage 4 — physical ceilings
    single_hand_min_ioi_ms: float = 70.0    # conservative 14 Hz single-hand ceiling
    double_buzz_band_ms: float = 100.0      # IOI range flagged as possible double/buzz

    # Stage 5 — player geometry
    handedness: str = "right"   # right / left
    style: str = "crossed"      # crossed / open

    # Stage 5 — HMM soft parameters (all tunable)
    alternation_weight: float = 2.0     # log-odds bonus for RLRL at moderate tempo
    reach_cost_scale: float = 1.5       # scales Euclidean reach penalty
    continuity_bonus: float = 0.5       # bonus for hand staying at same surface
    double_prior_speed: float = 0.8     # fraction of single_hand_min_ioi where doubles prior kicks in
    double_sig_weight: float = 0.8      # weight for double-stroke acoustic signature (intentionally modest)
    metric_anchor_weight: float = 1.0     # #1 strength of metric phase bias
    velocity_accent_weight: float = 1.0   # #2 strength of accent/ghost bias
    marginal_neighbor_weight: float = 1.5 # #0b alternation context in the marginal

    # Stage 5 — ambiguity threshold
    ambiguity_threshold: float = 0.65   # confidence below this → flagged_ambiguous

    # Stage 5b
    rudiment_prior_weight: float = 0.3  # Tier 1 cap: cannot outvote real acoustic evidence
    self_prior_enabled: bool = False    # Tier 2 off by default


# Kit geometry: 2D positions for reach-cost computation.
# Origin is the snare. Right is positive X, up (toward hi-hat) is positive Y.
KIT_GEOMETRY = {
    "right_crossed": {
        "snare":   (0.0,  0.0),
        "hihat":   (-0.5, 0.3),
        "tom":     (0.4, -0.1),
        # Sub-toms from Stage 1.5, high pitch (rack, left) -> low pitch (floor, right):
        "tom0":    (-0.15, -0.10),
        "tom1":    (0.15, -0.10),
        "tom2":    (0.45, -0.05),
        "tom3":    (0.65,  0.00),
        "cymbal":  (0.7,  0.4),   # ride / crash
        "idle":    (0.0,  0.6),   # arms at rest
    },
    "right_open": {
        "snare":   (0.0,  0.0),
        "hihat":   (0.5,  0.3),   # hi-hat is right of snare in open setup
        "tom":     (0.4, -0.1),
        "cymbal":  (-0.5, 0.4),
        "idle":    (0.0,  0.6),
    },
    "left_crossed": {
        "snare":   (0.0,  0.0),
        "hihat":   (0.5,  0.3),
        "tom":     (-0.4,-0.1),
        "cymbal":  (-0.7, 0.4),
        "idle":    (0.0,  0.6),
    },
    "left_open": {
        "snare":   (0.0,  0.0),
        "hihat":   (-0.5, 0.3),
        "tom":     (-0.4,-0.1),
        "cymbal":  (0.6,  0.4),
        "idle":    (0.0,  0.6),
    },
}

# Natural leading hand per surface per setup (used as asymmetric emission prior)
NATURAL_HAND = {
    "right_crossed": {"hihat": "L", "snare": None, "tom": "R", "cymbal": "R"},
    "right_open":    {"hihat": "R", "snare": None, "tom": "R", "cymbal": "R"},
    "left_crossed":  {"hihat": "R", "snare": None, "tom": "L", "cymbal": "L"},
    "left_open":     {"hihat": "L", "snare": None, "tom": "L", "cymbal": "L"},
}
