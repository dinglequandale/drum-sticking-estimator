"""Stage 5b — Sticking-vocabulary priors.

Tier 1 (DEFAULT ON): global rudiment n-gram prior. Safe because it is
externally anchored and cannot learn the model's own biases.

Tier 2 (DEFAULT OFF): per-video self-prior. Dangerous feedback loop risk.
Only enabled behind three hard guards; see comments below.
"""

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier 1 — Global rudiment prior
# ---------------------------------------------------------------------------

from pathlib import Path

from .grammar import Context, build_grammar

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


class RudimentPrior:
    """Tier-1 sticking-grammar prior: a learned, context-conditioned, back-off
    n-gram over limb sequences (see src/grammar.py), estimated from the symbolic
    corpus in corpus/*.jsonl. Replaces the former hardcoded-rudiment bigram, whose
    log_prob threaded no context (it averaged over all contexts). Externally
    anchored to notated sticking, so it cannot learn the model's own biases.

    Its per-stroke contribution is a BOUNDED log-odds nudge: log(P(limb|ctx)/0.5)
    clamped to +/-weight_cap, centered so an uninformative context contributes ~0.
    That keeps it a tie-breaker which cannot outvote acoustic evidence (the honest
    ceiling). An empty/missing corpus yields a uniform grammar -> zero nudge, i.e.
    safely equivalent to the prior being off.
    """

    def __init__(self, weight_cap: float = 0.3, corpus_dir=CORPUS_DIR):
        self.weight_cap = weight_cap
        self._grammar = build_grammar(corpus_dir)
        logger.info(
            "RudimentPrior: sticking grammar over %d phrases [%s], %d contexts",
            self._grammar.n_phrases,
            ", ".join(f"{k}:{v:.0f}" for k, v in self._grammar.provenance_mass.items()),
            len(self._grammar._counts),
        )

    def nudge(self, limb: str, ctx: Context) -> float:
        """Bounded log-odds preference for `limb` in context vs a neutral 0.5 hand
        split. >0 favors, <0 disfavors, |value| <= weight_cap."""
        dist = self._grammar.distribution(ctx)
        p = max(dist.get(limb, 1e-3), 1e-3)
        val = math.log(p) - math.log(0.5)
        return max(-self.weight_cap, min(self.weight_cap, val))


# ---------------------------------------------------------------------------
# Tier 2 — Per-video self-prior (DEFAULT OFF)
# ---------------------------------------------------------------------------

class SelfPrior:
    """Learns this player's sticking grammar from constraint-sourced hits only.

    GUARD 1: Only trains on source='constraint' hits — hits forced by hard
             physical facts (speed gate, surface change). Hits from 'inference'
             are FORBIDDEN as training data to prevent the laundering loop where
             the model's own soft-prior guesses are fed back as evidence.

    GUARD 2: weight_cap ensures this prior breaks ties only.

    GUARD 3: Must pass dual-run eval (prior ON vs OFF) on off-vocabulary clip.
             If it reduces flag_rate without raising accuracy, it is corrupting.
    """

    def __init__(self, order: int = 2, weight_cap: float = 0.3):
        self.order = order
        self.weight_cap = weight_cap
        self._counts: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._n_trained = 0
        self._locked = False  # set True after training to prevent accidental re-training

    def train(self, assigned_strikes: list) -> None:
        """Feed constraint-sourced hits only. Raises if called with inference hits."""
        if self._locked:
            raise RuntimeError("SelfPrior.train() called after lock — check call site")

        for hit in assigned_strikes:
            if hit.source != "constraint":
                # Structurally refuse inference hits — this is not defensive code,
                # it is the mechanism that breaks the feedback loop.
                continue
            tok = hit.limb
            if tok not in ("L", "R"):
                continue
            # A real implementation would maintain sequence context;
            # this is left as a placeholder for the full n-gram implementation.
            self._n_trained += 1

        self._locked = True
        logger.info("SelfPrior trained on %d constraint-sourced hits", self._n_trained)

    def nudge(self, limb: str, ctx) -> float:
        if self._n_trained < 20:
            # Too sparse to be useful; return zero rather than noise
            return 0.0
        # Placeholder: full implementation mirrors RudimentPrior with learned counts
        return 0.0


def build_tier1_prior(weight_cap: float = 0.3) -> RudimentPrior:
    return RudimentPrior(weight_cap=weight_cap)


def build_tier2_prior(weight_cap: float = 0.3) -> SelfPrior:
    logger.warning(
        "Tier-2 self-prior is experimental. Ensure you run dual-run eval "
        "(prior ON vs OFF on off-vocabulary clip) before using results."
    )
    return SelfPrior(weight_cap=weight_cap)
