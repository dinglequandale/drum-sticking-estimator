"""Learned sticking grammar — a context-conditioned, provenance-weighted,
stateful back-off n-gram over limb sequences. Estimated from the symbolic
corpus in `corpus/*.jsonl` (no audio). This is the real implementation behind
leg 3 of the fusion model; it replaces the hand-tuned placeholder in
`stage5b_priors.RudimentPrior` (see DRUM_GRAMMAR.md and corpus/schema.md).

Conditioning for P(limb_t | context):
  - hist        : the preceding primary limbs (order-k; wraps on cyclic phrases)
  - surface     : where this stroke lands
  - metric_class: strong / weak / off (from `pos` on the phrase subdivision)
  - accent      : whether this stroke is accented (known at inference from velocity)
  - speed_tier  : density class (often null in tempo-agnostic method books)

Estimation is count-based with Katz-style stupid back-off: the most specific
context with enough mass wins, with an add-alpha smoothed conditional there.
Grace notes (flam/drag soft strokes) are kept in the corpus but NOT counted as
history tokens in v1 — the grammar predicts the primary-limb stream, which is
what the audio front-end actually onsets. Their structure is future work.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LIMBS = ("L", "R", "F")
ORDER = 2                 # primary-limb history length
MIN_COUNT = 3.0           # min weighted mass at a back-off level to trust it
ALPHA = 0.3               # add-alpha smoothing within a level
LEVEL_DISCOUNT = 0.4      # stupid-back-off discount per level dropped

# Real human choices outweigh exhaustive permutation enumeration (the circularity
# guard — see corpus/schema.md). Enumeration is a down-weighted background.
PROVENANCE_WEIGHT = {
    "rudiment": 1.0,
    "applied": 1.0,
    "convention": 1.0,
    "enumeration": 0.15,
}


@dataclass(frozen=True)
class Context:
    hist: tuple        # up to ORDER preceding primary limbs, most-recent LAST
    surface: str = "snare"
    metric: str = "off"
    accent: bool = False
    speed: str | None = None


def metric_class(pos, subdivision: str | None) -> str:
    """Strong (beat) / weak (off-beat 8th) / off (finer subdivision)."""
    if pos is None:
        return "off"
    p = int(pos)
    if subdivision == "triplet":
        return "strong" if p % 3 == 0 else "off"
    # default 8th/16th grid
    if p % 4 == 0:
        return "strong"
    if p % 2 == 0:
        return "weak"
    return "off"


def _projections(ctx: Context):
    """Back-off context keys, most-specific first. Tag prefixes keep levels
    distinct so a shorter history never collides with a longer one."""
    h = ctx.hist
    return [
        ("h", h, "s", ctx.surface, "m", ctx.metric, "a", ctx.accent, "v", ctx.speed),
        ("h", h, "s", ctx.surface, "a", ctx.accent),
        ("h", h, "a", ctx.accent),
        ("h", h),
        ("h", h[-1:]),     # shorten history to last limb only
        (),                # unigram
    ]


class StickingGrammar:
    def __init__(self, order: int = ORDER):
        self.order = order
        self._counts: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.provenance_mass: dict[str, float] = defaultdict(float)  # weighted strokes per provenance
        self.n_phrases = 0
        self.n_duplicates = 0

    # -- estimation -------------------------------------------------------
    def fit(self, phrases: list[dict]) -> "StickingGrammar":
        seen = set()
        for ph in phrases:
            sig = self._signature(ph)
            if sig in seen:           # identical sticking already counted — skip (curation guard)
                self.n_duplicates += 1
                continue
            seen.add(sig)
            self.n_phrases += 1
            prov = ph.get("provenance", "rudiment")
            w = PROVENANCE_WEIGHT.get(prov, 1.0)
            primaries = [s for s in ph["strokes"] if not s.get("grace")]
            self.provenance_mass[prov] += w * len(primaries)
            limbs = [s["limb"] for s in primaries]
            n = len(limbs)
            cyclic = ph.get("cyclic", False)
            subdiv = ph.get("subdivision")
            speed = ph.get("speed_tier")
            for t in range(n):
                hist = self._history(limbs, t, cyclic)
                ctx = Context(
                    hist=hist,
                    surface=primaries[t].get("surface", "snare"),
                    metric=metric_class(primaries[t].get("pos"), subdiv),
                    accent=bool(primaries[t].get("accent")),
                    speed=speed,
                )
                for key in _projections(ctx):
                    self._counts[key][limbs[t]] += w
        return self

    @staticmethod
    def _signature(ph: dict) -> tuple:
        """Identity of a sticking phrase for dedup: the full stroke pattern plus
        loop/grid context. Permutations differ here (intended); verbatim repeats
        collapse. Provenance is NOT in the signature — the same pattern shouldn't
        be double-counted just because two books print it."""
        return (
            ph.get("cyclic", False), ph.get("subdivision"),
            tuple((s["limb"], s.get("surface", "snare"),
                   bool(s.get("accent")), bool(s.get("grace")))
                  for s in ph["strokes"]),
        )

    def _history(self, limbs, t, cyclic) -> tuple:
        out = []
        for j in range(self.order, 0, -1):
            idx = t - j
            if idx < 0:
                if cyclic:
                    out.append(limbs[idx])   # negative index wraps
                # non-cyclic start: omit (shorter history)
            else:
                out.append(limbs[idx])
        return tuple(out)

    # -- query ------------------------------------------------------------
    def distribution(self, ctx: Context) -> dict[str, float]:
        """Add-alpha smoothed P(limb | ctx) at the most specific level with mass."""
        for key in _projections(ctx):
            dist = self._counts.get(key)
            if dist and sum(dist.values()) >= MIN_COUNT:
                total = sum(dist.values()) + ALPHA * len(LIMBS)
                return {lb: (dist.get(lb, 0.0) + ALPHA) / total for lb in LIMBS}
        # nothing learned anywhere → uniform over hands (F unlikely)
        return {"L": 0.5, "R": 0.5, "F": 0.0}

    def log_prob(self, limb: str, ctx: Context) -> float:
        """log P(limb | ctx), with a back-off level penalty folded in. Used as a
        soft additive term in stage5; callers should still cap its magnitude."""
        for level, key in enumerate(_projections(ctx)):
            dist = self._counts.get(key)
            if dist and sum(dist.values()) >= MIN_COUNT:
                total = sum(dist.values()) + ALPHA * len(LIMBS)
                p = (dist.get(limb, 0.0) + ALPHA) / total
                return math.log(p) + level * math.log(LEVEL_DISCOUNT)
        return math.log(0.5 if limb in ("L", "R") else 1e-3)


def load_corpus(corpus_dir: str | Path = "corpus") -> list[dict]:
    phrases = []
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                phrases.append(json.loads(line))
    return phrases


def build_grammar(corpus_dir: str | Path = "corpus") -> StickingGrammar:
    return StickingGrammar().fit(load_corpus(corpus_dir))


if __name__ == "__main__":
    import sys
    g = build_grammar(Path(__file__).resolve().parent.parent / "corpus")
    total_mass = sum(g.provenance_mass.values()) or 1.0
    print(f"phrases: {g.n_phrases} kept, {g.n_duplicates} dup(s) skipped | "
          f"learned contexts: {len(g._counts)}")
    print("provenance mass balance (informative signal must dominate enumeration):")
    for prov, mass in sorted(g.provenance_mass.items(), key=lambda kv: -kv[1]):
        print(f"    {prov:12} {mass:7.1f}  ({100*mass/total_mass:4.1f}%)")
    print()

    def show(label, ctx):
        d = g.distribution(ctx)
        print(f"{label:42} L={d['L']:.2f} R={d['R']:.2f}  ->{max(d, key=d.get)}")
        return d

    print("=== FACT 1: paradiddle diddle flips the lead ===")
    d_rr = show("after R,R (end of a diddle)", Context(hist=("R", "R")))
    d_ll = show("after L,L (end of a diddle)", Context(hist=("L", "L")))

    print("\n=== FACT 2: single strokes alternate ===")
    show("after ...L (single-stroke context)", Context(hist=("R", "L")))
    show("after ...R", Context(hist=("L", "R")))

    print("\n=== FACT 3: context-conditioning sharpens an ambiguous call ===")
    show("after R,L, no metric context", Context(hist=("R", "L")))
    show("after R,L on a strong beat", Context(hist=("R", "L"), metric="strong"))

    # NOTE: surface->hand conditioning (hi-hat tends R, etc.) is present in the
    # applied data but currently below MIN_COUNT, so back-off correctly declines
    # to trust it. Combining surface + history needs interpolation (v2) and more
    # curated applied volume per surface — deliberately not built yet.

    ok = d_rr["L"] > d_rr["R"] and d_ll["R"] > d_ll["L"]
    print("\nSELF-TEST:", "PASS" if ok else "FAIL",
          "- a diddle (double) hands the lead to the other hand")
    sys.exit(0 if ok else 1)
