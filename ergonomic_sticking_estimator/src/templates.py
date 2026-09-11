"""Stage 5 v2 — sticking template vocabulary.

A *template* is a short, lead-relative sticking cell (a "grouping"): a sequence of
hand tokens plus an accent pattern. The segmental decoder covers the hand-strike
stream with cells and pays a cost to dress each cell in a template (see
DESIGN_STAGE5_SEGMENTAL.md). Hands are chosen per *grouping*, never per isolated note.

Design decisions for v1 (documented so nobody "fixes" them blindly):

- **Hand-only tokens.** Tokens are "R"/"L". Feet (kick=F) are hard-assigned upstream
  and handled by the *decoder* as fixed anchors between/among cells — they affect IOI
  and cross-cell continuity but do not consume a template slot. This keeps the vocabulary
  small and avoids hardcoding linear cells (e.g. the RLKK sample phrase) that would overfit.
  Revisit if explicit linear idioms prove necessary.
- **Surface-agnostic.** A template says nothing about which drum. The cost terms read the
  *observed* surfaces off the strikes. So tom count (1 lumped tom vs. N sub-classified toms)
  is irrelevant here — tom expansion is purely additive downstream.
- **Accent is part of identity.** Accent pattern per slot: +1 accent, 0 normal, -1 ghost.
  A paradiddle is not "RLRR" — it is accent-R, L, R, R. Matching observed dynamics to this
  pattern is how the decoder selects the sticking *and* its phase (accents ride the lead hand).
- **The generic fallback is mandatory** (see spec's ban on a hard pattern bank): it lets any
  hand sequence compete on physics alone, so the vocabulary grants *discounts, never
  exclusivity*. Off-vocabulary playing is never force-fit into a rudiment.

Grace notes (flams/drags) are deferred — they need Stage 3 grace clustering. Marked TODO.

Runnable standalone (no heavy deps): `python templates.py` prints and self-checks.
"""

from dataclasses import dataclass

# Accent codes
ACCENT, NORMAL, GHOST = 1, 0, -1


@dataclass(frozen=True)
class Template:
    name: str
    family: str            # single | double | paradiddle | triple | roll | fill | fallback
    tokens: tuple          # ("R","L","R","R"); empty for the fallback
    accents: tuple         # ints in {+1,0,-1} aligned to tokens; empty for the fallback
    cyclic: bool = False   # loops (last stroke transitions back to first)
    is_fallback: bool = False
    # Base idiom nudge (log-cost): 0 = very common, >0 = rarer. Capped downstream so it
    # only ever breaks ties; the fallback carries 0 (pure physics, no discount, no penalty).
    idiom_cost: float = 0.0

    def __post_init__(self):
        if self.is_fallback:
            return
        if len(self.tokens) != len(self.accents):
            raise ValueError(f"{self.name}: tokens/accents length mismatch")
        if any(t not in ("R", "L") for t in self.tokens):
            raise ValueError(f"{self.name}: tokens must be R/L (feet are decoder anchors)")
        if any(a not in (ACCENT, NORMAL, GHOST) for a in self.accents):
            raise ValueError(f"{self.name}: accents must be in {{-1,0,+1}}")

    @property
    def length(self) -> int:
        return len(self.tokens)

    def mirror(self) -> "Template":
        """R<->L swap. Accents stay on the same *slot* (still on the lead hand, which is
        now the other physical hand). A cell may start on either hand depending on context,
        so the decoder tries both a template and its mirror."""
        if self.is_fallback:
            return self
        swap = {"R": "L", "L": "R"}
        return Template(
            name=self.name + "_m",
            family=self.family,
            tokens=tuple(swap[t] for t in self.tokens),
            accents=self.accents,
            cyclic=self.cyclic,
            idiom_cost=self.idiom_cost,
        )


# --- The curated vocabulary (R-lead canonical forms; mirrors generated on demand) ------
# Accent placements follow DRUM_GRAMMAR.md. idiom_cost is a *base* rarity nudge only.

_CANONICAL = [
    # Singles / alternation
    Template("single_pair",       "single",     ("R", "L"),                     (NORMAL, NORMAL),                         cyclic=True,  idiom_cost=0.0),
    Template("single_stroke",     "single",     ("R", "L", "R", "L"),           (NORMAL,) * 4,                            cyclic=True,  idiom_cost=0.0),
    # Doubles
    Template("diddle",            "double",     ("R", "R"),                     (NORMAL, NORMAL),                         cyclic=False, idiom_cost=0.3),
    Template("double_stroke",     "double",     ("R", "R", "L", "L"),           (NORMAL,) * 4,                            cyclic=True,  idiom_cost=0.0),
    # Paradiddle family (accent on the lead hand; the diddle hands off the lead)
    Template("paradiddle",        "paradiddle", ("R", "L", "R", "R"),           (ACCENT, NORMAL, NORMAL, NORMAL),         cyclic=True,  idiom_cost=0.0),
    Template("double_paradiddle", "paradiddle", ("R", "L", "R", "L", "R", "R"), (ACCENT, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL), cyclic=True, idiom_cost=0.4),
    Template("paradiddle_diddle", "paradiddle", ("R", "L", "R", "R", "L", "L"), (ACCENT, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL), cyclic=True, idiom_cost=0.4),
    # Triples / rolls
    Template("triple_stroke",     "triple",     ("R", "R", "R"),                (NORMAL, NORMAL, NORMAL),                 cyclic=False, idiom_cost=0.5),
    Template("five_stroke_roll",  "roll",       ("R", "R", "L", "L", "R"),      (NORMAL, NORMAL, NORMAL, NORMAL, ACCENT), cyclic=False, idiom_cost=0.4),
    # Around-the-kit fill cell (accent as the lead lands on the new drum)
    Template("three_across",      "fill",       ("R", "L", "L"),                (ACCENT, NORMAL, NORMAL),                 cyclic=False, idiom_cost=0.5),
    # TODO: flam/drag cells once Stage 3 grace clustering exists.
]

# The mandatory generic fallback: any hand sequence, scored on physics only, no discount.
FALLBACK = Template("fallback", "fallback", (), (), is_fallback=True, idiom_cost=0.0)

# Full vocabulary the decoder searches: canonical forms + mirrors + fallback.
VOCABULARY: list[Template] = _CANONICAL + [t.mirror() for t in _CANONICAL] + [FALLBACK]


def templates_of_length(n: int) -> list[Template]:
    """Named (non-fallback) templates that exactly cover n hand strikes.
    The fallback covers any length and is handled separately by the decoder."""
    return [t for t in VOCABULARY if not t.is_fallback and t.length == n]


if __name__ == "__main__":
    # Self-checks + human eyeball.
    sym = {ACCENT: "!", NORMAL: " ", GHOST: "."}
    print(f"{len(_CANONICAL)} canonical + {len(_CANONICAL)} mirrors + 1 fallback "
          f"= {len(VOCABULARY)} templates\n")
    print(f"{'name':20} {'fam':11} len  sticking (accent: ! ghost: .)")
    print("-" * 60)
    for t in _CANONICAL:
        dressed = " ".join(tok + sym[a] for tok, a in zip(t.tokens, t.accents))
        cyc = "○" if t.cyclic else " "
        print(f"{t.name:20} {t.family:11} {t.length:>3} {cyc} {dressed}")

    # Assertions
    para = next(t for t in _CANONICAL if t.name == "paradiddle")
    assert para.accents[0] == ACCENT and para.accents[1:] == (NORMAL,) * 3
    assert para.mirror().tokens == ("L", "R", "L", "L")
    assert para.mirror().accents == para.accents  # accent stays on the (now-left) lead slot
    assert {t.length for t in templates_of_length(4)} == {4}
    assert FALLBACK.mirror() is FALLBACK
    assert all(len(t.tokens) == len(t.accents) for t in VOCABULARY)
    print("\nself-checks passed ✓")
