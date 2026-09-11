"""Build the seed sticking corpus (corpus/rudiments.jsonl) from a compact,
auditable notation. This transcribes only the CONFIDENT (✓) rudiments from
DRUM_GRAMMAR.md plus one enumeration row read directly off Stick Control p6.
The ⚠ grace-note rudiments are deliberately omitted until verified against the
PDFs. Run:  ./.venv/Scripts/python.exe corpus/build_seed.py

Notation (per token, space-separated; spaces are cosmetic and ignored):
    R  L  F        tap with right / left / foot
    R! L!          accented stroke
    (l) (r)        one grace note (flam) in that hand, attached to the next primary
    (ll) (rr)      two grace notes (drag) in that hand, attached to the next primary
A phrase string is parsed left→right into played-order stroke objects; grace
notes are emitted as their own strokes (grace=true) immediately before the
primary they decorate. `pos` is the running stroke index (uniform grid).
"""

import json
import re
from pathlib import Path

OUT = Path(__file__).with_name("rudiments.jsonl")

# (family, lead, notation, provenance, subdivision, cyclic)
# Only ✓-confident forms. R-lead and L-lead written separately (mirrors are
# distinct phrases — lead hand is contextual, never assumed; see DRUM_GRAMMAR.md).
PHRASES = [
    # --- Roll family (rudiment) ---
    ("single_stroke_roll",  "R", "R L R L R L R L",            "rudiment", "16th", True),
    ("double_stroke_roll",  "R", "R R L L R R L L",            "rudiment", "16th", True),
    ("triple_stroke_roll",  "R", "R R R L L L",                "rudiment", "triplet", True),
    # odd rolls: k doubles + accented release single, then it mirrors on repeat
    ("five_stroke_roll",    "R", "R R L L R!  L L R R L!",     "rudiment", "16th", True),
    ("seven_stroke_roll",   "R", "R R L L R R L!",             "rudiment", "triplet", False),
    ("nine_stroke_roll",    "R", "R R L L R R L L R!",         "rudiment", "16th", False),
    # --- Paradiddle family (rudiment) — diddle flips the lead; full cycle written ---
    ("single_paradiddle",   "R", "R! L R R  L! R L L",         "rudiment", "16th", True),
    ("double_paradiddle",   "R", "R! L R L R R  L! R L R L L", "rudiment", "triplet", True),
    ("triple_paradiddle",   "R", "R! L R L R L R R  L! R L R L R L L", "rudiment", "16th", True),
    # paradiddle-diddle keeps the lead (no flip); mirror is a separate phrase
    ("single_paradiddle_diddle", "R", "R! L R R L L",          "rudiment", "triplet", True),
    ("single_paradiddle_diddle", "L", "L! R L L R R",          "rudiment", "triplet", True),
    # --- Flam family (✓ only) ---
    ("flam",                "R", "(l)R!",                      "rudiment", "8th", False),
    ("flam",                "L", "(r)L!",                      "rudiment", "8th", False),
    ("flam_tap",            "R", "(l)R! R  (r)L! L",           "rudiment", "16th", True),
    ("flam_accent",         "R", "(l)R! L R  (r)L! R L",       "rudiment", "triplet", True),
    ("flam_paradiddle",     "R", "(l)R! L R R  (r)L! R L L",   "rudiment", "16th", True),
    ("swiss_army_triplet",  "R", "(l)R! R L  (r)L! L R",       "rudiment", "triplet", True),
    # --- Drag family (✓ only) ---
    ("drag",                "R", "(ll)R",                      "rudiment", "8th", False),
    ("drag",                "L", "(rr)L",                      "rudiment", "8th", False),
    ("single_drag_tap",     "R", "(ll)R! L  (rr)L! R",         "rudiment", "16th", True),
    # --- Enumeration sample (Stick Control p6, ex.25 — read directly) ---
    ("single_beat_combination", "R", "R L R R  L L R R  L L R R  L R L L",
                                                               "enumeration", "16th", False),
]

# Applied (genuine in-context choices, surface-mapped). CURATED: one distinct
# idea per entry — never permutation runs (see corpus/schema.md). Surfaces come
# from a per-phrase {limb: surface} map. (family, lead, notation, subdivision,
# cyclic, surface_map)
APPLIED_PHRASES = [
    # Garibaldi, Future Sounds p.9 (read directly): the foundational "two sound
    # levels" idea — a single paradiddle split across the kit, RH on hi-hat, LH on
    # snare. Teaches surface->hand (hat->R, snare->L) for this linear voicing.
    ("two_sound_level_paradiddle", "R", "R L R R L R L L", "8th", True,
        {"R": "hihat", "L": "snare"}),
]

TOKEN = re.compile(r"\((l+|r+)\)|([RLF])(!?)")


def parse(notation: str, surface_map: dict | None = None):
    def surf(limb):
        return (surface_map or {}).get(limb, "snare")
    strokes = []
    pos = 0
    for m in TOKEN.finditer(notation):
        grace_run, limb, bang = m.group(1), m.group(2), m.group(3)
        if grace_run:  # one or two grace notes (flam / drag), in this hand
            hand = "L" if grace_run[0] == "l" else "R"
            for _ in grace_run:
                strokes.append({"limb": hand, "surface": surf(hand),
                                "accent": False, "grace": True, "pos": pos})
        else:
            strokes.append({"limb": limb, "surface": surf(limb),
                            "accent": bang == "!", "grace": False, "pos": pos})
            pos += 1  # only primaries advance the metric grid
    return strokes


def main():
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, (family, lead, notation, prov, subdiv, cyclic) in enumerate(PHRASES):
            rec = {
                "source": f"seed_{prov}_{i:02d}_{family}",
                "provenance": prov,
                "family": family,
                "lead": lead,
                "subdivision": subdiv,
                "speed_tier": None,
                "cyclic": cyclic,
                "strokes": parse(notation),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
        for j, (family, lead, notation, subdiv, cyclic, smap) in enumerate(APPLIED_PHRASES):
            rec = {
                "source": f"seed_applied_{j:02d}_{family}",
                "provenance": "applied",
                "family": family,
                "lead": lead,
                "subdivision": subdiv,
                "speed_tier": None,
                "cyclic": cyclic,
                "strokes": parse(notation, smap),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} phrases -> {OUT}")


if __name__ == "__main__":
    main()
