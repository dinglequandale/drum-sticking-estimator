"""Stage 7 — Pattern extraction and output formatting.

Produces:
  1. A list of AssignedStrike objects (the source-of-truth JSON stream).
  2. A human-readable grid (text tab aligned to the metrical grid).
  3. Recurring sticking patterns with bar/beat locations called out.
  4. Every irreducibly ambiguous hit is explicitly marked — never silently guessed.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict

from .types import AssignedStrike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

def extract_patterns(assigned: list[AssignedStrike], min_occurrences: int = 2) -> list[dict]:
    """Find recurring sticking subsequences and report their locations."""
    tokens = [a.limb for a in assigned if a.limb in ("L", "R")]
    positions = [a.quantized_pos for a in assigned if a.limb in ("L", "R")]

    patterns = []
    for length in range(4, min(len(tokens), 16) + 1):
        seen: dict[str, list] = defaultdict(list)
        for i in range(len(tokens) - length + 1):
            seq = "".join(tokens[i: i + length])
            seen[seq].append(positions[i])
        for seq, locs in seen.items():
            if len(locs) >= min_occurrences:
                patterns.append({
                    "sticking": seq,
                    "length": length,
                    "occurrences": len(locs),
                    "positions": [p for p in locs if p is not None],
                    "rudiment_match": _match_rudiment(seq),
                })

    # Deduplicate: remove patterns fully contained in longer ones
    patterns = _deduplicate_patterns(patterns)
    patterns.sort(key=lambda p: -p["occurrences"])
    return patterns


def _match_rudiment(seq: str) -> str | None:
    """Return a rudiment name if the pattern matches, else None."""
    canonical = {
        "RLRL": "single-stroke roll",
        "LRLR": "single-stroke roll",
        "RRLL": "double-stroke roll",
        "LLRR": "double-stroke roll",
        "RLRR": "paradiddle",
        "LRLL": "paradiddle",
        "LRLL": "paradiddle (L)",
        "RLLR": "paradiddle-diddle seed",
        "RLRRLRLL": "double paradiddle",
        "LRLLRLRR": "double paradiddle (L)",
    }
    return canonical.get(seq)


def _deduplicate_patterns(patterns: list[dict]) -> list[dict]:
    patterns_sorted = sorted(patterns, key=lambda p: -p["length"])
    kept = []
    for p in patterns_sorted:
        dominated = any(p["sticking"] in q["sticking"] for q in kept)
        if not dominated:
            kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def to_json(assigned: list[AssignedStrike], patterns: list[dict], beat_info: dict) -> str:
    data = {
        "hits": [_hit_to_dict(a) for a in assigned],
        "patterns": patterns,
        "meta": {
            "total_hits": len(assigned),
            "flagged_ambiguous": sum(1 for a in assigned if a.flagged_ambiguous),
            "tempo_bpm": _mean_tempo(beat_info),
            "meter": beat_info.get("meter", "unknown"),
        },
    }
    return json.dumps(data, indent=2)


def _hit_to_dict(a: AssignedStrike) -> dict:
    return {
        "time": round(a.time, 4),
        "quantized_pos": round(a.quantized_pos, 4) if a.quantized_pos is not None else None,
        "limb": a.limb,
        "confidence": round(a.confidence, 3),
        "source": a.source,
        "flagged_ambiguous": a.flagged_ambiguous,
        "limb_probs": {k: round(v, 3) for k, v in a.limb_probs.items()},
    }


def _mean_tempo(beat_info: dict) -> float:
    tc = beat_info.get("tempo_curve", [])
    if not tc:
        return 0.0
    import numpy as np
    return float(np.mean([b for _, b in tc]))


# ---------------------------------------------------------------------------
# Human-readable grid
# ---------------------------------------------------------------------------

def to_grid(assigned: list[AssignedStrike], beat_info: dict) -> str:
    """Render a text piano-roll style grid aligned to beats."""
    meter = beat_info.get("meter", "4/4")
    subdivision = beat_info.get("subdivision", 16)
    beats_per_bar = int(meter.split("/")[0])
    cells_per_bar = beats_per_bar * (subdivision // 4)

    # Group hits by bar
    bars: dict[int, list] = defaultdict(list)
    for hit in assigned:
        if hit.quantized_pos is None:
            continue
        bar_num = int(hit.quantized_pos)
        bars[bar_num].append(hit)

    if not bars:
        return "(no quantized hits)"

    lines = []
    header = f"Sticking grid | {meter} | 1/{subdivision} resolution | ? = ambiguous\n"
    lines.append(header)
    lines.append(f"{'Bar':<4} | {'Sticking (L/R/F/?)'}")
    lines.append("-" * 60)

    for bar_num in sorted(bars):
        hits_in_bar = sorted(bars[bar_num], key=lambda h: h.quantized_pos)
        grid_cells = ["."] * cells_per_bar

        for hit in hits_in_bar:
            frac = hit.quantized_pos - bar_num
            cell = int(round(frac * cells_per_bar))
            cell = max(0, min(cell, cells_per_bar - 1))
            label = "?" if hit.flagged_ambiguous else hit.limb
            grid_cells[cell] = label

        row = " ".join(grid_cells)
        lines.append(f"{bar_num:<4} | {row}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern summary
# ---------------------------------------------------------------------------

def pattern_summary(patterns: list[dict]) -> str:
    if not patterns:
        return "No recurring patterns detected."
    lines = ["Recurring sticking patterns:"]
    for p in patterns[:10]:
        rudiment = f" ({p['rudiment_match']})" if p["rudiment_match"] else ""
        pos_str = ", ".join(f"{x:.2f}" for x in p["positions"][:5])
        if len(p["positions"]) > 5:
            pos_str += "..."
        lines.append(
            f"  {p['sticking']}{rudiment} — {p['occurrences']}x at bars: {pos_str}"
        )
    return "\n".join(lines)
