"""Stage 6 — Self-similarity propagation.

Secondary lift, not the backbone. When the drummer repeats a figure,
resolve the unambiguous instance first, then propagate its sticking to
ambiguous repeats. Real playing is not repetitive enough for this to
carry the system.

Similarity is defined over (quantized inter-onset pattern, surface sequence)
ONLY — never over the sticking itself. Matching on sticking would be circular.
"""

import logging
from collections import defaultdict

import numpy as np

from .types import AssignedStrike

logger = logging.getLogger(__name__)

MIN_PATTERN_LENGTH = 4    # minimum time-slots for a repeatable figure
MIN_CLUSTER_SIZE = 2      # minimum occurrences to form a cluster


def propagate(assigned: list[AssignedStrike], config) -> list[AssignedStrike]:
    """Find repeated figures; propagate high-confidence stickings to ambiguous matches."""

    # Group hits into bars/phrases for matching
    tokens = _to_skeleton_tokens(assigned)
    if len(tokens) < MIN_PATTERN_LENGTH * 2:
        logger.info("Stage 6: too few events for self-similarity (%d)", len(tokens))
        return assigned

    clusters = _find_recurring_figures(tokens)
    if not clusters:
        logger.info("Stage 6: no recurring figures found")
        return assigned

    n_propagated = 0
    for cluster in clusters:
        anchor_idx = _find_best_anchor(cluster, assigned)
        if anchor_idx is None:
            continue

        anchor_sticking = [assigned[i].limb for i in cluster[anchor_idx]]

        for occurrence_idx, occurrence in enumerate(cluster):
            if occurrence_idx == anchor_idx:
                continue
            if all(assigned[i].confidence >= config.ambiguity_threshold for i in occurrence):
                continue  # already confident; don't overwrite

            for j, hit_idx in enumerate(occurrence):
                if j >= len(anchor_sticking):
                    break
                hit = assigned[hit_idx]
                if hit.confidence >= config.ambiguity_threshold:
                    continue
                # Propagate only if the skeleton (surface) matches
                anchor_hit_idx = cluster[anchor_idx][j] if j < len(cluster[anchor_idx]) else None
                if anchor_hit_idx is None:
                    continue
                if assigned[anchor_hit_idx].surface != hit.surface:
                    continue

                new_limb = anchor_sticking[j]
                assigned[hit_idx] = AssignedStrike(
                    time=hit.time,
                    surface=hit.surface,
                    velocity=hit.velocity,
                    quantized_pos=hit.quantized_pos,
                    limb=new_limb,
                    limb_probs={new_limb: 0.75, _other_hand(new_limb): 0.25},
                    confidence=0.75,
                    source="propagation",
                    flagged_ambiguous=False,
                )
                n_propagated += 1

    logger.info("Stage 6: propagated sticking to %d hits across %d clusters",
                n_propagated, len(clusters))
    return assigned


def _to_skeleton_tokens(hits: list[AssignedStrike]) -> list[tuple]:
    """Convert hit list to (surface, ioi_quantized) tokens — no sticking included."""
    tokens = []
    for i, hit in enumerate(hits):
        if i == 0:
            ioi_q = 0
        else:
            raw_ioi = hit.time - hits[i - 1].time
            ioi_q = _quantize_ioi(raw_ioi)
        tokens.append((hit.surface, ioi_q))
    return tokens


def _quantize_ioi(ioi_sec: float) -> int:
    """Bin inter-onset interval into coarse categories for matching."""
    ms = ioi_sec * 1000
    if ms < 75:   return 1
    if ms < 150:  return 2
    if ms < 250:  return 3
    if ms < 400:  return 4
    return 5


def _find_recurring_figures(tokens: list[tuple]) -> list[list[list[int]]]:
    """Find recurring n-gram patterns using a sliding window approach.

    Returns list of clusters; each cluster is a list of occurrences;
    each occurrence is a list of hit indices.
    """
    clusters = []
    n = len(tokens)

    for length in range(MIN_PATTERN_LENGTH, min(n // 2, 16) + 1):
        occurrences: dict[tuple, list[int]] = defaultdict(list)
        for start in range(n - length + 1):
            pattern = tuple(tokens[start: start + length])
            occurrences[pattern].append(start)

        for pattern, starts in occurrences.items():
            if len(starts) < MIN_CLUSTER_SIZE:
                continue
            # Filter overlapping occurrences
            non_overlapping = _remove_overlaps(starts, length)
            if len(non_overlapping) < MIN_CLUSTER_SIZE:
                continue
            cluster = [list(range(s, s + length)) for s in non_overlapping]
            clusters.append(cluster)

    # Remove clusters dominated by a longer cluster covering the same hits
    clusters = _deduplicate_clusters(clusters)
    return clusters


def _remove_overlaps(starts: list[int], length: int) -> list[int]:
    result = []
    last_end = -1
    for s in sorted(starts):
        if s >= last_end:
            result.append(s)
            last_end = s + length
    return result


def _deduplicate_clusters(clusters: list) -> list:
    if not clusters:
        return clusters
    # Keep only the longest cluster for any given set of hit indices
    seen_anchors = set()
    unique = []
    for cluster in sorted(clusters, key=lambda c: -len(c[0])):
        anchor_key = tuple(cluster[0])
        if anchor_key not in seen_anchors:
            seen_anchors.add(anchor_key)
            unique.append(cluster)
    return unique


def _find_best_anchor(cluster: list[list[int]], assigned: list[AssignedStrike]) -> int | None:
    """Return the index of the occurrence with highest mean confidence."""
    best_score = -1.0
    best_idx = None
    for occ_idx, occurrence in enumerate(cluster):
        mean_conf = float(np.mean([assigned[i].confidence for i in occurrence if i < len(assigned)]))
        if mean_conf > best_score:
            best_score = mean_conf
            best_idx = occ_idx
    return best_idx


def _other_hand(limb: str) -> str:
    return "R" if limb == "L" else "L"
