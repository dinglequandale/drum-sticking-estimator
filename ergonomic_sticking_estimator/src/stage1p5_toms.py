"""Stage 1.5 — tom sub-classification (built on top of ADTOF).

ADTOF's `adtofAll` model was trained on the reduced 5-class taxonomy, so every tom collapses
to a single `tom` class. But hand assignment around the kit needs to know *which* tom (a
left->right sweep is the signal behind "three-across -> RLL"). We recover that here without
any training data and without fixed pitch cutoffs:

    1. estimate each tom hit's pitch (spectral peak in the tom band),
    2. CLUSTER the clip's tom pitches unsupervised (adaptive to this kit's tuning),
    3. rank clusters by pitch and relabel surfaces `tom0` (highest) .. `tomN` (lowest).

High pitch = rack/high tom (kit left); low pitch = floor tom (kit right) — the ordering the
reach model (step 4) maps to X positions. The only fixed numbers are a physical *search range*
for the estimator and a *dimensionless* cluster-quality floor, both tuning-invariant.

Runnable: `python -m src.stage1p5_toms` runs synthetic clustering checks (no audio needed).
"""

import logging

import numpy as np

from .types import Strike, PipelineConfig

logger = logging.getLogger(__name__)


def _tom_pitch(y: np.ndarray, sr: int, onset_time: float, window_ms: float,
               fmin: float, fmax: float):
    """Dominant pitch (Hz) in the tom band over a window after the onset, or None.
    Spectral-peak based — robust on inharmonic drum hits where YIN is unstable."""
    center = int(onset_time * sr)
    win = int(window_ms / 1000.0 * sr)
    seg = y[center: center + win]
    if len(seg) < 64:
        return None
    seg = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or spec[band].max() <= 0:
        return None
    return float(freqs[band][np.argmax(spec[band])])


def _cluster_pitches(pitches: list, max_k: int, sil_floor: float):
    """Unsupervised per-clip clustering of tom pitches. Returns (labels, k).

    Clusters in log-Hz (pitch is perceived multiplicatively). Picks k in 1..max_k by the best
    silhouette score, accepting a split only if it clears `sil_floor` — a dimensionless quality
    gate, so it adapts to any tuning rather than using fixed pitch boundaries."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = np.log(np.array(pitches)).reshape(-1, 1)
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=int), 1

    best_k, best_labels, best_sil = 1, np.zeros(n, dtype=int), -1.0
    for k in range(2, min(max_k, n) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
        if len(set(labels)) < k:
            continue
        sil = silhouette_score(x, labels)
        if sil > best_sil:
            best_k, best_labels, best_sil = k, labels, sil

    if best_k > 1 and best_sil >= sil_floor:
        return best_labels, best_k
    return np.zeros(n, dtype=int), 1


def subclassify_toms(strikes: list, wav_path: str, config: PipelineConfig) -> list:
    """Relabel `tom` strikes to `tom0`..`tomN` (highest->lowest pitch) via per-clip clustering.
    Returns a new strike list; non-tom strikes and unresolved toms are left untouched."""
    if not config.subclassify_toms:
        return strikes
    tom_idx = [i for i, s in enumerate(strikes) if s.surface == "tom"]
    if len(tom_idx) < 2:
        return strikes

    import librosa
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    valid, pitches = [], []
    for i in tom_idx:
        p = _tom_pitch(y, sr, strikes[i].time, config.tom_pitch_window_ms,
                       config.tom_pitch_fmin, config.tom_pitch_fmax)
        if p is not None:
            valid.append(i)
            pitches.append(p)
    if len(valid) < 2:
        return strikes

    labels, k = _cluster_pitches(pitches, config.tom_cluster_max, config.tom_cluster_sil_floor)
    if k == 1:
        logger.info("Tom sub-classification: %d tom hits stayed one tom (kit not split)", len(valid))
        return strikes

    # Rank clusters by mean pitch, high -> low, so tom0 is the highest (rack) tom.
    cluster_mean = {c: float(np.mean([pitches[j] for j in range(len(labels)) if labels[j] == c]))
                    for c in set(labels)}
    rank = {c: r for r, c in enumerate(sorted(cluster_mean, key=lambda c: -cluster_mean[c]))}

    new = list(strikes)
    for j, i in enumerate(valid):
        s = strikes[i]
        new[i] = Strike(time=s.time, surface=f"tom{rank[labels[j]]}",
                        velocity=s.velocity, quantized_pos=s.quantized_pos)
    ranked_hz = [round(cluster_mean[c]) for c in sorted(cluster_mean, key=lambda c: -cluster_mean[c])]
    logger.info("Tom sub-classification: %d tom hits -> %d sub-toms at ~%s Hz", len(valid), k, ranked_hz)
    return new


if __name__ == "__main__":
    # Synthetic checks for the clustering logic (no audio).
    cfg = PipelineConfig()

    def labels_k(pitches):
        return _cluster_pitches(pitches, cfg.tom_cluster_max, cfg.tom_cluster_sil_floor)

    # Two well-separated toms (high ~200 Hz, floor ~90 Hz) -> k=2
    _, k2 = labels_k([205, 198, 210, 202, 92, 88, 95, 90])
    print("two-tom kit -> k =", k2)
    assert k2 == 2

    # One tom, tight spread -> k=1 (no spurious split)
    _, k1 = labels_k([150, 152, 148, 151, 149, 150])
    print("one-tom kit -> k =", k1)
    assert k1 == 1

    # Three toms -> k=3
    _, k3 = labels_k([260, 255, 258, 175, 170, 178, 95, 90, 92])
    print("three-tom kit -> k =", k3)
    assert k3 == 3

    print("\ntom clustering checks passed ✓")
