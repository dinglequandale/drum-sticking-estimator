"""Stage 2 — Beat/tempo/grid via madmom DBN downbeat tracker.

Quantizes each Strike to the nearest metrical subdivision and annotates
it with local tempo. Tempo is passed per-event downstream because it
gates hard constraints (Stage 4) and tilts the singles-vs-doubles prior
(Stage 5).
"""

import logging
from typing import Optional

import numpy as np

from .types import Strike

logger = logging.getLogger(__name__)

SUBDIVISIONS = [4, 8, 12, 16, 24, 32]  # tested in order; first that fits is used


def add_grid(wav_path: str, strikes: list[Strike]) -> tuple[list[Strike], dict]:
    """Add quantized_pos to each strike. Also return beat_info dict.

    beat_info keys: beats, downbeats, tempo_curve (time->bpm), meter
    """
    beats, downbeats, tempo_curve = _run_madmom(wav_path)
    meter = _infer_meter(beats, downbeats)
    subdivision = _detect_subdivision(strikes, beats, meter)

    for strike in strikes:
        strike.quantized_pos = _quantize(strike.time, beats, downbeats, subdivision, meter)

    logger.info("Beat tracking: meter=%s, subdivision=1/%d, %d beats",
                meter, subdivision, len(beats))
    return strikes, {
        "beats": beats,
        "downbeats": downbeats,
        "tempo_curve": tempo_curve,
        "meter": meter,
        "subdivision": subdivision,
    }


def get_local_tempo(beat_info: dict, time: float) -> float:
    """Return interpolated BPM at `time` in seconds."""
    tc = beat_info["tempo_curve"]
    if len(tc) == 0:
        return 120.0
    times = np.array([t for t, _ in tc])
    bpms = np.array([b for _, b in tc])
    return float(np.interp(time, times, bpms))


def _run_madmom(wav_path: str) -> tuple[np.ndarray, np.ndarray, list]:
    try:
        from . import _madmom_compat  # noqa: F401  (restores names madmom 0.16.1 needs)
        import madmom
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

        act = RNNDownBeatProcessor()(wav_path)
        # RNNDownBeatProcessor emits activations at 100 fps; the DBN tracker
        # must be told the frame rate or it can't convert bpm bounds to frames.
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        # proc: array of (time, beat_number) where beat_number resets at downbeats

        beats = proc[:, 0]
        beat_nums = proc[:, 1]
        downbeats = beats[beat_nums == 1]

        tempo_curve = _estimate_tempo_curve(beats)
        return beats, downbeats, tempo_curve

    except ImportError as e:
        raise ImportError(
            "madmom not found. Install with:\n"
            "  pip install Cython numpy<2 && pip install madmom"
        ) from e
    except Exception as e:
        logger.error("madmom failed: %s — using librosa fallback for beats", e)
        return _librosa_beat_fallback(wav_path)


def _librosa_beat_fallback(wav_path: str) -> tuple[np.ndarray, np.ndarray, list]:
    import librosa
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = beat_frames
    downbeats = beats[::4]  # assume 4/4
    tempo_curve = [(0.0, float(tempo))]
    logger.warning("Using librosa beat fallback (less accurate than madmom)")
    return beats, downbeats, tempo_curve


def _estimate_tempo_curve(beats: np.ndarray) -> list[tuple[float, float]]:
    """Return list of (time, bpm) pairs, one per inter-beat interval."""
    if len(beats) < 2:
        return [(0.0, 120.0)]
    curve = []
    for i in range(len(beats) - 1):
        ioi = beats[i + 1] - beats[i]
        bpm = 60.0 / ioi if ioi > 0 else 120.0
        curve.append((float(beats[i]), float(bpm)))
    return curve


def _infer_meter(beats: np.ndarray, downbeats: np.ndarray) -> str:
    """Guess '3/4' or '4/4' from beats-per-bar."""
    if len(downbeats) < 2:
        return "4/4"
    # Count how many beats fall between consecutive downbeats
    counts = []
    for i in range(min(len(downbeats) - 1, 8)):
        t0, t1 = downbeats[i], downbeats[i + 1]
        n = np.sum((beats >= t0) & (beats < t1))
        counts.append(n)
    median_count = int(np.median(counts)) if counts else 4
    return f"{median_count}/4"


def _detect_subdivision(strikes: list[Strike], beats: np.ndarray, meter: str) -> int:
    """Find the smallest subdivision that aligns with actual hits."""
    if len(strikes) == 0 or len(beats) < 2:
        return 16
    beat_dur = float(np.median(np.diff(beats)))
    for subdiv in SUBDIVISIONS:
        grid_unit = beat_dur / (subdiv / 4)
        errors = []
        for s in strikes:
            beat_phase = (s.time % beat_dur) / grid_unit
            error = abs(beat_phase - round(beat_phase)) * grid_unit
            errors.append(error)
        mean_err = np.mean(errors)
        if mean_err < 0.020:  # 20 ms tolerance
            return subdiv
    return 16


def _quantize(time: float, beats: np.ndarray, downbeats: np.ndarray,
              subdivision: int, meter: str) -> Optional[float]:
    """Return a float encoding bar.beat_fraction, e.g. 2.75 = bar 2 beat 3 of 4."""
    if len(beats) < 2:
        return None
    beat_dur = float(np.median(np.diff(beats)))
    grid = beat_dur / (subdivision / 4)

    # Find nearest beat
    idx = int(np.argmin(np.abs(beats - time)))
    beat_time = beats[idx]

    # Count bars
    if len(downbeats) > 0:
        bar_idx = int(np.searchsorted(downbeats, beat_time, side="right")) - 1
        bar_idx = max(bar_idx, 0)
    else:
        bar_idx = 0

    beats_per_bar = int(meter.split("/")[0])
    beat_in_bar = (idx % beats_per_bar) + 1  # 1-indexed

    # Sub-beat fraction
    offset = time - beat_time
    frac = round(offset / grid) / (subdivision / 4)

    return float(bar_idx + 1) + (beat_in_bar - 1 + frac) / beats_per_bar
