"""Stage 1 — Onset, surface, velocity via ADTOF, with librosa spectral-flux validation."""

import os

# ADTOF was written for Keras 2; TF 2.16+ ships Keras 3. Route tf.keras to the
# tf_keras shim. Must be set before tensorflow/keras is first imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import glob
import logging
import tempfile
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from .types import Strike, SURFACES

logger = logging.getLogger(__name__)

# Minimum fraction of librosa onsets that ADTOF should recover; below this → warning.
ADTOF_COVERAGE_WARNING_THRESHOLD = 0.70
LIBROSA_ONSET_TOLERANCE_SEC = 0.025  # 25 ms window for matching onsets

# ADTOF emits MIDI drum pitches; collapse them onto our 5 canonical surfaces.
ADTOF_PITCH_TO_SURFACE = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hihat", 44: "hihat", 46: "hihat",
    49: "cymbal", 51: "cymbal", 52: "cymbal", 53: "cymbal", 55: "cymbal", 57: "cymbal", 59: "cymbal",
}
# ADTOF's CRNN framing needs a few seconds of context; pad shorter clips with
# silence and discard onsets that fall in the padded tail.
ADTOF_MIN_DURATION_SEC = 6.0
ADTOF_VELOCITY_WINDOW_MS = 30.0


def detect_events(wav_path: str, adtof_model=None) -> list[Strike]:
    """Run ADTOF on wav_path, validate against librosa, return list[Strike].

    adtof_model: optional pre-loaded model instance (saves reload time across calls).

    When ADTOF is unavailable (not installed / failed), we degrade to librosa
    spectral-flux onsets with a heuristic surface classifier so the rest of the
    pipeline still has events to work with. Surfaces from this path are guesses,
    not ADTOF predictions — the kick/hand split is the part it gets reliably.
    """
    adtof_events = _run_adtof(wav_path, adtof_model)
    librosa_onsets = _run_librosa_fallback(wav_path)

    _validate_coverage(adtof_events, librosa_onsets)

    if adtof_events:
        return adtof_events

    if len(librosa_onsets) == 0:
        return []

    logger.warning(
        "ADTOF unavailable — using librosa fallback as the event source "
        "(%d onsets). Surfaces are heuristic (kick vs hand), not ADTOF.",
        len(librosa_onsets),
    )
    return _onsets_to_strikes(wav_path, librosa_onsets)


def load_adtof_model():
    """Load and return (model, hparams). Call once and reuse across clips."""
    from . import _madmom_compat  # noqa: F401 — patch madmom before adtof imports it
    try:
        from adtof.model.model import Model
    except ImportError as e:
        raise ImportError(
            "adtof not installed. Install from source (Windows: set PYTHONUTF8=1):\n"
            "  pip install --no-deps git+https://github.com/MZehren/ADTOF\n"
            "  pip install tf_keras pandas matplotlib jellyfish pyunpack pretty_midi beautifulsoup4\n"
            "and run with TF_USE_LEGACY_KERAS=1."
        ) from e

    model, hparams = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    if not getattr(model, "weightLoadedFlag", False):
        raise RuntimeError("ADTOF reported its pretrained weights were not restored.")
    logger.info("ADTOF model loaded (Frame_RNN / adtofAll)")
    return model, hparams


def _run_adtof(wav_path: str, model=None) -> list[Strike]:
    try:
        if model is None:
            model = load_adtof_model()
        adtof_model, hparams = model
        return _adtof_predict(wav_path, adtof_model, hparams)
    except Exception as e:
        logger.error("ADTOF failed: %s — falling back to librosa", e)
        return []


def _adtof_predict(wav_path: str, adtof_model, hparams: dict) -> list[Strike]:
    """Run ADTOF's predictFolder, parse its (time, MIDI-pitch) output into Strikes."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    orig_dur = len(y) / sr

    work_dir = tempfile.mkdtemp(prefix="adtof_")
    # ADTOF chokes on very short clips; pad to a minimum length on a temp copy.
    if orig_dur < ADTOF_MIN_DURATION_SEC:
        pad = np.zeros(int((ADTOF_MIN_DURATION_SEC - orig_dur) * sr))
        in_path = os.path.join(work_dir, "in.wav")
        sf.write(in_path, np.concatenate([y, pad]), sr)
    else:
        in_path = wav_path

    out_dir = os.path.join(work_dir, "out")
    adtof_model.predictFolder(in_path, out_dir, writeMidi=False, **hparams)

    txts = glob.glob(os.path.join(out_dir, "*.txt"))
    if not txts:
        logger.warning("ADTOF produced no output file")
        return []

    events: list[tuple[float, str]] = []
    with open(txts[0]) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            t, pitch = float(parts[0]), int(float(parts[1]))
            if t > orig_dur:  # drop onsets in the silent padding
                continue
            surface = ADTOF_PITCH_TO_SURFACE.get(pitch)
            if surface is None:
                logger.debug("Unmapped ADTOF pitch %d", pitch)
                continue
            events.append((t, surface))

    if not events:
        return []

    strikes = _attach_velocity(y, sr, events)
    strikes.sort(key=lambda s: s.time)
    logger.info("ADTOF detected %d events", len(strikes))
    return strikes


def _attach_velocity(y: np.ndarray, sr: int, events: list[tuple[float, str]]) -> list[Strike]:
    """ADTOF gives no velocity; estimate it from peak amplitude at each onset."""
    win = max(8, int(ADTOF_VELOCITY_WINDOW_MS / 1000.0 * sr))
    peaks = []
    for t, _ in events:
        c = int(t * sr)
        seg = y[max(0, c): c + win]
        peaks.append(float(np.max(np.abs(seg))) if len(seg) else 0.0)
    peak_max = max(peaks) or 1.0
    return [
        Strike(time=float(t), surface=s, velocity=float(np.clip(p / peak_max, 0.05, 1.0)))
        for (t, s), p in zip(events, peaks)
    ]


def _run_librosa_fallback(wav_path: str) -> np.ndarray:
    """Return onset times (seconds) from librosa spectral-flux detection."""
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units="time",
            pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.07, wait=10,
        )
        logger.info("librosa fallback detected %d onsets", len(onset_frames))
        return onset_frames
    except Exception as e:
        logger.warning("librosa fallback failed: %s", e)
        return np.array([])


# --- librosa fallback surface classifier ---------------------------------
# Band cutoffs for the heuristic surface guess. A kick dumps most of its energy
# into the sub-bass; hand hits (snare/cymbal) sit far above it. We deliberately
# only split kick-vs-hand — that's the one distinction downstream constraints
# depend on (kick → Foot). Guessing among hand surfaces would inject a spurious
# natural-hand bias, so every non-kick onset is labelled "snare" (neutral prior).
_SUBBASS_HZ = 150.0
_KICK_SUBBASS_FRAC = 0.30   # sub-bass energy fraction above which an onset is a kick
_KICK_MAX_CENTROID = 2500.0  # but reject bright-yet-boomy hits (e.g. low crash)
_FALLBACK_WINDOW_MS = 50.0


def _onsets_to_strikes(wav_path: str, onset_times: np.ndarray) -> list[Strike]:
    """Build Strikes from librosa onsets, guessing surface + velocity from spectra."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    win = max(8, int(_FALLBACK_WINDOW_MS / 1000.0 * sr))

    peaks: list[float] = []
    feats: list[Optional[tuple]] = []
    for t in onset_times:
        center = int(t * sr)
        seg = y[max(0, center): center + win]
        if len(seg) < 8:
            peaks.append(0.0)
            feats.append(None)
            continue
        peaks.append(float(np.max(np.abs(seg))))
        windowed = seg * np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        total = float(spec.sum()) + 1e-10
        sub_frac = float(spec[freqs < _SUBBASS_HZ].sum()) / total
        centroid = float((freqs * spec).sum() / total)
        feats.append((sub_frac, centroid))

    peak_max = max(peaks) or 1.0
    strikes: list[Strike] = []
    for t, peak, feat in zip(onset_times, peaks, feats):
        if feat is None:
            continue
        sub_frac, centroid = feat
        surface = _classify_surface(sub_frac, centroid)
        velocity = float(np.clip(peak / peak_max, 0.05, 1.0))
        strikes.append(Strike(time=float(t), surface=surface, velocity=velocity))

    strikes.sort(key=lambda s: s.time)
    n_kick = sum(1 for s in strikes if s.surface == "kick")
    logger.info("Fallback classified %d strikes (%d kick / %d hand)",
                len(strikes), n_kick, len(strikes) - n_kick)
    return strikes


def _classify_surface(sub_frac: float, centroid: float) -> str:
    """Heuristic surface from band energy: kick vs hand (snare). Nothing finer."""
    if sub_frac > _KICK_SUBBASS_FRAC and centroid < _KICK_MAX_CENTROID:
        return "kick"
    return "snare"


def _validate_coverage(adtof_events: list[Strike], librosa_onsets: np.ndarray):
    """Warn if ADTOF misses a significant chunk of librosa-detected onsets."""
    if len(librosa_onsets) == 0:
        return
    if len(adtof_events) == 0:
        logger.warning("ADTOF produced no events; librosa found %d — check your ADTOF install",
                       len(librosa_onsets))
        return

    adtof_times = np.array([s.time for s in adtof_events])
    matched = 0
    for t in librosa_onsets:
        if np.any(np.abs(adtof_times - t) <= LIBROSA_ONSET_TOLERANCE_SEC):
            matched += 1

    coverage = matched / len(librosa_onsets)
    if coverage < ADTOF_COVERAGE_WARNING_THRESHOLD:
        logger.warning(
            "ADTOF covers only %.0f%% of librosa onsets (threshold %.0f%%). "
            "Check audio quality or ADTOF model.",
            coverage * 100, ADTOF_COVERAGE_WARNING_THRESHOLD * 100,
        )
    else:
        logger.info("ADTOF/librosa onset coverage: %.0f%%", coverage * 100)


def extract_attack_envelope(wav_path: str, onset_time: float, window_ms: float = 30.0) -> dict:
    """Extract attack-envelope features for a single onset — used in Stage 5 double-stroke detection."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    center = int(onset_time * sr)
    win = int(window_ms / 1000 * sr)
    segment = y[max(0, center): center + win]
    if len(segment) == 0:
        return {"peak_amplitude": 0.0, "rise_time_ms": 0.0, "spectral_centroid": 0.0}

    peak_amp = float(np.max(np.abs(segment)))
    # Rise time: samples from start to peak
    peak_idx = int(np.argmax(np.abs(segment)))
    rise_time_ms = peak_idx / sr * 1000.0
    # Spectral centroid of attack window
    if peak_idx > 4:
        spec = np.abs(np.fft.rfft(segment[:peak_idx + 1]))
        freqs = np.fft.rfftfreq(peak_idx + 1, 1 / sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-10))
    else:
        centroid = 0.0

    return {
        "peak_amplitude": peak_amp,
        "rise_time_ms": rise_time_ms,
        "spectral_centroid": centroid,
    }
