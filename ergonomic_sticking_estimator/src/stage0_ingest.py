"""Stage 0 — Ingest: accept audio/video, optionally separate drum stem with Demucs."""

import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

ADTOF_SR = 44100  # ADTOF and madmom both expect 44.1 kHz


def ingest(input_path: str, separate: bool = "auto", output_dir: str = None) -> tuple[str, bool]:
    """Return (wav_path, was_separated).

    separate='auto': run Demucs if the file appears to be a mixed recording
                     (heuristic: energy in non-drum frequency bands is high).
    separate=True:   always separate.
    separate=False:  never separate.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="dse_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio from video if needed
    audio_path = _extract_audio_if_needed(input_path, out_dir)

    # Load to check whether separation is warranted
    y, sr = librosa.load(str(audio_path), sr=ADTOF_SR, mono=True)

    should_separate = _decide_separation(separate, y, sr)

    if should_separate:
        drum_path = _run_demucs(audio_path, out_dir)
        if drum_path is None:
            logger.warning("Demucs failed; proceeding with original audio")
            should_separate = False
            drum_path = audio_path
    else:
        drum_path = audio_path

    # Normalize and save canonical WAV
    final_path = out_dir / "drum_stem.wav"
    y_out, sr_out = librosa.load(str(drum_path), sr=ADTOF_SR, mono=True)
    y_out = _normalize(y_out)
    sf.write(str(final_path), y_out, sr_out)

    logger.info("Ingest complete: %s (separated=%s)", final_path, should_separate)
    return str(final_path), should_separate


def _ffmpeg_exe() -> str:
    """Path to an ffmpeg binary. Prefer the one bundled with imageio-ffmpeg
    so we don't depend on a system install / PATH; fall back to 'ffmpeg'."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_audio_if_needed(path: Path, out_dir: Path) -> Path:
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if path.suffix.lower() not in video_exts:
        return path

    out_wav = out_dir / "extracted_audio.wav"
    cmd = [_ffmpeg_exe(), "-y", "-i", str(path), "-ac", "1", "-ar", str(ADTOF_SR), str(out_wav)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return out_wav


def _decide_separation(mode, y: np.ndarray, sr: int) -> bool:
    if mode is True:
        return True
    if mode is False:
        return False
    # Auto: estimate non-drum energy ratio using a simple spectral heuristic.
    # Drum energy concentrates in broadband transients; tonal/melodic content
    # shows up as stable harmonic peaks. If harmonic-to-percussive ratio is high,
    # the mix likely has melodic instruments.
    import librosa
    harm, perc = librosa.effects.hpss(y)
    harm_energy = np.mean(harm ** 2)
    perc_energy = np.mean(perc ** 2) + 1e-10
    ratio = harm_energy / perc_energy
    logger.info("Harmonic/percussive energy ratio: %.3f", ratio)
    return ratio > 0.15  # empirical threshold; tune on your test set


def _run_demucs(audio_path: Path, out_dir: Path) -> Path | None:
    try:
        import sys
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=drums",
            "-o", str(out_dir),
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error("Demucs stderr: %s", result.stderr.decode())
            return None
        # Demucs writes to out_dir/<model>/<filename>/drums.wav
        candidates = list(out_dir.rglob("drums.wav"))
        if not candidates:
            logger.error("Demucs ran but produced no drums.wav")
            return None
        return candidates[0]
    except Exception as e:
        logger.error("Demucs exception: %s", e)
        return None


def _normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    return y / peak * 0.95
