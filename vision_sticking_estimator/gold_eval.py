"""CAVEATED real-audio sanity check for sample_1 (see session decision).

Purpose: prove the real-audio eval loop runs end-to-end (ADTOF -> stages ->
Stage-5 sticking -> eval vs ground truth) and get a first, heavily-caveated real
number for the grammar prior ON vs OFF.

GROUND-TRUTH CAVEATS (do not over-trust this number):
  - sample_1 is a single strict-alternation pattern ("R L ... R L R L"), the
    easiest sticking case — low discriminating power for the grammar, whose value
    is on doubles/handoffs (absent here).
  - The on-screen notation is too low-res to transcribe exactly (read ~15 hits;
    audio shows 17 onsets; spectral kick-classification finds ~4). GT here is
    BUILT FROM AUDIO, not the notation: onsets are classified kick(F)/snare by
    low-band energy, snare hits get the notation's strict R/L alternation. Kick
    classification and alternation parity are approximate.

A discriminating verdict needs higher-res clips containing doubles/rudiments.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np

GOLD = "samples/sample_1.gold.json"


def extract():
    """Real onsets + per-onset velocity + kick/snare surface + GT limbs from
    sample_1 audio. GT: kick->F, snare->strict R/L alternation (from notation).
    Bypasses ADTOF/madmom (numpy-2 ABI break); librosa runs fine on numpy 2."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    wav = Path(tempfile.mkdtemp()) / "a.wav"
    subprocess.run([ff, "-y", "-i", "samples/sample_1.mp4", "-ac", "1", "-ar", "44100", str(wav)],
                   capture_output=True)
    y, sr = librosa.load(str(wav), sr=44100, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True, hop_length=512)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = freqs < 150
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    env = env / (env.max() + 1e-9)

    hits, ann, parity = [], [], 0
    for t in onsets:
        f = min(max(int(round(t * sr / 512)), 0), S.shape[1] - 1)
        col = S[:, f]
        ratio = float(col[low].sum() / (col.sum() + 1e-9))
        vel = float(np.clip(env[f], 0.0, 1.0))
        if ratio > 0.30:
            surface, limb, tier = "kick", "F", "easy"
        else:
            surface, limb, tier = "snare", ("R" if parity % 2 == 0 else "L"), "hard"
            parity += 1
        hits.append((float(t), surface, vel, limb))
        ann.append({"time": round(float(t), 3), "limb": limb, "tier": tier, "surface": surface})
    Path(GOLD).write_text(json.dumps(ann, indent=2))
    print(f"wrote {GOLD}: {len(ann)} hits "
          f"({sum(a['limb']=='F' for a in ann)} kick, {sum(a['limb'] in 'LR' for a in ann)} hand)")
    return hits


def main():
    hits = extract()
    from src.types import Strike, TimeSlot, PipelineConfig
    from src.stage4_constraints import apply_constraints
    from src.stage5_hmm import assign_limbs
    from src.stage5b_priors import build_tier1_prior

    # Build Stage-5 input from REAL onsets (mirrors synth_eval._to_stage5_input).
    slots = [TimeSlot(t, [Strike(t, surf, vel, None)]) for (t, surf, vel, _limb) in hits]
    iois = np.diff([t for t, *_ in hits])
    bpm = 60.0 / (2 * float(np.median(iois)))  # treat median IOI as an 8th note
    beat_info = {"tempo_curve": [(0.0, bpm), (hits[-1][0] + 1.0, bpm)],
                 "meter": "4/4", "subdivision": 8}
    gt = [limb for *_, limb in hits]

    print(f"\nreal-audio Stage-5 sanity check (bpm~{bpm:.0f}, ADTOF bypassed)")
    print(f"{'run':4} {'hand_acc':>9} {'flagged':>8}")
    for label, w in (("OFF", 0.0), ("ON", 0.3)):
        cfg = PipelineConfig()
        cfg.rudiment_prior_weight = w
        cds = apply_constraints(slots, cfg, beat_info)
        prior = build_tier1_prior(weight_cap=w)
        assigned = assign_limbs(cds, beat_info, None, cfg, prior)
        pred = {round(a.time * 1000): a for a in assigned}
        hand = [(g, pred[round(t * 1000)]) for (t, surf, _v, g) in hits
                if surf == "snare" and round(t * 1000) in pred]
        correct = sum(1 for g, a in hand if a.limb == g)
        flagged = sum(1 for g, a in hand if a.flagged_ambiguous)
        acc = correct / len(hand) if hand else float("nan")
        print(f"{label:4} {acc:9.3f} {flagged:8d}   ({correct}/{len(hand)} hand hits)")


if __name__ == "__main__":
    main()
