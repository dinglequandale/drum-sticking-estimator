"""Audio-anchored vision probe — does MediaPipe Hands give a usable which-hand
signal at drum onsets, and does its confidence track correctness?

This is a *probe*, not the production vision prong. It validates (or kills) the
vision leg of the fusion architecture on a single curated clip before we invest
further. See memory: project-fusion-architecture.

Pipeline:
  1. Extract audio (bundled ffmpeg) -> librosa onset times (fast; no ADTOF).
  2. For each onset, grab the matching video frame.
  3. MediaPipe HandLandmarker (Tasks API) -> per-hand landmarks + handedness.
  4. Heuristic striking-hand vote (the lower hand is the one coming down) with a
     confidence proxy = MediaPipe score blended with the vertical-separation margin.
  5. Dump a per-onset table + a handful of landmark overlay frames for eyeballing.

Needs hand_landmarker.task in cwd (downloaded from Google's mediapipe-models).
Run:  ./.venv/Scripts/python.exe vision_probe.py samples/sample_1.mp4
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUT_DIR = Path("probe_out")
MODEL = "hand_landmarker.task"
N_OVERLAYS = 10  # how many onset frames to render for visual inspection

WRIST, MIDDLE_TIP = 0, 12


def extract_audio(video_path: Path) -> Path:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(tempfile.mkdtemp(prefix="vprobe_")) / "audio.wav"
    cmd = [ffmpeg, "-y", "-i", str(video_path), "-ac", "1", "-ar", "44100", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[-500:]}")
    return out


def detect_onsets(wav_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(wav_path), sr=44100, mono=True)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True, hop_length=512)


def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=MODEL)
    opts = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2, running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def analyze(frame_bgr, landmarker) -> dict:
    """Which-hand vote + confidence for one frame. Abstains on no detection."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)

    out = {"n_hands": 0, "hands": [], "vote": None, "confidence": 0.0, "basis": "abstain"}
    if not res.hand_landmarks:
        return out, res

    detected = []
    for lms, handed in zip(res.hand_landmarks, res.handedness):
        cat = handed[0]
        detected.append({
            "mp_label": cat.category_name, "mp_score": round(cat.score, 3),
            "wrist_x": round(lms[WRIST].x, 3), "wrist_y": round(lms[WRIST].y, 3),
            "tip_y": round(lms[MIDDLE_TIP].y, 3),
        })

    out["n_hands"] = len(detected)
    out["hands"] = detected

    detected.sort(key=lambda d: d["tip_y"], reverse=True)   # lower (greater y) first = striker
    striker = detected[0]
    out["vote"] = "L" if striker["wrist_x"] < 0.5 else "R"   # left half of frame = camera-left

    if len(detected) == 1:
        out["confidence"] = round(0.4 * striker["mp_score"], 3)
        out["basis"] = "single-hand"
    else:
        margin = detected[0]["tip_y"] - detected[1]["tip_y"]
        out["confidence"] = round(min(1.0, striker["mp_score"] * (0.5 + 4.0 * margin)), 3)
        out["basis"] = "two-hand-margin"
    return out, res


def draw_overlay(frame, res, info, t):
    h, w = frame.shape[:2]
    for lms, handed in zip(res.hand_landmarks, res.handedness):
        for lm in lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 200, 255), -1)
        wx, wy = int(lms[WRIST].x * w), int(lms[WRIST].y * h)
        cv2.circle(frame, (wx, wy), 6, (255, 0, 0), 2)
        cv2.putText(frame, handed[0].category_name[0], (wx + 8, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"t={t:.2f} vote={info['vote']} conf={info['confidence']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main(video_path: str):
    video_path = Path(video_path)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"[1/4] extracting audio from {video_path.name} ...")
    wav = extract_audio(video_path)

    print("[2/4] detecting onsets ...")
    onsets = detect_onsets(wav)
    print(f"      {len(onsets)} onsets, {onsets[0]:.2f}s .. {onsets[-1]:.2f}s")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[3/4] video: {fps:.2f} fps, {n_frames} frames")

    landmarker = make_landmarker()
    overlay_idxs = set(np.linspace(0, len(onsets) - 1, N_OVERLAYS).astype(int))

    print(f"[4/4] {'t(s)':>7} {'nH':>3} {'vote':>4} {'conf':>5}  basis  hands(label:score x,tipY)")
    results = []
    for i, t in enumerate(onsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(round(t * fps)), n_frames - 1))
        ok, frame = cap.read()
        if not ok:
            continue
        info, res = analyze(frame, landmarker)
        results.append((t, info))
        hsumm = " | ".join(f"{d['mp_label'][0]}:{d['mp_score']} x{d['wrist_x']} ty{d['tip_y']}"
                           for d in info["hands"])
        print(f"      {t:7.2f} {info['n_hands']:3d} {str(info['vote']):>4} "
              f"{info['confidence']:5.2f}  {info['basis']:16} {hsumm}")
        if i in overlay_idxs and res.hand_landmarks:
            draw_overlay(frame, res, info, t)
            cv2.imwrite(str(OUT_DIR / f"onset_{i:03d}_{t:.2f}s.png"), frame)

    cap.release()

    n = len(results)
    n_detect = sum(1 for _, r in results if r["n_hands"] > 0)
    n_two = sum(1 for _, r in results if r["n_hands"] == 2)
    confs = [r["confidence"] for _, r in results if r["vote"]]
    print("\n=== SUMMARY ===")
    print(f"onsets analyzed:        {n}")
    print(f"hands detected (>=1):   {n_detect} ({100*n_detect/max(n,1):.0f}%)")
    print(f"both hands detected:    {n_two} ({100*n_two/max(n,1):.0f}%)")
    if confs:
        print(f"confidence mean/median: {np.mean(confs):.2f} / {np.median(confs):.2f}")
    print(f"overlay frames: {OUT_DIR}/  ({len(list(OUT_DIR.glob('*.png')))} images)")
    print("\nNOTE: MediaPipe Left/Right uses a mirror convention; the L/R *vote* here is\n"
          "position-based. Both need orientation calibration against GT. This probe\n"
          "answers detection reliability + signal presence, not final accuracy.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/sample_1.mp4")
