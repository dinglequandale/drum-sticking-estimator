"""FastAPI server for the Drum Sticking Estimator UI.

Run with:  uvicorn server:app --reload --port 8000
"""

import asyncio
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Drum Sticking Estimator")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "dse_sessions"
UPLOAD_DIR.mkdir(exist_ok=True)

# CPU-bound pipeline work runs in a thread pool so the event loop stays free.
_executor = ThreadPoolExecutor(max_workers=2)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_MIME = {
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".webm": "video/webm",
}

# In-process session store: file_id → session dict.
# Restarting the server clears sessions; acceptable for a test tool.
sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile):
    file_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / file_id
    session_dir.mkdir()

    orig_path = session_dir / file.filename
    content = await file.read()
    orig_path.write_bytes(content)

    # Extract audio in a thread (ffmpeg / librosa load)
    loop = asyncio.get_event_loop()
    try:
        wav_path = await loop.run_in_executor(
            _executor, _extract_audio, str(orig_path), str(session_dir)
        )
    except Exception as e:
        raise HTTPException(500, f"Audio extraction failed: {e}")

    suffix = Path(file.filename).suffix.lower()
    sessions[file_id] = {
        "orig_path": str(orig_path),
        "wav_path": wav_path,
        "filename": file.filename,
        "is_video": suffix in VIDEO_EXTS,
        "cached_result": None,
        "cached_config": None,
    }

    return {
        "file_id": file_id,
        "filename": file.filename,
        "audio_url": f"/api/audio/{file_id}",
        "video_url": f"/api/video/{file_id}" if suffix in VIDEO_EXTS else None,
    }


def _extract_audio(orig_path: str, out_dir: str) -> str:
    from src.stage0_ingest import ingest
    wav_path, _ = ingest(orig_path, separate=False, output_dir=out_dir)
    return wav_path


# ---------------------------------------------------------------------------
# Media serving
# ---------------------------------------------------------------------------

@app.get("/api/audio/{file_id}")
async def get_audio(file_id: str):
    sess = _require_session(file_id)
    return FileResponse(sess["wav_path"], media_type="audio/wav")


@app.get("/api/video/{file_id}")
async def get_video(file_id: str):
    sess = _require_session(file_id)
    if not sess["is_video"]:
        raise HTTPException(404)
    suffix = Path(sess["orig_path"]).suffix.lower()
    return FileResponse(sess["orig_path"], media_type=VIDEO_MIME.get(suffix, "video/mp4"))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    file_id: str
    start: float
    end: float
    handedness: str = "right"
    style: str = "crossed"
    ambiguity_threshold: float = 0.65


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    sess = _require_session(req.file_id)

    config_key = (req.handedness, req.style, req.ambiguity_threshold)
    cached = sess["cached_result"]

    if cached is None or sess["cached_config"] != config_key:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor, _run_pipeline,
                sess["wav_path"], req.handedness, req.style, req.ambiguity_threshold
            )
        except Exception as e:
            raise HTTPException(500, f"Pipeline error: {e}")
        sess["cached_result"] = result
        sess["cached_config"] = config_key
    else:
        result = cached

    hits_in_region = [
        {
            "time": round(h.time, 4),
            "limb": h.limb,
            "confidence": round(h.confidence, 3),
            "limb_probs": {k: round(v, 3) for k, v in h.limb_probs.items()},
            "source": h.source,
            "flagged": h.flagged_ambiguous,
        }
        for h in result.assigned_hits
        if req.start <= h.time <= req.end
    ]

    tc = result.beat_info.get("tempo_curve", [])
    tempo = round(tc[0][1]) if tc else 0

    return {
        "hits": hits_in_region,
        "patterns": result.patterns[:6],
        "meta": {
            "tempo_bpm": tempo,
            "meter": result.beat_info.get("meter", "4/4"),
            "total_in_clip": len(result.assigned_hits),
            "in_region": len(hits_in_region),
            "flagged": sum(1 for h in hits_in_region if h["flagged"]),
        },
    }


# ADTOF model is expensive to load (TensorFlow init); load once and reuse across
# requests. Loaded lazily on the first analyze so the server still starts fast.
_adtof_model = None
_adtof_lock = threading.Lock()


def _get_adtof_model():
    global _adtof_model
    if _adtof_model is None:
        with _adtof_lock:
            if _adtof_model is None:
                from src.stage1_onsets import load_adtof_model
                _adtof_model = load_adtof_model()
    return _adtof_model


def _run_pipeline(wav_path: str, handedness: str, style: str, ambiguity_threshold: float):
    from src.pipeline import run
    from src.types import PipelineConfig
    config = PipelineConfig(
        handedness=handedness,
        style=style,
        ambiguity_threshold=ambiguity_threshold,
    )
    return run(wav_path, config=config, separate=False, adtof_model=_get_adtof_model())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_session(file_id: str) -> dict:
    if file_id not in sessions:
        raise HTTPException(404, "Session not found — please re-upload the file")
    return sessions[file_id]


# Static files served last so /api/* routes take priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
