"""AutoClipper FastAPI server with an async job queue.

Endpoints:
    POST /api/process        -> submit a video (YouTube URL or local path); returns job_id
    GET  /api/status/{id}    -> poll job status, logs, and result
    GET  /api/jobs           -> list recent jobs
    GET  /api/files/{id}/{f} -> download a generated clip / asset

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import contextvars
import json
import logging
import os
import sys
from typing import Optional
import shutil
import threading
import time
import uuid
import urllib.error

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from autoclipper import config, pipeline
from autoclipper.trending import get_trending_ideas
from autoclipper.trending_youtube import fetch_trending, enrich_trending, CATEGORY_MAP
from autoclipper.subtitles import build_ass, burn_ass, cues_to_srt
from autoclipper.hooks import add_hook_to_video
from autoclipper import voiceover as voiceover_mod
from autoclipper.youtube_uploader import (
    get_auth_url,
    exchange_code,
    is_authenticated,
    is_user_authenticated,
    upload_video,
    has_client_secret,
)
from autoclipper import db as user_db
from autoclipper import auth as auth_mod
from autoclipper.db import (
    User,
    UserCreate,
    UserLogin,
    UserSettingsUpdate,
    UserPublic,
    settings_for,
    apply_settings,
    get_user_by_username,
    create_user,
    save_youtube_token,
    load_youtube_token,
)
from autoclipper.auth import (
    get_current_user,
    hash_password,
    verify_password,
    create_access_token,
    public_user,
)
from autoclipper.utils import (
    ensure_ffmpeg, logger as ac_logger, probe_video_info, sanitize_filename, write_json,
)

app = FastAPI(title="AutoClipper API", version="0.1.0")

# Enable CORS for cross-domain frontend deployments (e.g. Railway separate services)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional frontend static build directory resolution (single-container Railway deployment)
_FRONTEND_DIST = None
for _cand in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist"),
    "/app/frontend/dist",
]:
    if os.path.isdir(_cand) and os.path.isfile(os.path.join(_cand, "index.html")):
        _FRONTEND_DIST = _cand
        logger.info("Found frontend build at %s", _FRONTEND_DIST)
        break

if _FRONTEND_DIST:
    _assets = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="frontend_assets")


# --- Output cleanup -------------------------------------------------------
# Job folders under OUTPUT_ROOT accumulate fast (each job keeps source video,
# every generated clip, vertical re-encodes, subtitles, hooks...). A background
# task prunes folders older than JOB_RETENTION_HOURS so disk doesn't fill up.
# Folders that were saved into the Library are preserved.
JOB_RETENTION_HOURS = float(os.environ.get("JOB_RETENTION_HOURS", "24"))
_CLEANUP_INTERVAL = 60 * 30  # seconds between sweeps


def _is_library_folder(name: str) -> bool:
    """Library folders are keyed by sanitized video title, not a 12-hex job id."""
    manifest = os.path.join(config.OUTPUT_ROOT, name, f"{name}_library.json")
    return os.path.isfile(manifest)


def _sweep_old_jobs() -> int:
    """Delete job folders (not library folders) older than the retention window."""
    if not os.path.isdir(config.OUTPUT_ROOT):
        return 0
    # Never touch these shared/persistent folders ("analysis" has its own sweep).
    protected = {"library", "uploads", "models", "analysis"}
    cutoff = time.time() - JOB_RETENTION_HOURS * 3600
    removed = 0
    for name in os.listdir(config.OUTPUT_ROOT):
        if name in protected:
            continue
        path = os.path.join(config.OUTPUT_ROOT, name)
        if not os.path.isdir(path):
            continue
        if _is_library_folder(name):
            continue  # keep saved library folders
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


async def _cleanup_loop():
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        try:
            n = await asyncio.to_thread(_sweep_old_jobs)
            if n:
                logger.info("Cleanup: removed %d old job folder(s) (>%.0fh).", n, JOB_RETENTION_HOURS)
            m = await asyncio.to_thread(_sweep_old_analyses)
            if m:
                logger.info(
                    "Cleanup: removed %d old analysis folder(s) (>%.0fh).",
                    m, ANALYSIS_RETENTION_HOURS,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Cleanup sweep failed: %s", e)


@app.on_event("startup")
async def _startup_cleanup():
    user_db.init_db()
    asyncio.create_task(_cleanup_loop())


@app.post("/api/cleanup")
async def manual_cleanup(request: Request, hours: float = None):
    """Force a cleanup sweep. Optional `hours` override (query or JSON body)."""
    global JOB_RETENTION_HOURS
    override = hours
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("hours") is not None:
            override = float(body["hours"])
    except Exception:
        pass
    saved = JOB_RETENTION_HOURS
    if override is not None:
        JOB_RETENTION_HOURS = max(0.0, override)
    try:
        removed = await asyncio.to_thread(_sweep_old_jobs)
    finally:
        JOB_RETENTION_HOURS = saved
    return {"ok": True, "removed": removed, "retention_hours": override or saved}


# --- Per-job state --------------------------------------------------------
_current_job_id = contextvars.ContextVar("current_job_id", default=None)
jobs: dict[str, dict] = {}
_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_JOBS)


class JobLogHandler(logging.Handler):
    """Route autoclipper log records to the active job's log buffer."""

    def emit(self, record: logging.LogRecord):
        jid = _current_job_id.get()
        job = jobs.get(jid) if jid else None
        if job is None:
            return
        job["logs"].append(self.format(record))


ac_logger.addHandler(JobLogHandler())
ac_logger.setLevel(logging.INFO)


# --- Request models -------------------------------------------------------
class SubtitleRequest(BaseModel):
    name: Optional[str] = None        # library folder name
    job_id: Optional[str] = None      # or a processing job id (one of name/job_id required)
    filename: str
    cues: list  # [{"start": float, "end": float, "text": str}]
    style: dict = {}


# --- Job runner -----------------------------------------------------------
def _run_pipeline_sync(job_id: str, source: str, output_dir: str, opts: dict, on_clip=None, on_progress=None):
    _current_job_id.set(job_id)
    return pipeline.run(
        source, output_dir=output_dir, on_clip=on_clip, on_progress=on_progress, **opts
    )


async def _run_job(job_id: str, source: str, opts: dict):
    job = jobs[job_id]
    output_dir = os.path.join(config.OUTPUT_ROOT, job_id)
    # Pre-populate result so incremental clips can be streamed via polling.
    job["result"] = {
        "title": None,
        "source": source,
        "job_id": job_id,
        "video_file": None,
        "language": None,
        "duration": None,
        "clips": [],
        "elapsed_seconds": 0,
        "progress": job.get("progress"),
    }
    lock = threading.Lock()

    def on_clip_ready(clip, index, total):
        with lock:
            job["result"]["clips"].append({
                **clip,
                "index": index,
                "render_status": "rendered",
                "qa_status": clip.get("qa_status") or "pending",
            })
            job["result"]["title"] = job["result"].get("title")  # will be set later

    def on_progress_update(event):
        with lock:
            job["progress"] = dict(event)
            if job.get("result") is not None:
                job["result"]["progress"] = dict(event)

    try:
        async with _semaphore:
            job["status"] = "processing"
            result = await asyncio.to_thread(
                _run_pipeline_sync,
                job_id,
                source,
                output_dir,
                opts,
                on_clip=on_clip_ready,
                on_progress=on_progress_update,
            )
        job["result"] = result
        job["status"] = "completed"
        job["progress"] = {
            "stage": "completed",
            "message": "Clip generation, QA, and metadata saving are complete.",
            "updated_at": time.time(),
            "qa_status": (result.get("quality_assurance") or {}).get("status", "unknown"),
        }
    except Exception as e:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(e)
        job["progress"] = {
            "stage": "failed",
            "message": "Processing stopped. Review the job error and logs.",
            "updated_at": time.time(),
        }
    finally:
        job["finished_at"] = time.time()


# --- Routes ---------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint for Railway and container orchestrators."""
    return {"status": "ok", "name": "AutoClipper API", "version": "0.1.0", "jobs": len(jobs)}


@app.get("/")
async def root():
    if _FRONTEND_DIST:
        index_file = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
    return {"name": "AutoClipper API", "version": "0.1.0", "jobs": len(jobs)}


# --- Auth (register / login / per-user settings) ------------------------
@app.post("/api/auth/register", response_model=UserPublic)
async def register(req: UserCreate):
    """Create a new account. Username must be unique."""
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    with user_db.Session(user_db.engine) as session:
        if get_user_by_username(session, req.username):
            raise HTTPException(status_code=409, detail="Username already taken.")
        user = create_user(session, req, hash_password(req.password))
    return public_user(user)


@app.post("/api/auth/login")
async def login(req: UserLogin):
    """Authenticate and return a JWT access token."""
    with user_db.Session(user_db.engine) as session:
        user = get_user_by_username(session, req.username)
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserPublic)
async def me(current: User = Depends(get_current_user)):
    """Return the authenticated user's public profile."""
    return public_user(current)


@app.get("/api/auth/settings", response_model=None)
async def get_settings(current: User = Depends(get_current_user)):
    """Return the user's stored settings (keys decrypted for display).

    `undecryptable_fields` lists credentials saved under a previous encryption
    key that must be entered again; it is never populated with secret values.
    """
    from autoclipper.db import undecryptable_fields

    payload = settings_for(current).model_dump()
    payload["undecryptable_fields"] = undecryptable_fields(current)
    return payload


@app.put("/api/auth/settings", response_model=None)
async def update_settings(
    req: UserSettingsUpdate,
    current: User = Depends(get_current_user),
):
    """Update the user's stored settings (keys re-encrypted at rest)."""
    from autoclipper.db import undecryptable_fields

    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        apply_settings(db_user, req)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        result = settings_for(db_user).model_dump()
        result["undecryptable_fields"] = undecryptable_fields(db_user)
    return result


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a raw file and return its server-side filename (in uploads/)."""
    uploads = os.path.join(config.OUTPUT_ROOT, "uploads")
    os.makedirs(uploads, exist_ok=True)
    original = file.filename or "upload.mp4"
    ext = os.path.splitext(original)[1] or ".mp4"
    name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(os.path.splitext(original)[0])}{ext}"
    path = os.path.join(uploads, name)
    with open(path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return {"filename": name, "path": path}


@app.post("/api/process")
async def process(
    source: str = Form(""),
    file: UploadFile = File(None),
    whisper_model: str = Form(None),
    gemini_model: str = Form(None),
    vertical: bool = Form(True),
    use_yolo: bool = Form(True),
    subtitles: bool = Form(True),
    split_screen: bool = Form(False),
    force_hd: bool = Form(False),
    youtube_cookies: str = Form(None),
    groq_key: str = Form(None),
    gemini_key: str = Form(None),
    clip_count: int = Form(None),
    min_clip: float = Form(None),
    max_clip: float = Form(None),
    target_duration: float = Form(None),
    custom_prompt: str = Form(None),
    selected_clip: str = Form(None),
    selected_clips: str = Form(None),
    analysis_id: str = Form(None),
    aspect_ratio: str = Form(None),
    add_hook: bool = Form(False),
    montage: bool = Form(False),
    voiceover_text: str = Form(None),
    voiceover_engine: str = Form("auto"),
    voiceover_voice: str = Form(None),
    background_music_file: UploadFile = File(None),
    music_volume: float = Form(0.18),
    content_type: str = Form("general"),
    current: User = Depends(get_current_user),
):
    try:
        clip_count = pipeline._normalize_clip_count(clip_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_ffmpeg()

    # Per-user keys take precedence; fall back to the server's .env keys.
    from autoclipper.db import decrypt_value

    if not groq_key:
        groq_key = decrypt_value(current.groq_key) or config.GROQ_API_KEY
    if not gemini_key:
        gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    if not whisper_model and current.whisper_model:
        whisper_model = current.whisper_model
    if not gemini_model and current.gemini_model:
        gemini_model = current.gemini_model
    if not youtube_cookies and current.youtube_cookies:
        youtube_cookies = decrypt_value(current.youtube_cookies)

    # Single-step upload: stream the posted file straight into the pipeline.
    if file is not None:
        uploads = os.path.join(config.OUTPUT_ROOT, "uploads")
        os.makedirs(uploads, exist_ok=True)
        original = file.filename or "upload.mp4"
        ext = os.path.splitext(original)[1] or ".mp4"
        name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(os.path.splitext(original)[0])}{ext}"
        path = os.path.join(uploads, name)
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        source = path

    cached_transcript_path = None
    if analysis_id:
        analysis_entry = analyses.get(analysis_id)
        if analysis_entry is None:
            raise HTTPException(status_code=404, detail="Unknown analysis id.")
        if analysis_entry.get("status") != "done":
            raise HTTPException(status_code=409, detail="Video analysis is not complete yet.")
        analysis_result = analysis_entry.get("result") or {}
        analyzed_source = analysis_result.get("source")
        transcript_path = analysis_entry.get("transcript_path")
        analysis_root = os.path.abspath(os.path.join(config.OUTPUT_ROOT, "analysis"))
        if not analyzed_source or not transcript_path or not os.path.isfile(transcript_path):
            raise HTTPException(status_code=410, detail="The analysis cache expired; analyze the video again.")
        if os.path.commonpath([analysis_root, os.path.abspath(transcript_path)]) != analysis_root:
            raise HTTPException(status_code=400, detail="Invalid cached transcript path.")
        source = analyzed_source
        cached_transcript_path = transcript_path

    if not source or not source.strip():
        raise HTTPException(status_code=400, detail="Provide a 'source' URL/path or upload a 'file'.")

    try:
        selected_clips_data = json.loads(selected_clips) if selected_clips else None
        if selected_clips_data is not None and not isinstance(selected_clips_data, list):
            raise ValueError("selected_clips must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid selected_clips: {exc}")

    if not 0.0 <= music_volume <= 1.0:
        raise HTTPException(status_code=400, detail="music_volume must be between 0 and 1.")

    job_id = uuid.uuid4().hex[:12]
    music_path = None
    if background_music_file is not None:
        allowed_audio_exts = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
        ext = os.path.splitext(background_music_file.filename or "")[1].lower()
        if ext not in allowed_audio_exts:
            raise HTTPException(status_code=400, detail="Background music must be MP3, WAV, M4A, AAC, OGG, or FLAC.")
        job_dir = os.path.join(config.OUTPUT_ROOT, job_id)
        os.makedirs(job_dir, exist_ok=True)
        music_path = os.path.join(job_dir, "background_music" + ext)
        with open(music_path, "wb") as out:
            while chunk := await background_music_file.read(1024 * 1024):
                out.write(chunk)
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "source": source,
        "created_at": time.time(),
        "finished_at": None,
        "logs": [],
        "result": None,
        "error": None,
        "progress": {
            "stage": "queued",
            "message": "Waiting for a processing slot.",
            "updated_at": time.time(),
        },
    }
    opts = {
        "whisper_model": whisper_model,
        "gemini_model": gemini_model,
        "vertical": vertical,
        "use_yolo": use_yolo,
        "subtitles": subtitles,
        "split_screen": split_screen,
        "force_hd": force_hd,
        "cookies_text": youtube_cookies,
        "groq_key": groq_key,
        "gemini_key": gemini_key,
        "clip_count": clip_count,
        "min_clip": min_clip,
        "max_clip": max_clip,
        "target_duration": target_duration,
        "custom_prompt": custom_prompt,
        "selected_clip": json.loads(selected_clip) if selected_clip else None,
        "selected_clips": selected_clips_data,
        "cached_transcript_path": cached_transcript_path,
        "aspect_ratio": aspect_ratio or None,
        "add_hook": bool(add_hook),
        "montage": bool(montage),
        "voiceover_text": voiceover_text,
        "voiceover_engine": voiceover_engine or "auto",
        "voiceover_voice": voiceover_voice or None,
        "background_music_path": music_path,
        "music_volume": max(0.0, min(float(music_volume), 1.0)),
        "content_type": content_type or "general",
    }
    if (
        opts["min_clip"] is not None
        and opts["max_clip"] is not None
        and opts["min_clip"] > opts["max_clip"]
    ):
        opts["min_clip"], opts["max_clip"] = opts["max_clip"], opts["min_clip"]
    asyncio.create_task(_run_job(job_id, source, opts))
    return {"job_id": job_id, "status": "queued"}


# --- Analysis-first workflow ------------------------------------------------
# Heavy analysis shares the render semaphore and admission is capped before
# uploads are accepted, preventing repeated requests from exhausting resources.
ANALYSIS_RETENTION_HOURS = float(os.environ.get("ANALYSIS_RETENTION_HOURS", "24"))
MAX_PENDING_ANALYSES = max(1, int(os.environ.get("MAX_PENDING_ANALYSES", "4")))
analyses: dict = {}
pending_analysis_ids: set[str] = set()


def _sweep_old_analyses() -> int:
    """Delete analysis folders older than the retention window."""
    root = os.path.join(config.OUTPUT_ROOT, "analysis")
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - ANALYSIS_RETENTION_HOURS * 3600
    removed = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


@app.post("/api/analyze-video")
async def analyze_video(
    source: str = Form(""),
    file: UploadFile = File(None),
    whisper_model: str = Form(None),
    gemini_model: str = Form(None),
    force_hd: bool = Form(False),
    youtube_cookies: str = Form(None),
    groq_key: str = Form(None),
    gemini_key: str = Form(None),
    content_type: str = Form(None),
    angle_count: int = Form(4),
    current: User = Depends(get_current_user),
):
    """Analyze a video (download/upload -> transcribe -> understand) without cutting."""
    ensure_ffmpeg()

    from autoclipper.db import decrypt_value

    if not groq_key:
        groq_key = decrypt_value(current.groq_key) or config.GROQ_API_KEY
    if not gemini_key:
        gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    if not whisper_model and current.whisper_model:
        whisper_model = current.whisper_model
    if not gemini_model and current.gemini_model:
        gemini_model = current.gemini_model
    if not youtube_cookies and current.youtube_cookies:
        youtube_cookies = decrypt_value(current.youtube_cookies)

    if file is None and not source.strip():
        raise HTTPException(status_code=400, detail="Provide a 'source' URL/path or upload a 'file'.")

    # Reserve capacity before reading a potentially large upload. There is no
    # await between checking and reserving, so simultaneous requests cannot all
    # pass the admission limit.
    if len(pending_analysis_ids) >= MAX_PENDING_ANALYSES:
        raise HTTPException(
            status_code=429,
            detail=f"Analysis queue is full ({MAX_PENDING_ANALYSES} pending/active). Try again shortly.",
            headers={"Retry-After": "30"},
        )
    analysis_id = uuid.uuid4().hex[:12]
    pending_analysis_ids.add(analysis_id)
    try:
        if file is not None:
            uploads = os.path.join(config.OUTPUT_ROOT, "uploads")
            os.makedirs(uploads, exist_ok=True)
            original = file.filename or "upload.mp4"
            ext = os.path.splitext(original)[1] or ".mp4"
            name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(os.path.splitext(original)[0])}{ext}"
            path = os.path.join(uploads, name)
            with open(path, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    out.write(chunk)
            source = path

        if not source or not source.strip():
            raise HTTPException(status_code=400, detail="Provide a 'source' URL/path or upload a 'file'.")

        analyses[analysis_id] = {
            "analysis_id": analysis_id,
            "status": "queued",
            "source": source,
            "created_at": time.time(),
            "finished_at": None,
            "logs": [],
            "result": None,
            "error": None,
        }
    except Exception:
        pending_analysis_ids.discard(analysis_id)
        raise
    opts = {
        "whisper_model": whisper_model,
        "gemini_model": gemini_model,
        "force_hd": force_hd,
        "cookies_text": youtube_cookies,
        "groq_key": groq_key,
        "gemini_key": gemini_key,
        # An omitted type means the analysis model should detect it from the video.
        "content_type": content_type or None,
        "angle_count": angle_count or 4,
    }
    # Share the render/analysis concurrency budget and run work off the loop.
    asyncio.create_task(_run_analysis_bounded(analysis_id, source, opts))
    return {"analysis_id": analysis_id, "status": "queued"}


@app.get("/api/analyze-video/{analysis_id}")
async def analyze_video_status(analysis_id: str):
    """Poll an in-flight video analysis."""
    entry = analyses.get(analysis_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown analysis id.")
    return entry


async def _run_analysis_bounded(analysis_id: str, source: str, opts: dict):
    """Run expensive analysis under the same concurrency budget as rendering."""
    try:
        async with _semaphore:
            await asyncio.to_thread(_run_analysis, analysis_id, source, opts)
    finally:
        # Release admission even if the task is cancelled before the worker starts.
        pending_analysis_ids.discard(analysis_id)


def _run_analysis(analysis_id: str, source: str, opts: dict):
    """Worker: ingest -> probe -> transcribe -> whole-video understanding."""
    from autoclipper import ingest as ingest_mod
    from autoclipper.analyze import analyze_video_summary_and_angles
    from autoclipper.transcribe import transcribe
    from autoclipper.utils import probe_video_info

    entry = analyses[analysis_id]

    def log(msg: str, *args):
        try:
            text = msg % args if args else msg
        except (TypeError, ValueError):
            text = msg
        entry["logs"].append(text)
        logger.info("[analysis %s] %s", analysis_id, text)

    out_dir = os.path.join(config.OUTPUT_ROOT, "analysis", analysis_id)
    try:
        os.makedirs(out_dir, exist_ok=True)
        entry["status"] = "downloading"
        log("Fetching video…")
        local_source, title = ingest_mod.ingest(
            source, out_dir, force_hd=opts.get("force_hd", False),
            cookies_text=opts.get("cookies_text"),
            max_height=None if opts.get("force_hd", False) else 720,
        )

        entry["status"] = "probing"
        log("Probing video metadata…")
        video_info = probe_video_info(local_source)

        entry["status"] = "transcribing"
        log("Transcribing the full audio with Groq Whisper…")
        transcript = transcribe(
            local_source, out_dir,
            api_key=opts.get("groq_key"), model=opts.get("whisper_model"),
        )
        transcript_path = os.path.join(out_dir, "transcript.json")
        write_json(transcript_path, transcript)
        entry["transcript_path"] = transcript_path

        entry["status"] = "sampling_visuals"
        log("Sampling a bounded set of timestamped visual frames…")
        from autoclipper.visual_context import sample_video_frames

        visual_sampling = sample_video_frames(local_source, video_info.get("duration"))
        visual_samples = visual_sampling.get("frames") or []
        if visual_samples:
            log("Sampled %d visual frame(s) for multimodal context.", len(visual_samples))
        if visual_sampling.get("warning"):
            log(visual_sampling["warning"])

        entry["status"] = "analyzing"
        log("Reading the whole video with Gemini…")
        report = analyze_video_summary_and_angles(
            transcript,
            api_key=opts.get("gemini_key"),
            model=opts.get("gemini_model"),
            video_info=video_info,
            angle_count=opts.get("angle_count", 4),
            content_type=opts.get("content_type"),
            visual_samples=visual_samples,
            visual_sampling_warning=visual_sampling.get("warning"),
        )

        report.update(
            {
                "analysis_id": analysis_id,
                "title": title,
                "source": local_source,
                "language": transcript.get("language"),
                "video_info": video_info,
                "word_count": len(transcript.get("words") or []),
                "segment_count": len(transcript.get("segments") or []),
            }
        )
        entry["status"] = "done"
        entry["finished_at"] = time.time()
        entry["result"] = report
        log(
            "Analysis complete: %d chapter(s), %d angle(s).",
            len(report.get("chapters") or []),
            len(report.get("suggested_angles") or []),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Video analysis failed")
        entry["status"] = "error"
        entry["error"] = str(e)
        entry["finished_at"] = time.time()
        log(f"Error: {e}")
    finally:
        pending_analysis_ids.discard(analysis_id)


class GamingPrepareRequest(BaseModel):
    source: str = ""          # path/URL already on server, or empty if file uploaded
    cam_box: dict            # normalized {x,y,w,h} 0..1
    game_box: dict           # normalized {x,y,w,h} 0..1
    layout: str = "cam_top"  # cam_top | game_top | side
    file: str = None         # optional server-side uploaded filename (in uploads/)


@app.post("/api/gaming/prepare")
async def gaming_prepare(req: GamingPrepareRequest):
    """Compose a 9:16 gaming video from two regions of a single source.

    Runs as an async job (returns a `job_id` immediately); poll /api/status/{job_id}.
    When finished, `result.source` is the gaming_layout.mp4 path and
    `result.preview_url` can be used as the `source` for /api/process.
    """
    from autoclipper import gaming_layout
    from autoclipper import ingest as ingest_mod

    source = req.source
    if not source and req.file:
        source = os.path.join(config.OUTPUT_ROOT, "uploads", req.file)

    if not source:
        raise HTTPException(status_code=400, detail="Provide a valid 'source' URL or uploaded 'file'.")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "source": source,
        "created_at": time.time(),
        "finished_at": None,
        "logs": ["Queued gaming layout job…"],
        "result": None,
        "error": None,
    }
    asyncio.create_task(asyncio.to_thread(_run_gaming_job, job_id, source, req))
    return {"job_id": job_id, "status": "queued"}


def _run_gaming_job(job_id: str, source: str, req: GamingPrepareRequest):
    from autoclipper import gaming_layout
    from autoclipper import ingest as ingest_mod

    job = jobs[job_id]
    try:
        job["status"] = "downloading"
        # If the source is a YouTube (or other remote) URL, download it first.
        if not os.path.isfile(source) and ingest_mod._is_url(source):
            dl_dir = os.path.join(config.OUTPUT_ROOT, job_id)
            os.makedirs(dl_dir, exist_ok=True)
            job["logs"].append("Downloading source from URL…")
            source, _ = ingest_mod.ingest(
                source, dl_dir, force_hd=False, cookies_text=config.YOUTUBE_COOKIES, max_height=720
            )

        if not os.path.isfile(source):
            raise RuntimeError("Source file not found after download.")

        out_dir = os.path.join(config.OUTPUT_ROOT, job_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gaming_layout.mp4")

        job["status"] = "processing"
        job["logs"].append("Building 9:16 gaming layout…")
        gaming_layout.build_gaming_video(
            source, req.cam_box, req.game_box,
            layout=req.layout, out_path=out_path,
        )

        job["status"] = "done"
        job["finished_at"] = time.time()
        job["result"] = {
            "source": out_path,
            "layout": req.layout,
            "preview_url": f"/api/files/{job_id}/gaming_layout.mp4",
        }
        job["logs"].append("Gaming layout ready.")
    except Exception as e:  # noqa: BLE001
        logger.exception("Gaming layout failed")
        job["status"] = "error"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        job["logs"].append(f"Error: {e}")


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "source": job["source"],
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
        "logs": job["logs"],
        "result": job["result"],
        "error": job["error"],
        "progress": job.get("progress"),
    }


@app.get("/api/jobs")
async def list_jobs():
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "source": j["source"],
            "created_at": j["created_at"],
            "clips": [c.get("file") for c in (j["result"] or {}).get("clips", [])],
        }
        for j in sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)
    ]


@app.get("/api/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    path = os.path.join(config.OUTPUT_ROOT, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/api/trending")
async def trending(niche: str = "general", count: int = 10,
                   gemini_model: str = None,
                   current: User = Depends(get_current_user)):
    """Return AI-generated trending short-video ideas for a niche."""
    from autoclipper.db import decrypt_value

    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    if not gemini_model and current.gemini_model:
        gemini_model = current.gemini_model
    try:
        ideas = get_trending_ideas(
            niche=niche, api_key=gemini_key, model=gemini_model, count=count
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return {"niche": niche, "count": len(ideas), "ideas": ideas}


@app.get("/api/trending/youtube")
async def trending_youtube(
    region: str = "ID",
    category: str = "general",
    max_results: int = 12,
    window_days: int = 3,
    gemini_model: str = None,
    enrich: bool = True,
    current: User = Depends(get_current_user),
):
    """Return Explore-style most-popular YouTube videos for a region/category.

    YouTube retired the Trending page in July 2025 in favor of Explore
    (per-category destination pages). This mirrors that by pulling
    videos.list chart=mostPopular for the region and filtering by the
    chosen Explore category (Music, Gaming, News, ...). Shorts are
    filtered out; podcast mode keeps only longer-form videos (>10 min).
    `category` is one of the EXPLORE_CATEGORIES ids.
    """
    from autoclipper.db import decrypt_value

    # Provider keys are resolved only from encrypted account settings or server
    # configuration; do not accept API secrets in a URL query string.
    account_youtube_key = decrypt_value(current.youtube_api_key)
    youtube_key = account_youtube_key or config.YOUTUBE_API_KEY
    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    if not gemini_model and current.gemini_model:
        gemini_model = current.gemini_model
    if not youtube_key:
        if current.youtube_api_key:
            detail = (
                "A YouTube Data API key is stored for this account but cannot be decrypted; "
                "the backend encryption key may have changed. Re-save it in Settings, and keep "
                "AUTOCLIPPER_SECRET stable across production backend restarts."
            )
        else:
            detail = (
                "No YouTube Data API key is saved for this account. Click Save in Settings, "
                "or configure YOUTUBE_API_KEY on the server."
            )
        raise HTTPException(status_code=503, detail=detail)

    try:
        videos = fetch_trending(
            api_key=youtube_key, region=region, category=category,
            max_results=max(1, min(max_results or 12, 25)),
            window_days=max(1, min(window_days or 3, 7)),
        )
        # Fallback: if a niche filter yields nothing (YouTube's per-region
        # trending often lacks that category), return the general list and
        # flag it so the UI can show a hint.
        category_filtered = True
        if not videos and category != "general":
            videos = fetch_trending(
                api_key=youtube_key, region=region, category="general",
                max_results=max(1, min(max_results or 12, 25)),
                window_days=max(1, min(window_days or 3, 7)),
            )
            category_filtered = False
    except urllib.error.HTTPError as exc:
        logger.warning("YouTube Data API returned HTTP %s for Explore.", exc.code)
        if exc.code == 403:
            detail = (
                "YouTube Data API rejected the request. Check that the API is enabled, "
                "the key restrictions are correct, and quota is available."
            )
        elif exc.code in {400, 401}:
            detail = "YouTube Data API rejected the request. Check the key and Explore region settings."
        else:
            detail = "YouTube Data API is temporarily unavailable. Please try again later."
        raise HTTPException(status_code=502, detail=detail) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Could not reach YouTube Data API for Explore (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Could not reach YouTube Data API. Check your connection and try again.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Do not echo provider exceptions: they can contain request URLs or
        # implementation details that should not be returned to the browser.
        logger.warning("YouTube Explore request failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="YouTube Explore could not be loaded. Check the API key and server logs, then try again.",
        ) from exc

    insights = {}
    if enrich:
        try:
            insights = enrich_trending(
                videos, niche=category, api_key=gemini_key, model=gemini_model
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Trending enrichment skipped: %s", e)
    for v in videos:
        v["idea"] = insights.get(v["video_id"], "")
    return {"region": region, "category": category,
            "category_label": CATEGORY_MAP.get(category, ("0", category))[1],
            "category_filtered": category_filtered,
            "count": len(videos), "videos": videos}


def trending_youtube_category(category: str) -> str:
    from autoclipper.trending_youtube import CATEGORY_MAP

    return CATEGORY_MAP.get(category, ("0", "Explore"))[0]


@app.post("/api/subtitle")
async def subtitle(req: SubtitleRequest):
    if req.name:
        output_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    elif req.job_id:
        output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    else:
        raise HTTPException(status_code=400, detail="name or job_id required")
    video = os.path.join(output_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")

    base = os.path.splitext(req.filename)[0]
    ass_path = os.path.join(output_dir, f"{base}_edit.ass")
    out_name = f"{base}_edited.mp4"
    out_path = os.path.join(output_dir, out_name)

    video_info = probe_video_info(video)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(
            build_ass(
                req.cues,
                req.style,
                video_width=video_info.get("width") or 1080,
                video_height=video_info.get("height") or 1920,
            )
        )

    # Keep an SRT sidecar that matches the EDITED captions, named after the
    # edited video so job/library name-derived lookups stay in sync. Without
    # this, saving an edited clip would carry the stale pre-edit SRT and the
    # library could re-burn outdated text over the edited video.
    edited_srt_name = f"{base}_edited.srt"
    srt_text = cues_to_srt(req.cues)

    try:
        burn_ass(video, ass_path, out_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Burn failed: {e}")

    if srt_text:
        try:
            with open(os.path.join(output_dir, edited_srt_name), "w", encoding="utf-8") as f:
                f.write(srt_text)
        except OSError:
            edited_srt_name = None
    else:
        edited_srt_name = None

    # Save to library manifest if name is provided (same pattern as /api/enhance)
    if req.name:
        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            from autoclipper.utils import write_json
            import json as _json
            with open(manifest_path, "r", encoding="utf-8") as _mf:
                manifest = _json.load(_mf)
            orig_entry = None
            for c in manifest.get("clips", []):
                if c.get("file") == req.filename:
                    orig_entry = c
                    break
            if orig_entry is None:
                base_name = os.path.splitext(req.filename)[0]
                idx = int("".join(filter(str.isdigit, base_name)) or "0")
                for c in manifest.get("clips", []):
                    if c.get("index") == idx:
                        orig_entry = c
                        break
            if orig_entry is not None:
                edited_entry = dict(orig_entry)
                edited_entry["file"] = out_name
                edited_entry["subtitled"] = True
                # Track the edited sidecar (or drop the stale pre-edit one).
                if edited_srt_name:
                    edited_entry["srt"] = edited_srt_name
                else:
                    edited_entry.pop("srt", None)
                found = False
                for i, c in enumerate(manifest.get("clips", [])):
                    if c.get("file") == out_name:
                        manifest["clips"][i] = edited_entry
                        found = True
                        break
                if not found:
                    manifest.setdefault("clips", []).append(edited_entry)
                write_json(manifest_path, manifest)

    return {"filename": out_name, "srt": edited_srt_name}


class HookRequest(BaseModel):
    job_id: str
    filename: str
    text: str
    position: str = "top"  # top | center | bottom
    font_scale: float = 1.0
    size: str = "M"  # S | M | L
    entrance: str = "fade"  # fade | none
    hold_seconds: float = 5.0


@app.post("/api/hook")
async def hook(req: HookRequest):
    """Burn a viral hook text overlay onto a generated clip (OpenShorts-style)."""
    output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    video = os.path.join(output_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Hook text is required")

    base = os.path.splitext(req.filename)[0]
    out_name = f"hook_{base}.mp4"
    out_path = os.path.join(output_dir, out_name)

    try:
        size_scale = {"S": 0.7, "L": 1.4}.get((req.size or "M").upper(), 1.0)
        font_scale = req.font_scale * size_scale
        duration = req.hold_seconds if req.hold_seconds and req.hold_seconds > 0 else None
        add_hook_to_video(
            video, req.text, out_path,
            position=req.position, font_scale=font_scale,
            duration=duration, style="classic",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hook failed: {e}")

    return {"filename": out_name}


class HookPreviewRequest(BaseModel):
    job_id: str
    filename: str
    text: str
    position: str = "top"
    font_scale: float = 1.0
    size: str = "M"


@app.post("/api/hook/preview")
async def hook_preview(req: HookPreviewRequest):
    """Render a single preview frame (PNG) with the hook overlay, for the UI."""
    from autoclipper import hooks as _hooks

    output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    video = os.path.join(output_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Hook text is required")

    size_scale = {"S": 0.7, "L": 1.4}.get((req.size or "M").upper(), 1.0)
    font_scale = req.font_scale * size_scale

    preview_name = f"hook_preview_{os.path.splitext(req.filename)[0]}.png"
    preview_path = os.path.join(output_dir, preview_name)
    try:
        _hooks.create_hook_image(
            req.text, int(1080 * 0.9), preview_path, font_scale=font_scale
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")

    return FileResponse(preview_path, media_type="image/png",
                        filename=preview_name)


# --- Voice-over (Kokoro TTS locally + Gemini TTS fallback) --------------
class VoiceOverRequest(BaseModel):
    job_id: Optional[str] = None     # for generated clips
    name: Optional[str] = None       # for library clips
    filename: str
    text: str
    engine: str = "auto"             # auto | kokoro | gemini
    voice: str = None
    mode: str = "overlay"            # overlay (mix) | replace (dub)


@app.post("/api/voiceover")
async def voiceover(req: VoiceOverRequest, current: User = Depends(get_current_user)):
    """Generate a voice-over for a clip and burn it into a new mp4.

    engine='auto' picks Kokoro for English-ish text and Gemini for Indonesian
    (Gemini needs the user's stored GEMINI_API_KEY).
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Voice-over text is required")
    if req.mode not in ("overlay", "replace"):
        raise HTTPException(status_code=400, detail="mode must be 'overlay' or 'replace'")

    # Resolve the source video.
    if req.name:
        output_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    else:
        if not req.job_id:
            raise HTTPException(status_code=400, detail="job_id or name required")
        output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    video = os.path.join(output_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")

    base = os.path.splitext(req.filename)[0]
    wav_name = f"{base}_vo.wav"
    wav_path = os.path.join(output_dir, wav_name)
    out_name = f"{base}_vo.mp4"
    out_path = os.path.join(output_dir, out_name)

    # Gemini key (for fallback / Indonesian) comes from the authenticated user.
    from autoclipper.db import decrypt_value

    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY

    try:
        voiceover_mod.generate_voiceover(
            req.text, wav_path, engine=req.engine, voice=req.voice, gemini_key=gemini_key
        )
        voiceover_mod.mix_voiceover(video, wav_path, out_path, mode=req.mode)
    except Exception as e:  # noqa: BLE001
        detail = f"Voice-over failed: {e}"
        logger.warning(detail)
        raise HTTPException(status_code=500, detail=detail)

    return {"filename": out_name, "wav": wav_name}


@app.post("/api/voiceover/preview")
async def voiceover_preview(req: VoiceOverRequest, current: User = Depends(get_current_user)):
    """Generate only the voice-over WAV (no video mix) for a quick listen."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Voice-over text is required")

    if req.name:
        output_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    else:
        if not req.job_id:
            raise HTTPException(status_code=400, detail="job_id or name required")
        output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    if not os.path.isfile(os.path.join(output_dir, req.filename)):
        raise HTTPException(status_code=404, detail="Clip not found")

    preview_name = f"vo_preview_{os.path.splitext(req.filename)[0]}.wav"
    preview_path = os.path.join(output_dir, preview_name)
    from autoclipper.db import decrypt_value

    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    try:
        voiceover_mod.generate_voiceover(
            req.text, preview_path, engine=req.engine, voice=req.voice, gemini_key=gemini_key
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")
    return FileResponse(preview_path, media_type="audio/wav", filename=preview_name)


# --- Enhance (Hook + Voice-Over in one pass) ------------------------------

class EnhanceRequest(BaseModel):
    name: Optional[str] = None        # library folder name
    job_id: Optional[str] = None     # or a processing job id
    filename: str                     # clip file inside the folder/job
    # Hook overlay (optional)
    hook_text: str = ""
    hook_position: str = "top"       # top | center | bottom
    hook_size: str = "M"             # S | M | L (maps to font_scale)
    hook_style: str = "classic"      # classic | dark | yellow | red | outline | outline_yellow
    hook_hold: float = 5.0           # seconds the hook stays (0/None = whole clip)
    # Voice-over (optional)
    vo_text: str = ""
    vo_engine: str = "kokoro"        # kokoro | gemini | auto
    vo_voice: str = ""
    vo_mode: str = "overlay"         # overlay | replace
    vo_extend: bool = True           # if hook+vo: prepend intro (hook+vo) then play clip


@app.post("/api/enhance")
async def enhance(req: EnhanceRequest, current: User = Depends(get_current_user)):
    """Burn a viral hook AND/OR a voice-over onto a clip in a single pass.

    Either `hook_text` or `vo_text` (or both) may be provided. The hook is
    burned first, then the voice-over (if any) is mixed into the result.
    """
    if not req.hook_text.strip() and not req.vo_text.strip():
        raise HTTPException(status_code=400, detail="Provide hook_text or vo_text (or both).")

    if req.name:
        output_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    else:
        if not req.job_id:
            raise HTTPException(status_code=400, detail="name or job_id required")
        output_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    video = os.path.join(output_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")

    from autoclipper import hooks as _hooks
    from autoclipper.db import decrypt_value

    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    base = os.path.splitext(req.filename)[0]
    step_path = video

    orig_entry = None
    try:
        both = bool(req.hook_text.strip()) and bool(req.vo_text.strip())
        created_paths = []

        if both and req.vo_extend:
            font_scale = {"S": 0.8, "M": 1.0, "L": 1.25}.get(req.hook_size, 1.0)
            vo_wav = os.path.join(output_dir, f"vo_{base}.wav")
            voiceover_mod.generate_voiceover(
                req.vo_text, vo_wav, engine=req.vo_engine,
                voice=req.vo_voice or None, gemini_key=gemini_key,
            )
            created_paths.append(vo_wav)
            vo_dur = voiceover_mod.wav_duration(vo_wav)
            intro_dur = max(req.hook_hold if req.hook_hold and req.hook_hold > 0 else 0.0, vo_dur)
            intro_path = os.path.join(output_dir, f"intro_{base}.mp4")
            voiceover_mod.build_intro(
                video, vo_wav, intro_path,
                hook_text=req.hook_text, hook_position=req.hook_position,
                font_scale=font_scale, style=req.hook_style, duration=intro_dur,
            )
            created_paths.append(intro_path)
            final_out = os.path.join(output_dir, f"enhanced_{base}.mp4")
            voiceover_mod.concat_videos(intro_path, video, final_out)
            step_path = final_out
        else:
            hook_out = None
            vo_wav = None
            final_out = None

            # 1) Hook overlay (if requested)
            if req.hook_text.strip():
                hook_out = os.path.join(output_dir, f"hook_{base}.mp4")
                font_scale = {"S": 0.8, "M": 1.0, "L": 1.25}.get(req.hook_size, 1.0)
                duration = req.hook_hold if req.hook_hold and req.hook_hold > 0 else None
                _hooks.add_hook_to_video(
                    step_path, req.hook_text, hook_out,
                    position=req.hook_position, font_scale=font_scale,
                    duration=duration, style=req.hook_style,
                )
                step_path = hook_out

            # 2) Voice-over (if requested)
            if req.vo_text.strip():
                vo_wav = os.path.join(output_dir, f"vo_{base}.wav")
                voiceover_mod.generate_voiceover(
                    req.vo_text, vo_wav, engine=req.vo_engine,
                    voice=req.vo_voice or None, gemini_key=gemini_key,
                )
                created_paths.append(vo_wav)
                final_out = os.path.join(output_dir, f"enhanced_{base}.mp4")
                voiceover_mod.mix_voiceover(step_path, vo_wav, final_out, mode=req.vo_mode)
                if hook_out is not None:
                    created_paths.append(hook_out)  # hook was intermediate
                step_path = final_out

        # Remove all intermediate files (vo_wav, intro, hook if not final)
        for p in created_paths:
            if os.path.isfile(p) and p != step_path:
                os.remove(p)

        # Update library manifest so the enhanced clip appears with its metadata
        if req.name:
            manifest_path = os.path.join(output_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                from autoclipper.utils import write_json
                import json as _json
                with open(manifest_path, "r", encoding="utf-8") as _mf:
                    manifest = _json.load(_mf)
                # Find the matching original clip entry
                orig_entry = None
                for c in manifest.get("clips", []):
                    if c.get("file") == req.filename:
                        orig_entry = c
                        break
                if orig_entry is None:
                    base = os.path.splitext(req.filename)[0]
                    idx = int("".join(filter(str.isdigit, base)) or "0")
                    for c in manifest.get("clips", []):
                        if c.get("index") == idx:
                            orig_entry = c
                            break
                if orig_entry is not None:
                    # Build enhanced entry from original metadata
                    enhanced_entry = dict(orig_entry)
                    enhanced_entry["file"] = os.path.basename(step_path)
                    enhanced_entry["enhanced"] = True
                    # Ensure index is set
                    if "index" not in enhanced_entry:
                        enhanced_entry["index"] = len(manifest.get("clips", [])) + 1
                    # Update or append
                    found = False
                    for i, c in enumerate(manifest.get("clips", [])):
                        if c.get("file") == enhanced_entry["file"]:
                            manifest["clips"][i] = enhanced_entry
                            found = True
                            break
                    if not found:
                        manifest.setdefault("clips", []).append(enhanced_entry)
                    write_json(manifest_path, manifest)
        # Return metadata alongside the filename
        result_meta = {"filename": os.path.basename(step_path)}
        if req.name and orig_entry is not None:
            result_meta.update({
                "title": orig_entry.get("title", ""),
                "description": orig_entry.get("description", ""),
                "hook": req.hook_text or orig_entry.get("hook", ""),
            })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Enhance failed: {e}")

    return result_meta


# --- Chat-split (two-person interview/podcast) --------------------------------

class ChatSplitDetectRequest(BaseModel):
    name: str
    filename: str
    num_frames: int = 5


class ChatSplitRenderRequest(BaseModel):
    name: str
    filename: str
    person1: dict  # {x, y, w, h} normalized
    person2: dict


@app.post("/api/chat-split/detect")
async def chat_split_detect(req: ChatSplitDetectRequest):
    """Auto-detect two-person regions in a clip."""
    video = os.path.join(config.OUTPUT_ROOT, "library", req.name, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")
    from autoclipper.gaming_layout import detect_chat_regions
    try:
        p1, p2 = detect_chat_regions(video, req.num_frames)
        return {"person1": p1, "person2": p2}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat-split/render")
async def chat_split_render(req: ChatSplitRenderRequest):
    """Render a chat-split video from a library clip."""
    video = os.path.join(config.OUTPUT_ROOT, "library", req.name, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")
    from autoclipper.gaming_layout import build_chat_split
    base = os.path.splitext(req.filename)[0]
    out_name = f"chatsplit_{base}.mp4"
    out_path = os.path.join(config.OUTPUT_ROOT, "library", req.name, out_name)
    try:
        build_chat_split(video, req.person1, req.person2, out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat-split failed: {e}")

    # Update manifest
    manifest_path = os.path.join(config.OUTPUT_ROOT, "library", req.name, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = {}
        manifest.setdefault("clips", []).append({
            "file": out_name,
            "index": len(manifest.get("clips", [])) + 1,
            "title": f"Chat Split - {req.filename}",
            "chatsplit": True,
        })
        from autoclipper.utils import write_json
        write_json(manifest_path, manifest)

    return {"filename": out_name}


PREVIEW_CACHE_DIR = os.path.join(config.OUTPUT_ROOT, "cache", "previews")


@app.post("/api/enhance/preview")
async def enhance_preview(req: EnhanceRequest, current: User = Depends(get_current_user)):
    """Render a short video preview of the hook overlay (no voice-over).

    Returns an MP4 of the first few seconds with the hook burned on, so the
    user can see the overlay before committing to a full burn.  Preview files
    are stored in a dedicated cache directory (not the library folder).
    """
    if not req.hook_text.strip():
        raise HTTPException(status_code=400, detail="hook_text is required for preview")

    if req.name:
        clip_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    else:
        if not req.job_id:
            raise HTTPException(status_code=400, detail="name or job_id required")
        clip_dir = os.path.join(config.OUTPUT_ROOT, req.job_id)
    video = os.path.join(clip_dir, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")

    from autoclipper import hooks as _hooks

    os.makedirs(PREVIEW_CACHE_DIR, exist_ok=True)
    base = os.path.splitext(req.filename)[0]
    import hashlib
    cache_key = f"{req.hook_text}|{req.hook_position}|{req.hook_size}|{req.hook_style}|{req.hook_hold}"
    hsh = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
    preview_name = f"enhance_prev_{base}_{hsh}.mp4"
    preview_path = os.path.join(PREVIEW_CACHE_DIR, preview_name)
    try:
        font_scale = {"S": 0.8, "M": 1.0, "L": 1.25}.get(req.hook_size, 1.0)
        duration = req.hook_hold if req.hook_hold and req.hook_hold > 0 else None
        _hooks.add_hook_to_video(
            video, req.hook_text, preview_path,
            position=req.hook_position, font_scale=font_scale,
            duration=duration, style=req.hook_style,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")

    return {"filename": preview_name}


@app.get("/api/cache/{filename}")
async def serve_cache(filename: str):
    """Serve a cached preview file."""
    path = os.path.join(PREVIEW_CACHE_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Cache file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


class EnhanceDraftRequest(BaseModel):
    name: str
    filename: str
    prompt_hint: str = ""   # optional extra context


@app.post("/api/enhance/draft")
async def enhance_draft(req: EnhanceDraftRequest, current: User = Depends(get_current_user)):
    """Use Gemini to draft an OpenShorts-style hook from the clip's metadata."""
    output_dir = os.path.join(config.OUTPUT_ROOT, "library", req.name)
    manifest_path = os.path.join(output_dir, "manifest.json")
    hook = ""
    description = ""
    title = ""
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            for c in m.get("clips", []):
                if c.get("file") == req.filename:
                    hook = c.get("hook", "")
                    description = c.get("description", "")
                    title = c.get("title", "")
                    break
        except (OSError, json.JSONDecodeError):
            pass

    gemini_key = decrypt_value(current.gemini_key) or config.GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=400, detail="Set a Gemini API key in Settings to draft a hook.")
    prompt = (
        "You write viral short-video hooks in the style of OpenShorts: short, punchy, "
        "curiosity-driving, max 8 words, no hashtags, no quotes. "
        f"Video title: {title}\nVideo description: {description}\n"
        f"Existing hook idea: {hook}\nUser note: {req.prompt_hint}\n"
        "Return ONLY the hook text."
    )
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        resp = config.gemini_generate(
            client, config.GEMINI_MODEL, prompt
        )
        draft = (resp.text or "").strip().strip('"').strip()
    except Exception as e:  # noqa: BLE001
        # Fall back to the existing hook if the LLM call fails.
        if hook:
            draft = hook
        else:
            raise HTTPException(status_code=500, detail=f"Hook draft failed: {e}")

    return {"hook": draft}


# --- Saved Library -------------------------------------------------------
LIBRARY_ROOT = os.path.join(config.OUTPUT_ROOT, "library")


@app.get("/api/library")
async def list_library():
    if not os.path.isdir(LIBRARY_ROOT):
        return []
    items = []
    for name in sorted(os.listdir(LIBRARY_ROOT)):
        folder = os.path.join(LIBRARY_ROOT, name)
        if not os.path.isdir(folder):
            continue
        manifest_path = os.path.join(folder, "manifest.json")
        manifest = {}
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                manifest = {}
        clips = []
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".mp4"):
                clips.append(fn)
        if not clips:
            continue
        items.append(
            {
                "name": name,
                "title": manifest.get("title", name),
                "clips": [
                    {
                        "file": fn,
                        "index": int("".join(filter(str.isdigit, fn)) or "0"),
                        "title": _clip_meta(manifest, fn, i),
                        "description": _clip_desc(manifest, fn, i),
                        "hook": _clip_hook(manifest, fn, i),
                    }
                    for i, fn in enumerate(clips)
                ],
            }
        )
    return items


@app.get("/api/library/{name}/{filename}")
async def library_file(name: str, filename: str):
    path = os.path.join(LIBRARY_ROOT, name, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.delete("/api/library/{name}/{filename}")
async def delete_library_clip(name: str, filename: str):
    """Delete a single clip (and its SRT) from a library folder."""
    folder = os.path.join(LIBRARY_ROOT, name)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail="Library folder not found")
    removed = []
    import re
    srt_name = re.sub(r"\.mp4$", ".srt", filename, flags=re.IGNORECASE)
    for fn in (filename, srt_name):
        p = os.path.join(folder, fn)
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(fn)
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    # Drop the clip from the manifest.
    manifest_path = os.path.join(folder, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["clips"] = [
                c for c in manifest.get("clips", []) if c.get("file") != filename
            ]
            write_json(manifest_path, manifest)
        except (OSError, json.JSONDecodeError):
            pass
    # If no clips remain, remove the whole folder.
    remaining = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not remaining:
        shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True, "removed": removed}


@app.delete("/api/library/{name}")
async def delete_library_folder(name: str):
    """Delete an entire library folder (all clips for one source video)."""
    folder = os.path.join(LIBRARY_ROOT, name)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail="Library folder not found")
    shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True, "removed": name}


# --- YouTube Shorts upload (free, YouTube Data API v3) -------------------
class YouTubeAuthRequest(BaseModel):
    redirect_uri: str = None


class YouTubeUploadRequest(BaseModel):
    job_id: Optional[str] = None    # for generated clips
    name: Optional[str] = None      # for library clips
    filename: str
    title: str = ""
    description: str = ""
    publish_at: Optional[str] = None  # ISO 8601; if set -> scheduled (private until then)
    thumbnail: Optional[str] = None


@app.get("/api/youtube/status")
async def youtube_status(current: User = Depends(get_current_user)):
    token_json = load_youtube_token(current)
    return {
        "configured": has_client_secret(),
        "authenticated": is_user_authenticated(token_json),
    }


@app.get("/api/youtube/account")
async def youtube_account(current: User = Depends(get_current_user)):
    """Return connected YouTube channel info (name, avatar, stats)."""
    token_json = load_youtube_token(current)
    if not token_json or not is_user_authenticated(token_json):
        raise HTTPException(status_code=401, detail="Not authenticated with YouTube")
    try:
        from autoclipper.youtube_uploader import creds_from_json, _refresh_if_needed, API_SERVICE_NAME, API_VERSION
        from googleapiclient.discovery import build as _build

        creds = creds_from_json(token_json)
        creds, changed = _refresh_if_needed(creds)
        if changed:
            from autoclipper.db import save_youtube_token as _save
            with user_db.Session(user_db.engine) as session:
                db_user = session.get(User, current.id)
                if db_user is not None:
                    _save(db_user, creds.to_json())
                    session.add(db_user)
                    session.commit()

        youtube = _build(API_SERVICE_NAME, API_VERSION, credentials=creds)
        resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            mine=True,
        ).execute()
        items = resp.get("items", [])
        if not items:
            raise HTTPException(status_code=404, detail="No YouTube channel found")
        ch = items[0]
        snippet = ch.get("snippet", {})
        stats = ch.get("statistics", {})
        return {
            "id": ch.get("id"),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
            "country": snippet.get("country", ""),
            "subscriber_count": stats.get("subscriberCount", "0"),
            "video_count": stats.get("videoCount", "0"),
            "view_count": stats.get("viewCount", "0"),
        }
    except HTTPException:
        raise
    except Exception as e:
        ac_logger.error("Failed to fetch YouTube account info: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch YouTube account: {e}")


@app.post("/api/youtube/logout")
async def youtube_logout(current: User = Depends(get_current_user)):
    """Disconnect YouTube by clearing the stored OAuth token."""
    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is not None:
            db_user.youtube_token = None
            session.add(db_user)
            session.commit()
    return {"ok": True, "authenticated": False}


@app.post("/api/youtube/auth_url")
async def youtube_auth_url(
    req: YouTubeAuthRequest, current: User = Depends(get_current_user)
):
    if not has_client_secret():
        raise HTTPException(
            status_code=400,
            detail="YouTube client_secret.json not configured on the server.",
        )
    url = get_auth_url(req.redirect_uri)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to build auth URL")
    return {"auth_url": url}


@app.post("/api/youtube/callback")
async def youtube_callback(
    code: str, redirect_uri: str = None, current: User = Depends(get_current_user)
):
    ok = exchange_code(code, redirect_uri)
    if not ok:
        raise HTTPException(status_code=400, detail="Token exchange failed")
    # exchange_code persists the token to the global TOKEN_PATH; copy it onto
    # the authenticated user (encrypted) and remove the shared global copy.
    from autoclipper.youtube_uploader import TOKEN_PATH

    if os.path.isfile(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                token_json = f.read()
            with user_db.Session(user_db.engine) as session:
                db_user = session.get(User, current.id)
                if db_user is not None:
                    save_youtube_token(db_user, token_json)
                    session.add(db_user)
                    session.commit()
            os.remove(TOKEN_PATH)
        except OSError:
            pass
    return {"ok": True, "authenticated": is_authenticated(load_youtube_token(current))}


@app.post("/api/youtube/upload")
async def youtube_upload(
    req: YouTubeUploadRequest, current: User = Depends(get_current_user)
):
    token_json = load_youtube_token(current)
    if not token_json or not is_authenticated(token_json):
        raise HTTPException(status_code=401, detail="Not authenticated with YouTube")

    # Resolve the source video file.
    if req.name:
        video = os.path.join(config.OUTPUT_ROOT, "library", req.name, req.filename)
    else:
        if not req.job_id:
            raise HTTPException(status_code=400, detail="job_id or name required")
        video = os.path.join(config.OUTPUT_ROOT, req.job_id, req.filename)
    if not os.path.isfile(video):
        raise HTTPException(status_code=404, detail="Clip not found")

    thumb = None
    if req.thumbnail:
        if req.name:
            thumb = os.path.join(config.OUTPUT_ROOT, "library", req.name, req.thumbnail)
        else:
            thumb = os.path.join(config.OUTPUT_ROOT, req.job_id, req.thumbnail)
        if not os.path.isfile(thumb):
            thumb = None

    def _on_token_changed(new_json: str):
        with user_db.Session(user_db.engine) as session:
            db_user = session.get(User, current.id)
            if db_user is not None:
                save_youtube_token(db_user, new_json)
                session.add(db_user)
                session.commit()

    try:
        result = upload_video(
            video,
            title=req.title,
            description=req.description,
            publish_at=req.publish_at,
            thumbnail_path=thumb,
            token_json=token_json,
            on_token_changed=_on_token_changed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        detail = f"YouTube upload failed: {e}"
        # Try to extract a readable YouTube API error.
        content = getattr(e, "content", None)
        if content:
            try:
                err = json.loads(content.decode("utf-8"))
                reasons = [
                    f"{x.get('reason', '')}: {x.get('message', '')}"
                    for x in err.get("error", {}).get("errors", [])
                ]
                if reasons:
                    detail = "; ".join(reasons)
            except (ValueError, AttributeError):
                pass
        else:
            detail = f"{type(e).__name__}: {str(e)[:300]}"
        raise HTTPException(status_code=500, detail=detail)

    video_id = result.get("id")
    return {
        "ok": True,
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}" if video_id else None,
        "scheduled": bool(req.publish_at),
        "publish_at": req.publish_at,
    }


# ── Facebook Pages upload ───────────────────────────────────────────────────

class FacebookAuthRequest(BaseModel):
    redirect_uri: str


class FacebookCallbackRequest(BaseModel):
    code: str
    redirect_uri: str


class FacebookAppSettingsRequest(BaseModel):
    app_id: str
    app_secret: str


class FacebookUploadRequest(BaseModel):
    job_id: Optional[str] = None
    name: Optional[str] = None
    filename: str
    page_id: str
    title: str = ""
    description: str = ""
    as_reel: bool = False


@app.get("/api/facebook/status")
async def facebook_status(current: User = Depends(get_current_user)):
    """Check Facebook connection status."""
    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            return {"authenticated": False, "configured": False}
        fb_data = user_db.load_facebook_token(db_user)
        configured = bool(fb_data and fb_data.get("app_id"))
        authenticated = bool(fb_data and fb_data.get("user_token"))
        return {"authenticated": authenticated, "configured": configured}


@app.get("/api/facebook/pages")
async def facebook_pages(current: User = Depends(get_current_user)):
    """List Facebook Pages the user can manage."""
    from autoclipper.facebook_uploader import get_pages, is_authenticated as fb_auth

    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        fb_data = user_db.load_facebook_token(db_user)
        if not fb_data or not fb_data.get("user_token"):
            raise HTTPException(status_code=401, detail="Facebook not connected")

    user_token = fb_data["user_token"]
    try:
        pages = get_pages(user_token)
        return {"pages": pages}
    except Exception as e:
        ac_logger.error("Failed to fetch Facebook pages: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch pages: {e}")


@app.post("/api/facebook/auth_url")
async def facebook_auth_url(
    req: FacebookAuthRequest, current: User = Depends(get_current_user)
):
    """Generate Facebook Login OAuth URL."""
    from autoclipper.facebook_uploader import get_auth_url

    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        fb_data = user_db.load_facebook_token(db_user)
        if not fb_data or not fb_data.get("app_id"):
            raise HTTPException(
                status_code=400,
                detail="Facebook App ID not configured. Set it in Settings.",
            )

    url = get_auth_url(fb_data["app_id"], req.redirect_uri)
    return {"auth_url": url}


@app.post("/api/facebook/callback")
async def facebook_callback(
    req: FacebookCallbackRequest, current: User = Depends(get_current_user)
):
    """Exchange Facebook OAuth code for tokens."""
    from autoclipper.facebook_uploader import (
        exchange_code,
        exchange_long_lived,
        get_user_info,
    )

    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        fb_data = user_db.load_facebook_token(db_user)
        if not fb_data or not fb_data.get("app_id"):
            raise HTTPException(status_code=400, detail="Facebook App ID not configured")
        app_secret = fb_data.get("app_secret", "")

    try:
        # Exchange code for short-lived token
        token_data = exchange_code(
            fb_data["app_id"], app_secret, req.code, req.redirect_uri
        )
        user_token = token_data.get("access_token")

        # Exchange for long-lived token
        long_lived = exchange_long_lived(user_token, fb_data["app_id"], app_secret)
        long_token = long_lived.get("access_token", user_token)
        expires_in = long_lived.get("expires_in", 60 * 24 * 60 * 60)  # default 60 days

        # Get user info
        user_info = get_user_info(long_token)

        # Save token data
        import time
        token_save = {
            "app_id": fb_data["app_id"],
            "app_secret": app_secret,
            "user_token": long_token,
            "user_name": user_info.get("name", "") if user_info else "",
            "user_id": user_info.get("id", "") if user_info else "",
            "expires_at": time.time() + expires_in,
        }
        user_db.save_facebook_token(db_user, token_save)
        session.add(db_user)
        session.commit()

        return {
            "ok": True,
            "authenticated": True,
            "user_name": token_save["user_name"],
        }
    except Exception as e:
        ac_logger.error("Facebook callback error: %s", e)
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {e}")


@app.post("/api/facebook/upload")
async def facebook_upload(
    req: FacebookUploadRequest, current: User = Depends(get_current_user)
):
    """Upload a video to a Facebook Page."""
    from autoclipper.facebook_uploader import upload_video_to_page

    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        fb_data = user_db.load_facebook_token(db_user)
        if not fb_data or not fb_data.get("user_token"):
            raise HTTPException(status_code=401, detail="Facebook not connected")

    # Resolve video file
    if req.name:
        video_path = os.path.join(config.OUTPUT_ROOT, "library", req.name, req.filename)
    elif req.job_id:
        video_path = os.path.join(config.OUTPUT_ROOT, req.job_id, req.filename)
    else:
        raise HTTPException(status_code=400, detail="Either name or job_id is required")

    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {req.filename}")

    # Get page access token
    from autoclipper.facebook_uploader import get_pages
    pages = get_pages(fb_data["user_token"])
    page_data = next((p for p in pages if p["id"] == req.page_id), None)
    if not page_data:
        raise HTTPException(status_code=404, detail="Page not found or not accessible")

    try:
        result = upload_video_to_page(
            video_path=video_path,
            page_id=req.page_id,
            page_access_token=page_data["access_token"],
            title=req.title,
            description=req.description,
            as_reel=req.as_reel,
            user_token=fb_data["user_token"],
            app_id=fb_data.get("app_id"),
        )
        return {"ok": True, **result}
    except Exception as e:
        ac_logger.error("Facebook upload error: %s", e)
        raise HTTPException(status_code=500, detail=f"Facebook upload failed: {e}")


@app.post("/api/facebook/app_settings")
async def facebook_app_settings(
    req: FacebookAppSettingsRequest, current: User = Depends(get_current_user)
):
    app_id = req.app_id.strip()
    app_secret = req.app_secret.strip()
    """Save Facebook App ID and App Secret."""
    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        fb_data = user_db.load_facebook_token(db_user) or {}
        fb_data["app_id"] = app_id
        fb_data["app_secret"] = app_secret
        user_db.save_facebook_token(db_user, fb_data)
        session.add(db_user)
        session.commit()
    return {"ok": True}


@app.post("/api/facebook/logout")
async def facebook_logout(current: User = Depends(get_current_user)):
    """Disconnect Facebook."""
    with user_db.Session(user_db.engine) as session:
        db_user = session.get(User, current.id)
        if db_user is not None:
            db_user.facebook_token = None
            session.add(db_user)
            session.commit()
    return {"ok": True, "authenticated": False}


class SaveLibraryRequest(BaseModel):
    job_id: str
    filename: str
    srt: str = None
    title: str = None
    clip_title: str = None
    description: str = None
    hook: str = None


@app.post("/api/library/save")
async def save_to_library_endpoint(req: SaveLibraryRequest):
    """Copy a generated clip into the Saved Library (one clip at a time)."""
    src = os.path.join(config.OUTPUT_ROOT, req.job_id, req.filename)
    if not os.path.isfile(src):
        raise HTTPException(status_code=404, detail="Clip not found")

    safe_title = sanitize_filename(req.title or f"video_{req.job_id}")
    lib_root = LIBRARY_ROOT
    base = os.path.join(lib_root, safe_title)
    n = 1
    while os.path.exists(base):
        # If folder exists, reuse it; otherwise avoid collisions.
        base = os.path.join(lib_root, f"{safe_title}_{n}")
        n += 1
    os.makedirs(base, exist_ok=True)

    # Determine the next clip index in this folder.
    existing = [f for f in os.listdir(base) if f.lower().endswith(".mp4")]
    idx = len(existing) + 1
    dest = os.path.join(base, f"clip{idx}.mp4")
    try:
        shutil.copy2(src, dest)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Copy failed: {e}")

    # Copy the accompanying SRT (if present) so subtitles can be re-burned later.
    srt_name = os.path.basename(req.srt) if req.srt else os.path.splitext(req.filename)[0] + ".srt"
    srt_src = os.path.join(config.OUTPUT_ROOT, req.job_id, srt_name)
    if os.path.isfile(srt_src):
        try:
            shutil.copy2(srt_src, os.path.splitext(dest)[0] + ".srt")
        except OSError:
            pass

    # Update / create the manifest.
    manifest_path = os.path.join(base, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.setdefault("title", req.title or safe_title)
    manifest.setdefault("job_id", req.job_id)
    clips = manifest.get("clips", [])
    dest_name = f"clip{idx}.mp4"
    clips.append(
        {
            "index": idx,
            "file": dest_name,
            "title": req.clip_title or req.filename,
            "description": req.description or "",
            "hook": req.hook or "",
        }
    )
    manifest["clips"] = clips
    write_json(manifest_path, manifest)

    rel = os.path.relpath(base, config.OUTPUT_ROOT)
    return {"library": rel, "folder": os.path.basename(base), "file": f"clip{idx}.mp4"}


class RegisterBurnRequest(BaseModel):
    name: str
    source_file: str
    result_file: str


@app.post("/api/library/register")
async def register_burn(req: RegisterBurnRequest):
    """Record a burned (subtitled) clip into the library folder's manifest."""
    base = os.path.join(LIBRARY_ROOT, req.name)
    if not os.path.isdir(base):
        raise HTTPException(status_code=404, detail="Library folder not found")
    if not os.path.isfile(os.path.join(base, req.result_file)):
        raise HTTPException(status_code=404, detail="Burned file not found")

    manifest_path = os.path.join(base, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            manifest = {}

    base = os.path.splitext(req.source_file)[0]
    src_idx = int("".join(filter(str.isdigit, base)) or "0")
    for c in manifest.get("clips", []):
        if c.get("index") == src_idx:
            c["subtitled"] = req.result_file
            c["subtitled_title"] = req.result_file
            break
    else:
        clips = manifest.setdefault("clips", [])
        clips.append(
            {
                "index": src_idx,
                "title": req.source_file,
                "subtitled": req.result_file,
            }
        )
    write_json(manifest_path, manifest)
    return {"ok": True, "file": req.result_file}


def _clip_field(manifest: dict, filename: str, key: str, position: int = 0) -> str:
    for c in manifest.get("clips", []):
        if c.get("file") == filename:
            return c.get(key, "")
    # Fallback: match by extracted index (for older manifests).
    base = os.path.splitext(filename)[0]
    idx = int("".join(filter(str.isdigit, base)) or "0")
    for c in manifest.get("clips", []):
        if c.get("index") == idx:
            return c.get(key, "")
    # Last resort: positional match (older manifests with mismatched indices).
    clips = manifest.get("clips", [])
    if 0 <= position < len(clips):
        return clips[position].get(key, "")
    return ""


def _clip_meta(manifest: dict, filename: str, position: int = 0) -> str:
    return _clip_field(manifest, filename, "title", position)


def _clip_desc(manifest: dict, filename: str, position: int = 0) -> str:
    return _clip_field(manifest, filename, "description", position)


def _clip_hook(manifest: dict, filename: str, position: int = 0) -> str:
    return _clip_field(manifest, filename, "hook", position)


# --- SPA Fallback (when frontend is bundled with backend) ------------------
if _FRONTEND_DIST:
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_route(full_path: str):
        # Do not intercept API requests
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
        target = os.path.join(_FRONTEND_DIST, full_path)
        if os.path.isfile(target):
            return FileResponse(target)
        index_file = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not found")

