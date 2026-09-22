#!/usr/bin/env python3
"""Local web UI for meeting recordings, notes, and timestamped playback."""

from __future__ import annotations

import os
import threading
import uuid
import mimetypes
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

import pipeline

load_dotenv(pipeline.ROOT / ".env")

app = FastAPI(title="Meeting notes")
STATIC_DIR = pipeline.ROOT / "static"
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    id: str | None = None
    force: bool = False
    reanalyze: bool = False


def public_meeting(meeting: dict, job: dict | None = None) -> dict:
    payload = {key: value for key, value in meeting.items() if key != "mtime"}
    payload["when"] = meeting.get("mtime")
    payload["job"] = job
    return payload


def job_for_stem(stem: str) -> dict | None:
    with jobs_lock:
        matches = [job for job in jobs.values() if job.get("stem") == stem]
    if not matches:
        return None
    matches.sort(key=lambda item: item.get("updated", 0), reverse=True)
    return sanitize_job(matches[0])


def sanitize_job(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "stem": job.get("stem"),
        "status": job.get("status"),
        "step": job.get("step"),
        "message": job.get("message"),
        "progress": job.get("progress"),
        "error": job.get("error"),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/recordings")
def api_recordings() -> dict:
    return {
        "recordings": [
            public_meeting(meeting, job_for_stem(meeting["id"]))
            for meeting in pipeline.list_meetings()
        ]
    }


@app.get("/api/recordings/{stem}")
def api_recording(stem: str) -> dict:
    stem = unquote(stem)
    meeting = next((item for item in pipeline.list_meetings() if item["id"] == stem), None)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    analysis = pipeline.load_analysis(stem)
    return {
        "recording": public_meeting(meeting, job_for_stem(stem)),
        "analysis": analysis,
    }


@app.get("/api/recordings/{stem}/media")
def api_media(stem: str) -> FileResponse:
    stem = unquote(stem)
    media = pipeline.media_for_stem(stem)
    if media is None:
        raise HTTPException(status_code=404, detail="No video/audio file for this recording")
    media_type = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
    return FileResponse(media, filename=media.name, media_type=media_type)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    pipeline.ensure_dirs()
    name = Path(file.filename or "recording").name
    suffix = Path(name).suffix.lower()
    if suffix not in pipeline.AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    dest = pipeline.RECORDINGS_DIR / name
    with dest.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    meeting = pipeline.meeting_status(dest.stem)
    return {"recording": public_meeting(meeting)}


@app.post("/api/recordings/{stem}/meet-transcript")
async def api_upload_meet_transcript(stem: str, file: UploadFile = File(...)) -> dict:
    pipeline.ensure_dirs()
    stem = unquote(stem)
    if pipeline.media_for_stem(stem) is None and pipeline.load_analysis(stem) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if not suffix:
        if "pdf" in content_type:
            suffix = ".pdf"
        elif content_type.startswith("text/"):
            suffix = ".txt"
        else:
            suffix = ".pdf"
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Attach a Google Meet transcript PDF")
    pipeline.save_meet_transcript(stem, data, suffix)
    meeting = pipeline.meeting_status(stem)
    return {"recording": public_meeting(meeting, job_for_stem(stem))}


@app.delete("/api/recordings/{stem}/meet-transcript")
def api_delete_meet_transcript(stem: str) -> dict:
    stem = unquote(stem)
    path = pipeline.find_meet_transcript(stem)
    if path is not None and path.exists():
        path.unlink()
    meeting = pipeline.meeting_status(stem)
    return {"recording": public_meeting(meeting, job_for_stem(stem))}


@app.post("/api/analyze")
def api_analyze(body: AnalyzeRequest) -> dict:
    stem = (body.id or "").strip()
    if not stem:
        raise HTTPException(status_code=400, detail="Missing recording id")
    return start_analysis_job(stem, body.force, body.reanalyze)


@app.post("/api/recordings/{stem}/analyze")
def api_analyze_path(stem: str, body: AnalyzeRequest | None = None) -> dict:
    stem = unquote(stem)
    body = body or AnalyzeRequest()
    return start_analysis_job(stem, body.force, body.reanalyze)


def start_analysis_job(stem: str, force: bool, reanalyze: bool) -> dict:
    media = pipeline.media_for_stem(stem)
    if media is None:
        raise HTTPException(status_code=404, detail="Drop the recording file before analyzing")
    existing = job_for_stem(stem)
    if existing and existing.get("status") in {"queued", "running", "cancelling"}:
        return {"job": existing}
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "stem": stem,
        "status": "queued",
        "step": "queued",
        "message": "Waiting to start",
        "progress": 0.0,
        "error": None,
        "updated": 0,
        "cancel": threading.Event(),
    }
    with jobs_lock:
        jobs[job_id] = job
    print(f"[analyze] starting {stem} ({job_id})", flush=True)
    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id, media, force, reanalyze),
        daemon=True,
    )
    thread.start()
    return {"job": sanitize_job(job)}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": sanitize_job(job)}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") not in {"queued", "running", "cancelling"}:
            return {"job": sanitize_job(job)}
        cancel = job.get("cancel")
        if isinstance(cancel, threading.Event):
            cancel.set()
        job["status"] = "cancelling"
        job["step"] = "cancelling"
        job["message"] = "Stopping after the current step…"
        job["updated"] = job.get("updated", 0) + 1
        snapshot = sanitize_job(job)
    return {"job": snapshot}


def run_analysis_job(job_id: str, recording: Path, force: bool, reanalyze: bool) -> None:
    def update(status: str, step: str, message: str, progress: float, error: str | None = None) -> None:
        with jobs_lock:
            job = jobs[job_id]
            job.update(
                {
                    "status": status,
                    "step": step,
                    "message": message,
                    "progress": progress,
                    "error": error,
                    "updated": job.get("updated", 0) + 1,
                }
            )

    def cancel_check() -> None:
        with jobs_lock:
            job = jobs.get(job_id) or {}
            event = job.get("cancel")
        if isinstance(event, threading.Event) and event.is_set():
            raise pipeline.AnalysisCancelled("Analysis stopped")

    def on_progress(step: str, message: str, fraction: float) -> None:
        cancel_check()
        print(f"[analyze] {job_id} {step}: {message}", flush=True)
        update("running", step, message, fraction)

    try:
        cancel_check()
        update("running", "starting", f"Processing {recording.name}", 0.02)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in .env")
        client = OpenAI()
        pipeline.process_recording(
            recording,
            client=client,
            force=force,
            reanalyze=reanalyze,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )
        update("done", "done", "Notes ready", 1.0)
    except pipeline.AnalysisCancelled:
        update("cancelled", "cancelled", "Stopped. You can restart analysis.", jobs.get(job_id, {}).get("progress") or 0)
    except Exception as exc:  # noqa: BLE001
        update("error", "error", str(exc), jobs.get(job_id, {}).get("progress") or 0, str(exc))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8787")),
        reload=False,
    )


if __name__ == "__main__":
    main()
