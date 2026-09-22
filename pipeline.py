#!/usr/bin/env python3
"""Transcribe, translate, and analyze meeting recordings."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Callable

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
NOTES_DIR = ROOT / "notes"
TRANSCRIPTS_DIR = ROOT / "transcripts"
ANALYSIS_DIR = ROOT / "analyses"
MEET_DIR = ROOT / "meet_transcripts"
CACHE_DIR = ROOT / ".cache"

AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".mov",
    ".mkv",
    ".aac",
    ".ogg",
    ".flac",
}

MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MAX_CHUNK_SECONDS = 20 * 60
OVERLAP_SECONDS = 2
LONG_TRANSCRIPT_CHARS = 80_000
TRANSLATE_BATCH_CHARS = 12_000
MERGE_BATCH_CHARS = 18_000

ProgressFn = Callable[[str, str, float], None]
CancelFn = Callable[[], None]


class AnalysisCancelled(RuntimeError):
    """Raised when the user stops an in-progress analysis."""

TRANSCRIPTION_PROMPT = (
    "This is a Google Meet recording of a work meeting mixed Urdu and English. "
    "Transcribe the spoken conversation accurately. Preserve names, product terms, "
    "dates, numbers, and decisions."
)

TRANSLATE_INSTRUCTIONS = """
You clean up a timestamped meeting transcript.

The meeting is mixed Urdu and English. The source may be in Latin script, Arabic
script, or Devanagari (speech-to-text often writes Urdu as Hindi/Sanskrit script).
Rewrite EVERY segment in fluent English. Do not leave any Urdu, Hindi, Arabic,
or Devanagari characters in the text. Keep technical terms, product names, and
English phrases. Preserve meaning; do not summarize or omit asides.

Speaker labels:
- Keep the existing speaker label on each segment (Speaker A, Speaker B, ...).
- Do not invent personal names. Never guess who is speaking.

Return ONLY JSON:
{"segments": [{"index": 0, "speaker": "Speaker A", "text": "English..."}]}
Include every input index.
""".strip()

ANALYSIS_INSTRUCTIONS = """
You are an expert meeting analyst. You turn a timestamped transcript into notes
a busy teammate can act on without rewatching the recording.

The meeting is mixed Urdu and English, already translated into English. Use the
audio transcript as the source of truth for what was said. If a Google Meet
transcript is also provided, use it only as extra context for who was present
and how names map; when the two disagree, trust the audio transcript. Do not
invent attendees, dates, decisions, owners, or deadlines.

Voice and attribution:
- Attribute speech using the speaker labels in the audio transcript
  (Speaker A, Speaker B, or a real name ONLY if that label is already in the
  transcript / participant list).
- Never invent personal names for people on the call. If you do not know a
  speaker's name, write Speaker A, Speaker B, Speaker C — never a guessed name.
- Names of people who were discussed (clients, colleagues not in the call) may
  appear only if the audio transcript actually mentions them.
- Write "Speaker A said she merged two sections" if that is the label; write
  "Zainab said she merged two sections" only when Zainab is a labeled speaker.
- The participants array must be the speakers on the call, not people who were
  only mentioned. Without a Meet PDF, use Speaker A, Speaker B, Speaker C.
- Prefer active voice. The grammatical subject should be the speaker label
  whenever possible.
- Action items are real commitments or assigned tasks, not vague ideas.

Citations:
- The transcript lines start with timestamps like [3:42] or [1:02:15].
- Every overview sentence, note bullet, decision, action item, and open question
  MUST include "start": seconds (a number) for the most relevant utterance.
- Use the timestamp from that transcript line. Convert [3:42] to 222, [1:02:15] to 3735.
- If several moments support a point, use the earliest clear one.

Group discussion notes by topic. Capture disagreements, not only consensus.

Return ONLY valid JSON:
{
  "title": "string",
  "date": "string or null",
  "participants": ["string"],
  "overview": [{"text": "string", "start": 0}],
  "discussion_notes": [
    {"topic": "string", "items": [{"text": "string", "start": 0}]}
  ],
  "decisions": [{"text": "string", "start": 0, "actor": "string or null"}],
  "action_items": [
    {"owner": "string or null", "task": "string", "due": "string or null", "start": 0}
  ],
  "open_questions": [{"text": "string", "start": 0}]
}
""".strip()

MERGE_INSTRUCTIONS = """
You combine two transcripts of the same meeting into one labeled transcript.

Source 1 — AUDIO (authoritative wording and timestamps):
- Taken from the recording. This is what was actually said.
- Keep this wording. Do not summarize, translate further, or polish it into
  a new script.
- Speakers may be labeled Speaker A, Speaker B, Speaker C, or already named.

Source 2 — GOOGLE MEET / GEMINI (authoritative speaker names):
- Has real Google-account names and rough timestamps.
- Wording is often wrong. Do not copy Meet wording over the audio wording.
- Tiny lines are often noise mislabeled as a speaker ("H", "See?", "Oops",
  "style.", "What?", "Please.", one- or two-word fragments). Ignore those
  as speakers.

Your job:
- For every audio segment, assign the Meet participant who was actually speaking.
- Align by time and by distinctive content. Meet timestamps are coarse
  (every couple of minutes); do not require an exact timestamp match.
- One Meet person may cover several audio labels if diarization split them.
- One audio label may be several people if diarization mixed them — name
  each line independently when the Meet transcript makes that clear.
- Only use real participant names from the Meet transcript (usually "First Last").
- Do not invent names. Never introduce people who are not in the Meet roster.
- If you cannot name a line confidently, keep Speaker A / Speaker B / etc.

Return ONLY JSON:
{"segments": [{"index": 0, "speaker": "Jane Smith"}]}
Include every input index. Do not return text.
""".strip()


def ensure_dirs() -> None:
    for path in (RECORDINGS_DIR, NOTES_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR, MEET_DIR, CACHE_DIR):
        path.mkdir(exist_ok=True)


def default_models() -> tuple[str, str]:
    return (
        os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize"),
        os.getenv("ANALYSIS_MODEL", "gpt-5.6"),
    )


def list_media_files() -> list[Path]:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return sorted(
        (
            path
            for path in RECORDINGS_DIR.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def media_for_stem(stem: str) -> Path | None:
    for path in list_media_files():
        if path.stem == stem:
            return path
    return None


def meet_transcript_path(stem: str) -> Path:
    found = find_meet_transcript(stem)
    return found if found else MEET_DIR / f"{stem}.pdf"


def find_meet_transcript(stem: str) -> Path | None:
    for suffix in (".pdf", ".txt", ".md"):
        path = MEET_DIR / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def save_meet_transcript(stem: str, data: bytes, suffix: str) -> Path:
    ensure_dirs()
    suffix = suffix.lower() if suffix else ".pdf"
    if suffix not in {".pdf", ".txt", ".md"}:
        suffix = ".pdf"
    for old in (".pdf", ".txt", ".md"):
        existing = MEET_DIR / f"{stem}{old}"
        if existing.exists():
            existing.unlink()
    dest = MEET_DIR / f"{stem}{suffix}"
    dest.write_bytes(data)
    return dest


def load_meet_hint(stem: str) -> dict:
    path = find_meet_transcript(stem)
    if path is None:
        return {"names": [], "turns": []}
    parsed = parse_meet_transcript(extract_meet_text(path))
    turns, names = filter_meet_turns(parsed)
    return {"names": names, "turns": turns}


def _line_quality(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    good = sum(1 for line in lines if len(line.split()) >= 3)
    return good / len(lines)


def reconstruct_fragmented_meet_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text)
    collapsed = re.sub(
        r"\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+",
        r"\n\1\n",
        collapsed,
    )
    collapsed = re.sub(
        r"\s+([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){1,3}:)\s+",
        r"\n\1 ",
        collapsed,
    )
    return collapsed.strip()


def extract_meet_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        text = unicodedata.normalize(
            "NFKC", path.read_text(encoding="utf-8", errors="replace")
        )
        if _line_quality(text) < 0.4:
            text = reconstruct_fragmented_meet_text(text)
        return text.strip()
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        layout = ""
        try:
            layout = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            layout = ""
        plain = page.extract_text() or ""
        pages.append(layout if _line_quality(layout) >= _line_quality(plain) else plain)
    text = unicodedata.normalize("NFKC", "\n".join(pages)).strip()
    if _line_quality(text) < 0.4:
        text = reconstruct_fragmented_meet_text(text)
    if not text.strip():
        raise RuntimeError(
            "Could not read text from that PDF. Export a text-based Google Meet transcript PDF, not a screenshot."
        )
    return text.strip()


NAME_STOPWORDS = {
    "so",
    "um",
    "uh",
    "but",
    "and",
    "or",
    "the",
    "this",
    "that",
    "next",
    "now",
    "how",
    "you",
    "we",
    "what",
    "yeah",
    "yes",
    "maybe",
    "like",
    "right",
    "there",
    "then",
    "okay",
    "ok",
    "ai",
    "llm",
    "august",
    "hi",
    "hello",
    "thanks",
    "thank",
    "please",
    "style",
    "oops",
    "see",
}


def looks_like_person_name(name: str) -> bool:
    text = unicodedata.normalize("NFKC", name or "").strip().strip(":")
    text = re.sub(r"\s+", " ", text)
    parts = text.split()
    if not 2 <= len(parts) <= 4:
        return False
    if not all(re.fullmatch(r"[A-Z][A-Za-z.'’-]*", part) for part in parts):
        return False
    if any(part.lower() in NAME_STOPWORDS for part in parts):
        return False
    return True


def parse_meet_transcript(text: str) -> list[dict]:
    current_start = 0.0
    current_speaker: str | None = None
    turns: list[dict] = []
    skip_prefixes = (
        "transcription ended",
        "this editable transcript",
        "people can also change",
        "meeting transcript",
        "sync - transcript",
    )
    skip_exact = {"transcript", "transcription"}
    speaker_line = re.compile(
        r"^([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){1,3}):\s*(.*)$"
    )
    for raw in unicodedata.normalize("NFKC", text).splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in skip_exact or any(lowered.startswith(prefix) for prefix in skip_prefixes):
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", line):
            current_start = coerce_seconds(line) or 0.0
            continue
        speaker_match = speaker_line.match(line)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            spoken = speaker_match.group(2).strip()
            if not looks_like_person_name(speaker):
                continue
            current_speaker = speaker
            if spoken:
                turns.append({"start": current_start, "speaker": speaker, "text": spoken})
            continue
        if looks_like_person_name(line):
            current_speaker = line
            continue
        if current_speaker:
            turns.append(
                {"start": current_start, "speaker": current_speaker, "text": line}
            )
    return merge_meet_turns(turns)


def merge_meet_turns(turns: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "").strip()
        start = float(turn.get("start") or 0)
        if (
            merged
            and merged[-1]["speaker"] == speaker
            and abs(float(merged[-1]["start"]) - start) < 1
        ):
            merged[-1]["text"] = f"{merged[-1]['text']} {text}".strip()
        else:
            merged.append({"start": start, "speaker": speaker, "text": text})
    return merged


def filter_meet_turns(turns: list[dict]) -> tuple[list[dict], list[str]]:
    by_name: dict[str, list[dict]] = {}
    for turn in turns:
        speaker = str(turn.get("speaker") or "").strip()
        if not looks_like_person_name(speaker):
            continue
        by_name.setdefault(speaker, []).append(turn)
    kept: set[str] = set()
    for name, items in by_name.items():
        texts = [str(item.get("text") or "").strip() for item in items]
        substantial = [text for text in texts if len(text) >= 20]
        total = sum(len(text) for text in texts)
        if substantial or total >= 40 or len(items) >= 3:
            kept.add(name)
    filtered = [
        turn
        for turn in turns
        if turn["speaker"] in kept and len(str(turn.get("text") or "").strip()) >= 8
    ]
    names = sorted(
        kept, key=lambda name: (-sum(len(t["text"]) for t in by_name[name]), name)
    )
    return filtered, names


def is_unprocessed(recording: Path) -> bool:
    analysis_path = ANALYSIS_DIR / f"{recording.stem}.json"
    notes_path = NOTES_DIR / f"{recording.stem}.md"
    for existing in (analysis_path, notes_path):
        if existing.exists() and existing.stat().st_mtime >= recording.stat().st_mtime:
            return False
    return True


def meeting_status(stem: str) -> dict:
    recording = media_for_stem(stem)
    analysis = load_analysis(stem)
    analyzed = analysis is not None
    title = (analysis or {}).get("title") or stem
    return {
        "id": stem,
        "filename": recording.name if recording else f"{stem}",
        "has_media": recording is not None,
        "has_meet_transcript": find_meet_transcript(stem) is not None,
        "analyzed": analyzed,
        "title": title,
        "date": (analysis or {}).get("date"),
        "participants": (analysis or {}).get("participants") or [],
        "mtime": (
            recording.stat().st_mtime
            if recording
            else (ANALYSIS_DIR / f"{stem}.json").stat().st_mtime
            if (ANALYSIS_DIR / f"{stem}.json").exists()
            else (NOTES_DIR / f"{stem}.md").stat().st_mtime
        ),
    }


def list_meetings() -> list[dict]:
    ensure_dirs()
    stems: set[str] = set()
    for path in list_media_files():
        stems.add(path.stem)
    for folder, suffix in ((ANALYSIS_DIR, ".json"), (NOTES_DIR, ".md")):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix == suffix and path.name != ".gitkeep":
                stems.add(path.stem)
    meetings = [meeting_status(stem) for stem in stems]
    meetings.sort(key=lambda item: item["mtime"], reverse=True)
    return meetings


def load_analysis(stem: str) -> dict | None:
    json_path = ANALYSIS_DIR / f"{stem}.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    notes_path = NOTES_DIR / f"{stem}.md"
    if notes_path.exists():
        return parse_legacy_notes(stem)
    return None


def process_recording(
    recording: Path,
    client: OpenAI | None = None,
    ffmpeg: str | None = None,
    transcription_model: str | None = None,
    analysis_model: str | None = None,
    force: bool = False,
    reanalyze: bool = False,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> dict:
    ensure_dirs()
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract audio from recordings.")
    client = client or OpenAI()
    transcription_model = transcription_model or default_models()[0]
    analysis_model = analysis_model or default_models()[1]

    def progress(step: str, message: str, fraction: float) -> None:
        if cancel_check:
            cancel_check()
        if on_progress:
            on_progress(step, message, fraction)

    progress("starting", f"Processing {recording.name}", 0.02)

    segments_path = TRANSCRIPTS_DIR / f"{recording.stem}.json"
    transcript_txt_path = TRANSCRIPTS_DIR / f"{recording.stem}.txt"
    analysis_path = ANALYSIS_DIR / f"{recording.stem}.json"
    notes_path = NOTES_DIR / f"{recording.stem}.md"

    reuse_transcript = (
        not force
        and segments_path.exists()
        and (
            reanalyze
            or segments_path.stat().st_mtime >= recording.stat().st_mtime
        )
    )
    if reuse_transcript:
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
        progress("transcript", "Reusing cached English transcript", 0.45)
    else:
        progress("audio", "Extracting audio — a long recording can take a few minutes", 0.08)
        raw_segments = transcribe_recording(
            client, ffmpeg, recording, transcription_model, progress
        )
        progress("translate", "Translating transcript into English", 0.55)
        segments = translate_segments(
            client,
            analysis_model,
            recording.name,
            raw_segments,
            progress,
        )
        segments_path.write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        transcript_txt_path.write_text(format_transcript(segments), encoding="utf-8")

    if not segments:
        raise RuntimeError("Transcription was empty.")

    meet_path = find_meet_transcript(recording.stem)
    meet_hint = load_meet_hint(recording.stem)
    if meet_path is not None:
        progress("speakers", "Combining audio wording with Meet speaker names", 0.68)
        meet_text = extract_meet_text(meet_path)
        segments = merge_transcripts_with_llm(
            client,
            analysis_model,
            recording.name,
            segments,
            meet_hint,
            meet_text,
            progress,
        )
        segments_path.write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        transcript_txt_path.write_text(format_transcript(segments), encoding="utf-8")

    progress("analyze", f"Analyzing with {analysis_model}", 0.72)
    analysis = analyze_transcript(
        client,
        analysis_model,
        recording.name,
        format_transcript(segments),
        meet_hint=meet_hint,
    )
    analysis = sanitize_attributions(analysis, segments, meet_hint.get("names") or [])
    analysis["transcript"] = segments
    analysis["source"] = recording.name
    analysis["meet_transcript_used"] = meet_path is not None
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    notes_path.write_text(render_markdown(recording, analysis), encoding="utf-8")
    progress("done", "Notes ready", 1.0)
    return analysis


def transcribe_recording(
    client: OpenAI,
    ffmpeg: str,
    recording: Path,
    model: str,
    progress: ProgressFn | None = None,
) -> list[dict]:
    work_dir = CACHE_DIR / safe_stem(recording.stem)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    if model in {"gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}:
        model = "gpt-4o-transcribe-diarize"

    audio_path = work_dir / "audio.mp3"
    extract_audio(ffmpeg, recording, audio_path)
    chunks = split_audio(ffmpeg, audio_path, work_dir)
    segments: list[dict] = []
    for index, (chunk_path, start_seconds) in enumerate(chunks):
        if progress:
            progress(
                "transcribe",
                f"Transcribing chunk {index + 1}/{len(chunks)}",
                0.12 + 0.4 * ((index + 1) / max(len(chunks), 1)),
            )
        chunk_segments = transcribe_chunk(client, chunk_path, model)
        for segment in chunk_segments:
            rel_start = float(segment.get("start") or 0)
            rel_end = float(segment.get("end") or rel_start)
            if index > 0 and rel_start < OVERLAP_SECONDS:
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": start_seconds + rel_start,
                    "end": start_seconds + rel_end,
                    "speaker": segment.get("speaker") or "Speaker",
                    "text": text,
                }
            )
    return segments


def transcribe_chunk(client: OpenAI, chunk_path: Path, model: str) -> list[dict]:
    if model == "whisper-1":
        with chunk_path.open("rb") as audio_file:
            result = client.audio.translations.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
            )
        return segments_from_result(result, default_speaker="Speaker")

    kwargs: dict = {"model": model}
    if model == "gpt-4o-transcribe-diarize":
        kwargs["response_format"] = "diarized_json"
        kwargs["chunking_strategy"] = "auto"
    else:
        kwargs["prompt"] = TRANSCRIPTION_PROMPT
        if model == "gpt-transcribe":
            kwargs["languages"] = ["ur", "en"]

    with chunk_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(file=audio_file, **kwargs)
    return segments_from_result(result, default_speaker="Speaker")


def segments_from_result(result: object, default_speaker: str) -> list[dict]:
    payload = model_to_dict(result)
    raw_segments = payload.get("segments") or []
    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            item = model_to_dict(item)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(item.get("start") or 0),
                "end": float(item.get("end") or item.get("start") or 0),
                "speaker": normalize_speaker_label(item.get("speaker") or default_speaker),
                "text": text,
            }
        )
    if segments:
        return segments
    text = str(payload.get("text") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "speaker": normalize_speaker_label(default_speaker), "text": text}]


def translate_segments(
    client: OpenAI,
    model: str,
    filename: str,
    segments: list[dict],
    progress: ProgressFn | None = None,
) -> list[dict]:
    if not segments:
        return []
    translated: list[dict] = [dict(segment) for segment in segments]
    batches = batch_indices(segments, TRANSLATE_BATCH_CHARS)
    for batch_number, indices in enumerate(batches, start=1):
        if progress:
            progress(
                "translate",
                f"Translating into English ({batch_number}/{len(batches)})",
                0.52 + 0.16 * (batch_number / max(len(batches), 1)),
            )
        payload = [
            {
                "index": index,
                "speaker": normalize_speaker_label(segments[index].get("speaker")),
                "text": segments[index]["text"],
            }
            for index in indices
        ]
        response = client.responses.create(
            model=model,
            instructions=TRANSLATE_INSTRUCTIONS,
            input=(
                f"Recording filename: {filename}\n\n"
                "Segments:\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        )
        text = (getattr(response, "output_text", None) or "").strip()
        parsed = extract_json(text)
        by_index = {
            int(item["index"]): item
            for item in parsed.get("segments") or []
            if isinstance(item, dict) and "index" in item
        }
        for index in indices:
            item = by_index.get(index)
            if not item:
                continue
            if item.get("text"):
                translated[index]["text"] = str(item["text"]).strip()
            translated[index]["speaker"] = normalize_speaker_label(
                segments[index].get("speaker")
            )
    return translated


def merge_transcripts_with_llm(
    client: OpenAI,
    model: str,
    filename: str,
    segments: list[dict],
    meet_hint: dict,
    meet_text: str,
    progress: ProgressFn | None = None,
) -> list[dict]:
    names = [str(name).strip() for name in (meet_hint.get("names") or []) if str(name).strip()]
    meet_for_model = meet_text.strip()
    if len(meet_for_model) > 40_000:
        meet_for_model = meet_for_model[:40_000] + "\n[Meet transcript truncated]"

    merged = [dict(segment) for segment in segments]
    batches = batch_indices(segments, MERGE_BATCH_CHARS)
    for batch_number, indices in enumerate(batches, start=1):
        if progress:
            progress(
                "speakers",
                f"Combining transcripts ({batch_number}/{len(batches)})",
                0.62 + 0.08 * (batch_number / max(len(batches), 1)),
            )
        payload = [
            {
                "index": index,
                "start": round(float(segments[index].get("start") or 0), 1),
                "time": format_timestamp(float(segments[index].get("start") or 0)),
                "speaker": segments[index].get("speaker") or "Speaker",
                "text": segments[index].get("text") or "",
            }
            for index in indices
        ]
        roster = ", ".join(names) if names else "(read names from the Meet transcript)"
        response = client.responses.create(
            model=model,
            instructions=MERGE_INSTRUCTIONS,
            input=(
                f"Recording filename: {filename}\n"
                f"Likely participants: {roster}\n\n"
                "GOOGLE MEET / GEMINI TRANSCRIPT:\n"
                f"{meet_for_model}\n\n"
                "AUDIO SEGMENTS (keep this wording; only return speaker per index):\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        )
        text = (getattr(response, "output_text", None) or "").strip()
        try:
            parsed = extract_json(text)
        except json.JSONDecodeError:
            continue
        allowed = set(indices)
        for item in parsed.get("segments") or []:
            if not isinstance(item, dict) or "index" not in item:
                continue
            try:
                index = int(item["index"])
            except (TypeError, ValueError):
                continue
            if index not in allowed:
                continue
            proposed = str(item.get("speaker") or "").strip()
            if not proposed:
                continue
            if speaker_from_meet(proposed, names):
                merged[index]["speaker"] = proposed
    return merged


def speaker_from_meet(name: str, names: list[str]) -> bool:
    if names:
        return speaker_name_allowed(name, names) or name in names
    return looks_like_person_name(name)


def analyze_transcript(
    client: OpenAI,
    model: str,
    filename: str,
    transcript: str,
    meet_hint: dict | None = None,
) -> dict:
    if len(transcript) <= LONG_TRANSCRIPT_CHARS:
        return request_analysis(
            client, model, filename, transcript, meet_hint=meet_hint
        )

    sections = split_transcript(transcript, LONG_TRANSCRIPT_CHARS)
    partials = [
        request_analysis(
            client, model, filename, section, meet_hint=meet_hint
        )
        for section in sections
    ]
    merge_input = json.dumps(partials, indent=2, ensure_ascii=False)
    return request_analysis(
        client,
        model,
        filename,
        "Merge these partial meeting analyses into one coherent JSON object "
        "with the same schema. Deduplicate action items and decisions. Keep "
        "citations. Keep only facts supported by the partials.\n\n"
        f"{merge_input}",
        merging=True,
        meet_hint=meet_hint,
    )


def request_analysis(
    client: OpenAI,
    model: str,
    filename: str,
    transcript: str,
    merging: bool = False,
    meet_hint: dict | None = None,
) -> dict:
    extra = ""
    names = (meet_hint or {}).get("names") or []
    turns = (meet_hint or {}).get("turns") or []
    if names:
        extra += (
            "People on this call, from the attached Google Meet PDF. Use only these "
            "names for speakers. Do not invent names: "
            + ", ".join(names)
            + ".\n\n"
        )
        compact = [
            {
                "start": round(float(turn.get("start") or 0), 1),
                "speaker": turn.get("speaker"),
                "text": str(turn.get("text") or "")[:180],
            }
            for turn in turns[:180]
        ]
        extra += (
            "Attached Meet transcript (noisy extra context; if it conflicts with "
            "the audio transcript, trust the audio transcript):\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n\n"
        )
    else:
        extra += (
            "No Google Meet transcript was attached. Keep Speaker A, Speaker B, "
            "Speaker C labels. Do not invent personal names for speakers.\n\n"
        )
    user_input = (
        f"Recording filename: {filename}\n\n"
        + extra
        + ("Partial analyses to merge:\n\n" if merging else "Audio transcript:\n\n")
        + transcript
    )
    response = client.responses.create(
        model=model,
        instructions=ANALYSIS_INSTRUCTIONS,
        input=user_input,
    )
    text = (getattr(response, "output_text", None) or "").strip()
    if not text:
        raise RuntimeError("Analysis model returned an empty response.")
    try:
        parsed = extract_json(text)
    except json.JSONDecodeError:
        parsed = {
            "title": Path(filename).stem,
            "date": None,
            "participants": [],
            "overview": [{"text": text, "start": 0}],
            "discussion_notes": [],
            "decisions": [],
            "action_items": [],
            "open_questions": [],
        }
    return normalize_analysis(parsed, filename)


def normalize_analysis(analysis: dict, filename: str) -> dict:
    title = str(analysis.get("title") or Path(filename).stem).strip()
    overview = cited_items(analysis.get("overview"))
    notes = []
    for block in analysis.get("discussion_notes") or []:
        if isinstance(block, str):
            notes.append({"topic": "Discussion", "items": cited_items(block)})
            continue
        if not isinstance(block, dict):
            continue
        items = block.get("items")
        if items is None:
            items = block.get("notes")
        notes.append(
            {
                "topic": str(block.get("topic") or "Discussion").strip(),
                "items": cited_items(items),
            }
        )
    return {
        "title": title,
        "date": analysis.get("date"),
        "participants": [
            str(name).strip()
            for name in (analysis.get("participants") or [])
            if str(name).strip()
        ],
        "overview": overview,
        "discussion_notes": notes,
        "decisions": cited_items(analysis.get("decisions")),
        "action_items": normalize_action_items(analysis.get("action_items") or []),
        "open_questions": cited_items(analysis.get("open_questions")),
    }


def cited_items(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"text": value.strip(), "start": None}] if value.strip() else []
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("notes") or "").strip()
        if not text:
            return []
        item = {"text": text, "start": coerce_seconds(value.get("start"))}
        if value.get("actor"):
            item["actor"] = value["actor"]
        return [item]
    items = []
    if isinstance(value, list):
        for entry in value:
            items.extend(cited_items(entry))
    return items


def normalize_action_items(items: list) -> list[dict]:
    normalized = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                normalized.append(
                    {"owner": None, "task": item.strip(), "due": None, "start": None}
                )
            continue
        if not isinstance(item, dict):
            continue
        task = str(item.get("task") or item.get("text") or "").strip()
        if not task:
            continue
        normalized.append(
            {
                "owner": item.get("owner") or None,
                "task": task,
                "due": item.get("due") or None,
                "start": coerce_seconds(item.get("start")),
            }
        )
    return normalized


def coerce_seconds(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = re.fullmatch(r"(\d+):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return None
    parts = [int(part) for part in match.groups() if part is not None]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return None


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        stamp = format_timestamp(float(segment.get("start") or 0))
        speaker = segment.get("speaker") or "Speaker"
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(f"[{stamp}] {speaker}: {text}")
    return "\n".join(lines)


def render_markdown(recording: Path, analysis: dict) -> str:
    people = ", ".join(analysis.get("participants") or []) or "Not clearly identified"
    date = analysis.get("date") or "Not stated in the recording"
    lines = [
        f"# {analysis.get('title') or recording.stem}",
        "",
        f"**Source:** `{recording.name}`  ",
        f"**Date:** {date}  ",
        f"**Participants:** {people}",
        "",
        "## Overview",
        "",
    ]
    overview = analysis.get("overview") or []
    if overview:
        lines.extend(render_cited_paragraph(overview))
        lines.append("")
    else:
        lines.extend(["No overview could be produced.", ""])

    notes = analysis.get("discussion_notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        for block in notes:
            lines.append(f"### {block.get('topic') or 'Discussion'}")
            lines.append("")
            for item in block.get("items") or []:
                lines.append(f"- {item['text']}{citation_suffix(item)}")
            lines.append("")

    lines.append("## Decisions")
    lines.append("")
    decisions = analysis.get("decisions") or []
    if decisions:
        for item in decisions:
            actor = item.get("actor")
            prefix = f"**{actor}:** " if actor else ""
            lines.append(f"- {prefix}{item['text']}{citation_suffix(item)}")
    else:
        lines.append("- None clearly recorded.")
    lines.append("")

    lines.append("## Action items")
    lines.append("")
    actions = analysis.get("action_items") or []
    if actions:
        for item in actions:
            owner = item.get("owner") or "Unassigned"
            due = f" — due {item['due']}" if item.get("due") else ""
            lines.append(
                f"- **{owner}:** {item['task']}{due}{citation_suffix(item)}"
            )
    else:
        lines.append("- None clearly assigned.")
    lines.append("")

    lines.append("## Open questions")
    lines.append("")
    questions = analysis.get("open_questions") or []
    if questions:
        for item in questions:
            lines.append(f"- {item['text']}{citation_suffix(item)}")
    else:
        lines.append("- None raised.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_cited_paragraph(items: list[dict]) -> list[str]:
    parts = []
    for item in items:
        parts.append(f"{item['text']}{citation_suffix(item)}")
    return [" ".join(parts)]


def citation_suffix(item: dict) -> str:
    start = coerce_seconds(item.get("start"))
    if start is None:
        return ""
    return f" [{format_timestamp(start)}]"


def parse_legacy_notes(stem: str) -> dict:
    notes_path = NOTES_DIR / f"{stem}.md"
    text = notes_path.read_text(encoding="utf-8")
    transcript_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    transcript_text = (
        transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
    )
    if "## Transcript" in text and not transcript_text:
        transcript_text = text.split("## Transcript", 1)[1].strip()
        text = text.split("## Transcript", 1)[0].rstrip()

    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    date = meta_value(text, "Date")
    participants_raw = meta_value(text, "Participants")
    participants = [
        part.strip()
        for part in re.split(r",| and ", participants_raw or "")
        if part.strip() and part.strip().lower() != "not clearly identified"
    ]
    overview = section_body(text, "Overview")
    analysis = normalize_analysis(
        {
            "title": title_match.group(1).strip() if title_match else stem,
            "date": None if date in {None, "Not stated in the recording"} else date,
            "participants": participants,
            "overview": split_sentences(overview),
            "discussion_notes": parse_legacy_topics(section_body(text, "Notes")),
            "decisions": parse_legacy_bullets(section_body(text, "Decisions")),
            "action_items": parse_legacy_actions(section_body(text, "Action items")),
            "open_questions": parse_legacy_bullets(section_body(text, "Open questions")),
        },
        f"{stem}.md",
    )
    analysis["transcript"] = parse_legacy_transcript(transcript_text)
    analysis["source"] = meta_value(text, "Source") or f"{stem}"
    analysis["legacy"] = True
    return analysis


def meta_value(text: str, label: str) -> str | None:
    match = re.search(rf"\*\*{label}:\*\*\s*(.+)", text)
    if not match:
        return None
    return match.group(1).strip().strip("`")


def section_body(text: str, heading: str) -> str:
    pattern = rf"## {re.escape(heading)}\n+(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_legacy_topics(body: str) -> list[dict]:
    if not body:
        return []
    chunks = re.split(r"^### ", body, flags=re.MULTILINE)
    topics = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if "\n" in chunk:
            title, rest = chunk.split("\n", 1)
        else:
            title, rest = chunk, ""
        items = parse_legacy_bullets(rest) or split_sentences(rest)
        topics.append({"topic": title.strip(), "items": items})
    return topics


def parse_legacy_bullets(body: str) -> list[str]:
    items = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item.lower() not in {
                "none clearly recorded.",
                "none clearly assigned.",
                "none raised.",
            }:
                items.append(item)
    return items


def parse_legacy_actions(body: str) -> list[dict]:
    items = []
    for bullet in parse_legacy_bullets(body):
        match = re.match(r"\*\*(.+?):\*\*\s*(.+)", bullet)
        if not match:
            items.append({"task": bullet})
            continue
        owner, rest = match.group(1), match.group(2)
        due = None
        due_match = re.search(r"— due (.+)$", rest)
        if due_match:
            due = due_match.group(1).strip()
            rest = rest[: due_match.start()].strip()
        items.append({"owner": owner, "task": rest, "due": due})
    return items


def parse_legacy_transcript(text: str) -> list[dict]:
    if not text.strip():
        return []
    parts = re.split(r"\[(\d+:\d{2}(?::\d{2})?)\]", text)
    if len(parts) < 3:
        return [{"start": 0, "end": 0, "speaker": "Speaker", "text": text.strip()}]
    segments = []
    for index in range(1, len(parts), 2):
        start = coerce_seconds(parts[index]) or 0
        body = parts[index + 1].strip()
        if not body:
            continue
        speaker_match = re.match(r"^([^:]{1,40}):\s*(.*)", body, re.DOTALL)
        if speaker_match and len(speaker_match.group(1).split()) <= 4:
            speaker = speaker_match.group(1).strip()
            body = speaker_match.group(2).strip()
        else:
            speaker = "Speaker"
        segments.append(
            {"start": start, "end": start, "speaker": speaker, "text": body}
        )
    return segments


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_audio(ffmpeg: str, source: Path, dest: Path) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "48k",
            str(dest),
        ],
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("Audio extraction produced an empty file.")


def split_audio(ffmpeg: str, audio_path: Path, work_dir: Path) -> list[tuple[Path, float]]:
    duration = media_duration_seconds(ffmpeg, audio_path) or 0
    size = audio_path.stat().st_size
    if duration <= MAX_CHUNK_SECONDS and size <= MAX_UPLOAD_BYTES:
        return [(audio_path, 0.0)]

    chunk_seconds = MAX_CHUNK_SECONDS
    if duration > 0 and size > MAX_UPLOAD_BYTES:
        bytes_per_second = size / duration
        size_limited = max(60, int(MAX_UPLOAD_BYTES / bytes_per_second) - 5)
        chunk_seconds = min(chunk_seconds, size_limited)

    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 0
    while start < duration:
        chunk_path = work_dir / f"chunk_{index:03d}.mp3"
        length = min(chunk_seconds, duration - start)
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{length:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "48k",
                str(chunk_path),
            ],
        )
        chunks.append((chunk_path, start))
        if start + length >= duration:
            break
        start += chunk_seconds - OVERLAP_SECONDS
        index += 1
    return chunks


def find_ffmpeg() -> str | None:
    bundled = None
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        bundled = None
    for candidate in (shutil.which("ffmpeg"), bundled):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def media_duration_seconds(ffmpeg: str, path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return float(completed.stdout.strip())
        except ValueError:
            pass
    completed = subprocess.run(
        [ffmpeg, "-i", str(path)], capture_output=True, text=True, check=False
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg failed: {detail}")


def normalize_speaker_label(label: object) -> str:
    text = str(label or "Speaker").strip() or "Speaker"
    letter = re.fullmatch(r"(?:speaker\s*)?([A-Za-z])$", text, re.I)
    if letter:
        return f"Speaker {letter.group(1).upper()}"
    number = re.fullmatch(r"(?:speaker\s*)?(\d+)$", text, re.I)
    if number:
        index = int(number.group(1))
        if 1 <= index <= 26:
            return f"Speaker {chr(ord('A') + index - 1)}"
    return text


def speaker_name_allowed(name: str, allowed: list[str]) -> bool:
    proposed = re.sub(r"\s+", " ", name).strip().lower()
    if not proposed or proposed.startswith("speaker "):
        return False
    aliases = set()
    for person in allowed:
        full = re.sub(r"\s+", " ", person).strip().lower()
        if not full:
            continue
        aliases.add(full)
        aliases.add(full.split()[0])
    return proposed in aliases


_ATTR_VERBS = (
    "said",
    "says",
    "proposed",
    "asked",
    "noted",
    "explained",
    "acknowledged",
    "added",
    "suggested",
    "reported",
    "described",
    "confirmed",
    "emphasized",
)
_ATTR_RE = re.compile(
    r"\b((?:Speaker\s+[A-Z])|[A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){0,2})\s+"
    + r"("
    + "|".join(_ATTR_VERBS)
    + r")\b"
)


def unique_speaker_labels(segments: list[dict]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        label = str(segment.get("speaker") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def person_is_allowed(name: str, allowed_people: list[str], speaker_labels: list[str]) -> bool:
    text = re.sub(r"\s+", " ", name or "").strip()
    if not text:
        return False
    normalized = normalize_speaker_label(text)
    if re.fullmatch(r"Speaker [A-Z]$", normalized):
        return True
    if allowed_people:
        return speaker_name_allowed(text, allowed_people)
    speakers_lower = {label.lower() for label in speaker_labels}
    return text.lower() in speakers_lower or normalized.lower() in speakers_lower


def speaker_at_time(segments: list[dict], start: object) -> str:
    if not segments:
        return "Speaker"
    try:
        target = float(start)
    except (TypeError, ValueError):
        return str(segments[0].get("speaker") or "Speaker")
    best_label = str(segments[0].get("speaker") or "Speaker")
    best_start = -1.0
    for segment in segments:
        seg_start = float(segment.get("start") or 0)
        if seg_start <= target + 0.75 and seg_start >= best_start:
            best_start = seg_start
            best_label = str(segment.get("speaker") or best_label)
    return best_label


def rewrite_invented_attributions(
    text: str,
    start: object,
    segments: list[dict],
    allowed_people: list[str],
    speaker_labels: list[str],
    spoken_blob: str,
) -> str:
    if not text:
        return text
    replacement = speaker_at_time(segments, start)

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        verb = match.group(2)
        if person_is_allowed(name, allowed_people, speaker_labels):
            return match.group(0)
        if re.search(rf"\b{re.escape(name)}\b", spoken_blob, re.I):
            return match.group(0)
        return f"{replacement} {verb}"

    return _ATTR_RE.sub(repl, text)


def sanitize_attributions(
    analysis: dict,
    segments: list[dict],
    meet_names: list[str],
) -> dict:
    speaker_labels = unique_speaker_labels(segments)
    allowed_people = [str(name).strip() for name in meet_names if str(name).strip()]
    spoken_blob = " ".join(str(segment.get("text") or "") for segment in segments)

    def fix_text(item: dict, key: str = "text") -> None:
        item[key] = rewrite_invented_attributions(
            str(item.get(key) or ""),
            item.get("start"),
            segments,
            allowed_people,
            speaker_labels,
            spoken_blob,
        )

    for item in analysis.get("overview") or []:
        if isinstance(item, dict):
            fix_text(item)
    for item in analysis.get("open_questions") or []:
        if isinstance(item, dict):
            fix_text(item)
    for item in analysis.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        fix_text(item)
        actor = str(item.get("actor") or "").strip()
        if actor and not person_is_allowed(actor, allowed_people, speaker_labels):
            item["actor"] = speaker_at_time(segments, item.get("start"))
    for block in analysis.get("discussion_notes") or []:
        if not isinstance(block, dict):
            continue
        for item in block.get("items") or []:
            if isinstance(item, dict):
                fix_text(item)
    for item in analysis.get("action_items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("task"):
            fix_text(item, "task")
        owner = str(item.get("owner") or "").strip()
        if owner and not person_is_allowed(owner, allowed_people, speaker_labels):
            item["owner"] = speaker_at_time(segments, item.get("start"))

    if allowed_people:
        kept = [
            name
            for name in unique_speaker_labels(segments)
            if person_is_allowed(str(name), allowed_people, [])
        ]
        analysis["participants"] = kept or list(allowed_people)
    else:
        analysis["participants"] = speaker_labels
    return analysis


def format_timestamp(seconds: float) -> str:
    total = int(max(0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", stem).strip("._")
    return cleaned or "recording"


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", cleaned, 0)
    return parsed


def split_transcript(transcript: str, max_chars: int) -> list[str]:
    paragraphs = transcript.strip().split("\n")
    sections: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + 1
        if current and size + extra > max_chars:
            sections.append("\n".join(current))
            current = [paragraph]
            size = extra
        else:
            current.append(paragraph)
            size += extra
    if current:
        sections.append("\n".join(current))
    return sections or [transcript]


def batch_indices(segments: list[dict], max_chars: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, segment in enumerate(segments):
        extra = len(segment.get("text") or "") + 40
        if current and size + extra > max_chars:
            batches.append(current)
            current = [index]
            size = extra
        else:
            current.append(index)
            size += extra
    if current:
        batches.append(current)
    return batches


def model_to_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    for method in ("model_dump", "to_dict", "dict"):
        attr = getattr(value, method, None)
        if callable(attr):
            try:
                dumped = attr() if method != "model_dump" else attr()
            except TypeError:
                dumped = attr(exclude_none=True) if method == "model_dump" else attr()
            if isinstance(dumped, dict):
                return dumped
    payload = {}
    for key in ("text", "segments", "speaker", "start", "end"):
        if hasattr(value, key):
            payload[key] = getattr(value, key)
    return payload
