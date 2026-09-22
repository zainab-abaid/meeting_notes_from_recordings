#!/usr/bin/env python3
"""CLI: transcribe unprocessed recordings and write notes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import pipeline

ROOT = pipeline.ROOT


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    ffmpeg = pipeline.find_ffmpeg()
    if ffmpeg is None:
        print(
            "Could not find ffmpeg. Install it (e.g. `brew install ffmpeg`) "
            "or keep the imageio-ffmpeg package installed.",
            file=sys.stderr,
        )
        return 1

    try:
        recordings = resolve_recordings(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not recordings:
        print(
            f"No recordings found in {pipeline.RECORDINGS_DIR}.\n"
            "Drop a Google Meet download into recordings/, then run this script "
            "or start the UI with `python app.py`."
        )
        return 0

    client = OpenAI()
    transcription_model, analysis_model = pipeline.default_models()
    failures = 0
    for recording in recordings:
        print(f"\n=== {recording.name} ===")
        try:
            pipeline.process_recording(
                recording,
                client=client,
                ffmpeg=ffmpeg,
                transcription_model=transcription_model,
                analysis_model=analysis_model,
                force=args.force,
                reanalyze=args.reanalyze,
                on_progress=lambda _step, message, _frac: print(f"  {message}"),
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"Failed on {recording.name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting recording and write markdown notes."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Specific recording file. Defaults to unprocessed files in recordings/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe and re-analyze even if notes already exist.",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Reuse a cached transcript and only regenerate the notes.",
    )
    return parser.parse_args()


def resolve_recordings(args: argparse.Namespace) -> list[Path]:
    if args.path:
        path = Path(args.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Recording not found: {path}")
        if path.suffix.lower() not in pipeline.AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]

    candidates = sorted(pipeline.list_media_files(), key=lambda p: p.stat().st_mtime)
    if args.force or args.reanalyze:
        return candidates
    pending = [path for path in candidates if pipeline.is_unprocessed(path)]
    skipped = len(candidates) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-processed recording(s).")
    return pending


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
