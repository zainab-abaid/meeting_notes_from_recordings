# Meeting notes from recordings

Local app for Google Meet recordings. Drop in a video, optionally attach the Meet transcript PDF, and get timestamped notes you can click to play that moment.

## Setup

Python 3.10+ and an [OpenAI API key](https://platform.openai.com/api-keys). Install [ffmpeg](https://ffmpeg.org/) if you can (`brew install ffmpeg` on macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env`.

## Run

```bash
source .venv/bin/activate
python app.py
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787).

Without a Meet PDF, speakers stay Speaker A, B, C. Re-analyze reuses the cached transcript.

## CLI

```bash
python process_meeting.py
python process_meeting.py recordings/meeting.mp4 --reanalyze
```

`--force` re-transcribes. `--reanalyze` keeps the transcript and regenerates notes.
