# AGENTS.md — Open Vox (ovox)

## Purpose
Open Vox (`ovox`) transcribes audio and video to text and supports custom/self-hosted
speech-to-text (STT) models. It is published on PyPI as a CLI. A job server and a
browser add-on will follow later.

## Architecture (one core, two modes)
The same CLI commands work in two modes:

- **Direct mode** (default, the public PyPI use case): local `ffmpeg` extracts the
  audio, then the CLI calls the user's own STT endpoint with their own API key.
  No server is needed.
- **Server mode** (`--server URL` or config, later): the CLI extracts audio locally
  (~10 MB per hour), uploads it to the job API and polls for the result. Browsers
  upload raw video, and the server handles both.

Both modes share one core pipeline:
`validate input → extract/normalise audio (ffmpeg) → chunk → STT per chunk → stitch → format output`

Keep the core free of CLI and HTTP-server concerns so the server can wrap it later as
a thin job layer.

## Current epic: CLI direct mode
Build and harden direct mode before any server work. Suggested order:

1. **Config and API-key check**: load settings from `.env` and the environment.
   Add a command (for example `ovox check`) that makes a cheap authenticated
   call to the STT endpoint and reports a clear OK or failure. Never print the key.
2. **Input validation**: the file exists, is readable and non-empty, and `ffprobe`
   finds at least one audio stream. Give clear errors for missing files, for
   unsupported or corrupt media, and for video files without audio. Check that
   `ffmpeg` and `ffprobe` are on PATH.
3. **Audio extraction**: convert to mono, 16 kHz, compressed (for example MP3) in a temp dir, and always clean up.
4. **Chunking**: split under the endpoint's size and duration limits, preferably on
   silence and with a small overlap.
5. **Transcription**: call the STT endpoint for each chunk, with retries/backoff and
   bounded concurrency.
6. **Stitching**: merge chunk results, offset timestamps and de-duplicate the overlaps.
7. **Output formats**: `txt`, `srt`, `vtt`, `json`.
8. **`--json` timing output**: report per-stage timings (probe, extract, chunk, STT,
   stitch). These are used to benchmark direct mode against server mode later and
   to see where time goes (upload, queueing, STT).

Later (not now): define the job API contract early as an OpenAPI spec from FastAPI,
so the browser add-on can be developed against a stub in parallel.

## Conventions
- Python, packaged via `pyproject.toml`. Check it for the actual layout, entry
  point and dependencies before adding new ones.
- Keep dependencies few. Call `ffmpeg`/`ffprobe` through `subprocess` (no shell
  strings) and parse `ffprobe -print_format json`.
- Secrets live in `.env` only. `.env` stays git-ignored; keep `.env.example`
  updated whenever a setting is added.
- Use typed functions and small modules. Errors shown to users should be
  actionable messages with non-zero exit codes, not tracebacks.
- Write tests with `pytest`. Mock the STT HTTP calls, and generate tiny media
  fixtures with ffmpeg (for example a 2 s sine tone, plus a video with no audio)
  rather than committing large files. Tests must pass without network access or
  a real API key.
- Run tests before considering a task done.

## Don'ts
- Don't build server-mode or browser code during this epic.
- Don't commit real media, transcripts or keys.
- Don't hard-code a single STT vendor. Keep the endpoint and model configurable
  (an OpenAI-compatible `/v1/audio/transcriptions` is a sensible first target).
