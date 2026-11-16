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

Design decisions are in docs/design/; read them before changing the pipeline”

## Status

CLI Direct mode is in progress.

The repository structure is in place with a basic CLI, validation of media files and API key. Unit and integration tests of this is done.

**Config and API-key check**: load settings from `.env` and the environment.
   Add a command (`ovox check`) that makes a cheap authenticated
   call to the STT endpoint and reports a clear OK or failure.

**Input validation**: the file exists, is readable and non-empty, and `ffprobe`
   finds at least one audio stream. Give clear errors for missing files, for
   unsupported or corrupt media, and for video files without audio. Check that
   `ffmpeg` and `ffprobe` are on PATH.

**Audio extraction**: convert to mono, 16 kHz, compressed (for example MP3) in a temp dir, and always clean up.
There should be no new runtime dependencies, since pyproject.toml has none. Everything uses the standard library plus ffmpeg.
Add tests including one that runs real ffmpeg.
Also, build SttConfig from your existing settings loader and reuse whatever HTTP helper ovox check uses.

Audio extraction (audio.py)
Describing how it should work: ffmpeg takes the first audio stream only (-map 0:a:0) and outputs mono, 16 kHz, 32 kbps MP3. That works out to about 14 MB per hour of audio.
Everything happens inside workdir(), a TemporaryDirectory context manager. It cleans up on success, on exceptions and on Ctrl-C.
If ffmpeg fails, the error shows the last few lines of its stderr. There are timeouts, and an empty output file counts as an error.

**Chunking**: split under the endpoint's size and duration limits, preferably on
   silence and with a small overlap.

## Current epic: CLI direct mode
Build and harden direct mode before any server work. Suggested order:

Next in line:

5. **Transcription**: call the STT endpoint for each chunk, with retries/backoff and
   bounded concurrency.
Add an ovox transcribe subcommand that calls transcribe_file.
Install a SIGTERM handler in main that raises SystemExit, so temp cleanup also runs when the process is killed.

Design integration test with real audio files.
a. Committed files: to test actual transcription, commit one or two short speech clips, a few seconds each (max 500kb each). Put them in tests/data/ and use public-domain (CC0) audio or  own recording.
b. Local folder: set an environment variable at the folder and skip when it’s unset, so CI is not affected.

6. **Stitching**: merge chunk results, offset timestamps and de-duplicate the overlaps.
7. **Output formats**: `txt`, `srt`, `vtt`, `json`.
8. **`--json` timing output**: report per-stage timings (probe, extract, chunk, STT,
   stitch). These are used to benchmark direct mode against server mode later and
   to see where time goes (upload, queueing, STT).

## Work to wait with, later (not now):
Define the job API contract early as an OpenAPI spec from FastAPI,
so the browser add-on can be developed against a stub in parallel.

Also:
- putting ChunkSpec’s fields in a clearer order
- skipping chunks that are entirely silence

## Known issues to be adressed later

-

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
- Don't make any commits or other git work other than git reads.
