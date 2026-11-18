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
Build and harden direct mode before any server work.

* Add a new CLI command, ovox chunk, to perform the video/audio chunking and leave files in some dir.
It splits audio and video but does not do any transcription.
Preview and real run must use the same code. Both should use one function that decides where the chunks go, and that real transcription also uses.
Consider that the transcription service used (target) will have limits: such as file size limit (later: optionally a transfer duration limit).
This file size limit should be configurable in .env.example
chunking must respect the file size limit

CLI command and options:
- ovox chunk <file> --output-dir <dir>
- ovox chunk <file> --output-dir <dir> --dry-run

Plan:
What the plan should show (plain text and --json):
About the input: file, format, codec, duration, sample rate and channels, and for video, which audio track is used.
The settings used: maximum chunk length or size, overlap, the splitting method (fixed length or at silences), and where each setting came from (config, flag or default).
One row per chunk: index, start, end, duration and estimated size. If splitting at silences, also why each cut was placed there.
Totals and warnings: number of chunks and total audio length, plus warnings such as “chunk 7 is over the API size limit” or “no silence found, cut at the maximum length”.

Output a manifest file:
The dry-run plan JSON, saved to the folder: same fields, same structure.

Chunking:
audio: ovox chunk on an audio file should output one or more audio chunks.
video: ovox chunk on a video should output audio chunks by default.
One chunk is a normal result. If the input is already within the limits, ovox chunk should still write a single chunk and say that no split was needed.
Output files should be named with consistency, from the original filename and add 
ex. chunk_003_754.5-910.2.wav

How the target’s limits fit with splitting at silences

Turn the file size limit into a length limit. Using the bitrate of the format you send, work out a maximum chunk length, with a safety margin (say 90–95% of the limit), because container overhead and variable bitrate make sizes slightly unpredictable.
Look for a silence before that limit. Cut at the last suitable silence before the maximum length. If none is found, cut at the maximum length, and show that as a warning in the plan.
Check the actual size after writing. If a chunk still ends up over the limit, flag it or split it again, rather than letting the upload fail later.
The format you send matters a lot. Compressed mono speech can fit several times more minutes under the same size limit than uncompressed stereo. That’s worth deciding early, if it isn’t already, because it changes how many chunks you get.
Make room for a later upload time limit. Keep the limits together as a per-target setting: maximum file size now, maximum upload time later. An upload time limit then becomes another input to the same maximum-length calculation, based on an assumed upload speed, without changing how splitting works.

Tests (same approach as for check, checking structure rather than wording):

Unit tests on the planning function, using made-up durations and no audio files. Chunks cover the whole input with no gaps, overlaps match the setting, no chunk exceeds the maximum, and boundary cases behave (very short input, input exactly at the limit, zero overlap).
Integration tests with short audio and video clips generated by ffmpeg. The dry-run JSON should match what ovox chunk actually writes: same number of chunks, and durations within a small tolerance. A dry run should create no files.


Suggested order:
Next in line:

5. **Transcription**: call the STT endpoint for each chunk, with retries/backoff and
   bounded concurrency.
Add an ovox transcribe subcommand that calls transcribe_file.
Install a SIGTERM handler in main that raises SystemExit, so temp cleanup also runs when the process is killed.
CLI command and options:
- ovox transcribe <file> --dry-run
- ovox transcribe <file>

Resuming:
After chunking or after a stopped/failed transcription:
Resume means: read the manifest, check the source and settings match, and transcribe the chunks with no transcript file.

Track transcription progress separately, by checking whether each chunk’s transcript file exists (e.g. chunk_003.json), so the manifest never changes after it’s written.


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
- consider a resume mode after chunking, to continue with transcription
- add a CLI command to only **preview** chunking for example by a dry run to see what would be done

## Known Issues
Make a note whether each issie should be addressed soon or delayed.
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
