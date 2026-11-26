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

Done so far: the repository structure with a basic CLI, media-file validation
and the API-key check (`ovox check`); audio extraction and the chunk command
(`ovox chunk`, with `--dry-run`/`--plan-only`/`--json` and a manifest); and
transcription and the end-to-end pipeline (`ovox transcribe`, with
`--dry-run`/`--resume`/`-o`/`--json`): plain STT requests with
retries/backoff, bounded concurrency, a work folder for resume, word-matching
merge of hard-cut overlaps, and text-only output (segments JSON when the
endpoint returns segments). All have unit and integration tests; the STT
endpoint is faked with a local `http.server` in the integration tests.

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


5. **Transcription**: call the STT endpoint for each chunk, with retries/backoff and
   bounded concurrency.
Example transcription request:
export AUDIO_FILE="user-short-test-en.wav"
export API_KEY="sk-"
curl url/api/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F "model=voxtral-small-24b" \
  -F "timestamp_granularities[]=segment" \
  -F "response_format=verbose_json" \
  -F "file=@$HOME/data/audio/$AUDIO_FILE;type=audio/wav"
Example response:
{"text":"Dancing in the masquerade, idle truth in plain sight jaded, pop, roll, click, shot, who will I be today or not? But such a tide as moving seems asleep, too full for sound and foam, when that which drew from out the boundless deep turns again home, twilight and evening bell, and after that?","filename":"1150441d-8a3f-41b7-a9f3-d4516d10fc2b.wav"}

mp3 is supported. example response:
{"text":"Hi there, this is a test in English. Okay, see you later. Bye bye.","filename":"d387ab5c-8ce9-49e3-baa8-c613cababa51.mp3"}

timestamp_granularities[]=word is also supported.

Note about timestamps: Voxtral on vLLM doesn't provide timestamps, and Open WebUI silently drops the options. So the beta is text-only.
However, the implementation should work or give an option for a model host that does not work with words and segments in the response. That means text-only merge, and no segments JSON or subtitles. Timestamps are a bonus when the endpoint provides them, and longer term ask for timestamps to be passed through the gateway.

CLI command and options:
- ovox transcribe <file> --dry-run
- ovox transcribe <file>

--dry-run means: run all the local steps (validate, extract, chunk) into the work folder and print the plan (number of chunks, lengths, how many are already transcribed), but make no STT requests. A later real run then resumes from it.

The output transcript should be a .txt file created next to the input or in the current directory, with -o to choose, and progress printed to stderr. A JSON file with the global segments could be written alongside it.

Resuming: After chunking or after a stopped/failed transcription: Resume means: read the manifest, check the source and settings match, and transcribe the chunks with no transcript file.

The limits are implemented in the chunking process.
Track transcription progress separately, by checking whether each chunk’s transcript file exists (e.g. chunk_003.json), so the manifest never changes after it’s written.

Send a plain request: just model and file, with no verbose_json or timestamp options. That avoids depending on how Open WebUI happens to treat options it doesn't support.
Merge by word matching only. After a hard cut, the overlap words are removed by matching the end of one chunk's text with the start of the next. Silence cuts need no merging. This is the only merge for now, not a fallback.
Wait with: the segments JSON, the global timeline, subtitles and the optional speaker field. Each chunk_NNN.json still stores the text, the model asked for, and anything else useful from the reply.
Remove or rewrite the timestamp bullets in your plan and in transcription.md, which still describe verbose_json as the next step.

Implementation suggestions:
- Add an ovox transcribe subcommand that calls transcribe_file.
Install a SIGTERM handler in main that raises SystemExit, so temp cleanup also runs when the process is killed.
- Retries cover timeouts, network errors and HTTP 408, 409, 425, 429 and 5xx. They use capped exponential backoff with full jitter and honour Retry-After.
- HTTP 401/403 fails at once and points the user to ovox check. Other 4xx errors fail without retrying.
- Concurrency is bounded with a ThreadPoolExecutor. The first fatal error cancels any chunks not yet started.
- merge_texts removes words duplicated by a hard-cut overlap by matching the end of one chunk's words against the start of the next.
- The temp directory can no longer always be deleted.
- Each chunk's transcript is a file, written safely. Write each result to chunk_003.json.tmp, then rename it to chunk_003.json with os.replace.
- Use the timestamps for a global timeline. Add each chunk's start time to its segment times. That gives correct timestamps for the whole file, which is what subtitle output needs. Keep the word-matching dedupe for overlaps. Apply it only to the first segment after a hard cut. Most cuts land on silence and have no overlap, so this is rare.

Next: we can then add a pipeline.py with transcribe_file(): Direct mode end to end: extract -> chunk -> transcribe -> merge



--

The format you send matters a lot. Compressed mono speech can fit several times more minutes under the same size limit than uncompressed stereo. That’s worth deciding early, if it isn’t already, because it changes how many chunks you get.
Make room for a later upload time limit. Keep the limits together as a per-target setting: maximum file size now, maximum upload time later. An upload time limit then becomes another input to the same maximum-length calculation, based on an assumed upload speed, without changing how splitting works.


Suggested order:
Next in line:

Design integration test with real audio files.
a. Committed files: to test actual transcription, commit one or two short speech clips, a few seconds each (max 500kb each). Put them in tests/data/ and use public-domain (CC0) audio or  own recording.
b. Local folder: set an environment variable at the folder and skip when it’s unset, so CI is not affected.

6. **Stitching**: merge chunk results, offset timestamps and de-duplicate the overlaps.
7. **Output formats**: `txt`, `srt`, `vtt`, `json`.
8. **`--json` timing output**: report per-stage timings (probe, extract, chunk, STT,
   stitch). These are used to benchmark direct mode against server mode later and
   to see where time goes (upload, queueing, STT).

## Future work, later (not now):

- Define the job API contract as an OpenAPI spec from FastAPI, so the browser add-on can be developed against a stub in parallel.
- Speaker diarization: detecting multiple speakers. Two options: the STT endpoint does it vs a separate diarization step.
- putting ChunkSpec’s fields in a clearer order
- skipping chunks that are entirely silence
- Two runs on the same file at once. A lock file in the work folder would prevent this; it can wait until after the beta.
- Re-run behaviour: if only settings changed, with chunks already done, can add a --restart flag.

## Known Issues
Make a note whether each issue should be addressed soon or delayed.
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
- Don't stray from the core task given. Any refactoring or other changes should first be explicitly asked.
- Don't build server-mode or browser code during this epic.
- Don't commit real media, transcripts or keys.
- Don't hard-code a single STT vendor. Keep the endpoint and model configurable
  (an OpenAI-compatible `/v1/audio/transcriptions` is a sensible first target).
- Don't make any commits or other git work other than git reads.
