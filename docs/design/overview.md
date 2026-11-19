# Open Vox design overview

Status: current for the first beta (CLI direct mode). Last updated 2026-11.

Open Vox (`ovox`) transcribes audio and video files to text through a
speech-to-text (STT) endpoint, with support for custom transcription models.
This document gives the big picture and the cross-cutting choices. Details live
in the other design documents:

- [Audio pipeline](audio-pipeline.md): input validation, audio extraction, chunking.
- [Transcription](transcription.md): calling the STT endpoint and merging the results.
- [Testing](testing.md): how the code is tested.

## Scope: direct mode first

The first deliverable is **CLI direct mode**: `ovox` runs on the user's machine,
processes a local file and calls the STT endpoint directly. A server mode
(upload, job queue, results API) comes later.

Why direct mode first:

- It exercises the whole hard part (media handling, chunking, retries) with the
  fewest moving parts.
- A later server can reuse the same pipeline as a library, so the pipeline must
  not depend on the CLI (no printing or exiting from pipeline code; report
  progress through callbacks and failures through exceptions).

## Pipeline

```
input file
  -> validate      (ovox.media)      file exists, readable, non-empty, has an audio stream
  -> extract       (ovox.audio)      mono 16 kHz MP3 in a private temp dir
  -> plan chunks   (ovox.plan)       silence detection + chunk plan within endpoint limits
  -> cut chunks    (ovox.audio)      one file per planned chunk, stream copy
  -> transcribe    (STT endpoint)    per chunk, retries, bounded concurrency
  -> merge                           join texts, remove words repeated across overlaps
  -> transcript
```

`ovox.plan.build_plan` is the single place that decides where the chunks go. It
is used by every mode of `ovox chunk` and, in due course, by real transcription,
so the plan a preview prints is always an accurate description of what would be
uploaded. `ovox chunk` splits a file into chunks that fit the endpoint's limits,
without doing any transcription, in three modes:

- **real** (default): writes the chunks plus a JSON manifest to the output dir.
- **`--plan-only`**: writes just the manifest, no chunks.
- **`--dry-run`**: prints the plan and writes nothing — no files, no directory —
  so it is safe to run anywhere.

All intermediate files live in one temp directory per run, which is always
removed (see [Audio pipeline](audio-pipeline.md#temporary-files)).

## Cross-cutting decisions

### No runtime Python dependencies

The package has `dependencies = []`. It uses the standard library plus the
`ffmpeg` and `ffprobe` executables.

- Reason: easy install (`uv tool install`, pipx, containers), small attack
  surface, no dependency churn for a small tool.
- Cost: some code we would otherwise import (multipart upload, retry logic,
  `.env` parsing) is written by hand and must be tested.
- Revisit if hand-written code grows large or fragile, for example if we need
  HTTP/2, streaming uploads or a full config library. Dev dependencies (pytest,
  ruff, mypy, ...) are not restricted.

### ffmpeg as an external tool

Media work is done by calling `ffmpeg`/`ffprobe` as subprocesses rather than
through Python bindings.

- Reason: ffmpeg handles practically every format, the CLI is stable, and
  subprocess calls need no compiled dependencies.
- All calls go through one wrapper in `ovox.audio` (`run_ffmpeg`) that sets a
  timeout and captures stderr. This gives one place for timeouts and error
  messages, and one place to fake ffmpeg in unit tests.
- `ovox` checks that both tools are on `PATH` and reports a clear error if not.
- Arguments are passed as a list (never through a shell) and input paths are
  resolved to absolute paths, so names with spaces or a leading `-` are safe.

### Configuration

Settings come from environment variables, optionally loaded from a `.env` file
(see `.env.example`). Environment variables win over `.env`. The API key is never
printed or logged. `ovox check` makes a cheap authenticated call to the STT
endpoint and reports OK or a clear failure, so configuration problems show up
before a long job starts.

### Errors

- Each stage raises its own exception type with a message a user can act on
  (for example `InputValidationError`, `AudioExtractionError`). ffmpeg errors
  include the last lines of ffmpeg's stderr.
- The CLI turns these into a short message and a non-zero exit code; tracebacks
  are for unexpected errors only.
- Validation catches bad input early so later stages can assume a readable file
  with an audio stream.

### Tooling

Python 3.14, `uv`, hatchling, ruff (lint and format) and `mypy --strict` on
`src/ovox`. Code must pass all three and the test suite.

## Later: server mode

Not designed yet. Expected to reuse the pipeline unchanged and add job storage,
a queue, authentication and a results API. Open questions to settle then: where
temp files live, how to stream progress, per-user rate limits, and whether
chunks of one job are transcribed in parallel across workers.
