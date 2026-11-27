# Testing

Status: current for the first beta. Last updated 2026-11.

## Unit and integration tests are split by directory

```
tests/
  unit/          fast, no ffmpeg, no network, no real media
  integration/   real ffmpeg/ffprobe on generated media
```

- The directory is the split; there is no `integration` pytest marker.
  Run only unit tests with `pytest tests/unit`.
- pytest runs with `--import-mode=importlib`, so the same file name can exist in
  both directories (for example `unit/test_chunking.py` and
  `integration/test_chunking.py`) without `__init__.py` files.

Configuration in `pyproject.toml`

## Unit tests

Code is structured so that the logic is testable without ffmpeg:

- **Pure functions for the decisions.** Chunk planning (`plan_chunks`), limit
  calculation and parsing ffmpeg output (`parse_silencedetect`) take plain
  values and return plain values.
- **One seam for ffmpeg.** All ffmpeg and ffprobe calls go through
  `ovox.audio.run_ffmpeg`. Unit tests replace it with a fake that records the
  command and returns a chosen exit code, stdout and stderr, so they can check
  the exact command, the timeout, and error handling.
- **One seam for HTTP.** The function that sends one request to the STT endpoint
  is passed in (or replaced), so retry, concurrency and ordering logic is tested
  with fakes, and `sleep` is replaced so retry tests run instantly.

Besides example-based tests, chunk planning has a seeded randomized test that
checks its rules on many random inputs: chunks cover the whole file, cuts line
up with the overlaps, no chunk exceeds the limit, there is no tiny last chunk,
and the number of chunks stays reasonable. The seed is fixed, so failures are
reproducible.

## Integration tests

Integration tests run real `ffmpeg`/`ffprobe` and check what the unit tests
cannot: that the commands work, that output has the promised format, and that
real silence detection gives usable cut points.

### Test media is generated, not committed

The one exception: `tests/data/audio/` holds one or two short committed
speech clips (a few seconds, well under 500 kB, public-domain or own
recording) with a sibling `.txt` holding the expected transcript. They are
the fixtures the `live` tests below check against.

A session-scoped `media_dir` fixture in `tests/integration/conftest.py` creates
all media with ffmpeg's built-in sources (`sine`, `aevalsrc`, `testsrc`) through
one helper, `_media`. Each file is described by its parts, for example:

```python
_media(d / "tone_with_silence.wav", tone, ("silence", 1), tone)
_media(d / "video_with_audio.mp4", tone, video=True)
```

- Generated audio is **stereo 44.1 kHz**, so the cutting tests prove ovox
  really converts to mono 16 kHz (see `tests/integration/test_audio.py`).
- Tones and exact silences make timings predictable, so tests can assert where
  silences and cuts land (with small tolerances for MP3 encoder padding).
- Invalid inputs (empty file, corrupt file, text file) are written directly.

Generation takes a couple of seconds once per test session. Files land in
pytest's temp area (`/tmp/pytest-of-<user>/pytest-<n>/`), which pytest cleans up
itself, keeping the last few runs.

### Real recordings (optional)

Tests that need real recordings (for example speech, unusual codecs) are skipped
unless `OVOX_TEST_MEDIA_DIR` points to a folder of media files. Long or private
recordings are not committed, for size and licensing reasons.

Two kinds of tests use real recordings:

- **`local_media`** (no endpoint): every media file under
  `$OVOX_TEST_MEDIA_DIR` is probed and must contain an audio stream
  (`tests/integration/test_media.py`). Run with
  `OVOX_TEST_MEDIA_DIR=./tests/data pytest -m local_media`.
- **`live`** (real STT endpoint): `tests/integration/test_live_transcribe.py`
  transcribes the committed short speech clips with the endpoint and key from
  the environment or `.env`, and checks the transcript against the committed
  expected `.txt` (word-based, so punctuation or case differences are fine).
  With `$OVOX_TEST_MEDIA_DIR` set it also transcribes your own files — the
  closest thing to a full test of real use — but only files that have a
  sibling `.txt` with the expected transcript:

  ```bash
  pytest -m live                                        # committed clips
  OVOX_TEST_MEDIA_DIR=~/media pytest -m "live and local_media"
  ```

  These cost real API calls, so the marker is excluded from the default run
  (`-m "not live"` in `addopts`, pyproject.toml); CI never runs them.

### The STT endpoint

No test calls a real STT endpoint by default. The `ovox check` logic is unit
tested by replacing `httpx.AsyncClient.get` with a fake that returns canned
responses: success, 401/404/500, timeout and connection errors. The tests
assert the structured fields (`success`, `status_code`, `endpoint`, `model`,
`duration_ms`, `error`) rather than message text, and include a 200-with-HTML
response that must be reported as a failure (a gateway answering a wrong URL
with a web page, see [Checking the
configuration](transcription.md#checking-the-configuration)). The CLI layer is
tested the same way with a mocked client and checks the exit code and the
`--json` output.

Once chunk-level transcription exists, the request code should be tested
against a small local fake server (standard library `http.server`) that can
answer with success, 429 with `Retry-After`, 401 or invalid JSON. A manual or
opt-in check against a real endpoint is `ovox check`.

## Quality gates

Every change passes `pytest`, `ruff check`, `ruff format --check` and
`mypy --strict` on `src/ovox`.
