# Transcription: STT endpoint, retries and merging

Status: current for the first beta. Last updated 2026-11.

This covers sending chunks to the speech-to-text (STT) endpoint and turning the
results into one transcript. See the [audio pipeline](audio-pipeline.md) for how
chunks are made.

## Endpoint contract

ovox targets an **OpenAI-compatible transcription endpoint**
(`POST .../audio/transcriptions`):

- multipart form upload with `file`, `model` and optionally `language`;
- bearer-token authentication;
- JSON response with a `text` field.

Why: this API shape is supported by OpenAI, many hosted providers, and
self-hosted servers for Whisper-style and custom models. Targeting it lets users
switch between hosted and their own models by changing the URL, model name and
key, which is the core of "custom transcription models".

Everything endpoint-specific lives in one function that sends a single request.
Supporting a different API later means replacing that function, not the
retry, concurrency or merge logic.

The multipart upload is built with the standard library (see "No runtime
dependencies" in the [overview](overview.md)).

## Checking the configuration

`ovox check` makes one cheap authenticated call to the endpoint and reports OK
or a clear failure (bad key, wrong URL, endpoint down). Users run it before long
jobs; transcription errors caused by authentication point back to it.

The check is a `GET` of the model list (`.../models`), derived from the
configured `STT_ENDPOINT` by replacing the `/audio/transcriptions` path. It is
sent with the same bearer key, so it also catches a bad key.

A success requires a 2xx response **and** a JSON body. A 2xx with a non-JSON
body is a failure (`error: unexpected_response`), not a success: gateways in
front of STT servers (for example Open WebUI) answer *any* unknown URL with 200
and a web page, so a wrong `STT_ENDPOINT` would otherwise look healthy.

The result always includes the configured `endpoint` and `model` and the call's
`duration_ms`, so failures are debuggable without re-running with more logging.
`status_code` is `null` when no HTTP response was received (the connection
failed or timed out). On failure, `error` is a short stable type that scripts
can match on instead of message text: `connection_error`, `timeout`,
`unauthorized`, `not_found`, `http_error` (any other 4xx/5xx) or
`unexpected_response`.

The exit code is the contract: 0 on success, non-zero on failure. Plain text
output shows the outcome, the error type (on failure), the endpoint, the
configured model and the duration; `--json` prints the result as JSON.

## Retries and backoff

Each chunk request is retried on transient failures:

- **Retried:** network errors, timeouts, and HTTP 408, 409, 425, 429, 500, 502,
  503 and 504.
- **Not retried:** other 4xx errors. 401 and 403 fail immediately with a hint to
  run `ovox check`. Invalid JSON or a missing `text` field also fail
  immediately.
- **Backoff:** exponential with full jitter, capped. Jitter spreads retries from
  parallel requests so they do not hit the endpoint at the same moment.
- **`Retry-After`:** when the server sends it (in seconds), it is used instead of
  the computed delay, capped to a sensible maximum.
- After the last attempt the chunk fails with a message naming the chunk and the
  last error.

Retry count, timeouts and backoff limits are settings with defaults in code.

## Concurrency

Chunks are transcribed in parallel with a bounded thread pool (a small default
of a few requests at a time, configurable).

- Threads rather than asyncio: the work is waiting on HTTP, the standard
  library's HTTP client is synchronous, and a thread pool keeps the code simple.
- The bound protects the endpoint and the user's rate limits.
- Results are returned in chunk order regardless of completion order.
- **Fail fast:** the first chunk that fails for good cancels chunks not yet
  started and the whole run fails. A partial transcript with gaps is worse than
  a clear error in the beta. Resuming or keeping partial results can come later
  (likely together with server mode).
- Progress is reported through a callback (chunks done / total), so the CLI can
  show it without the pipeline printing.

## Merging chunk texts

Chunk texts are joined in order with a space.

- After a **silence cut** (`overlap_before == 0`) nothing is repeated, so texts
  are joined as they are.
- After a **hard cut** the next chunk repeats the last `overlap` seconds of
  audio. The merge looks for the longest run of words (up to a small limit) that
  ends the previous text and starts the next one, comparing words in lower case
  without punctuation, and drops that run from the next text.

This is a text-only heuristic. It can miss a duplicate if the model transcribes
the overlap differently in the two chunks, which leaves a repeated word or two.
Since most cuts land on silence, this is rare.

Possible improvement: request word or segment timestamps (`verbose_json`) and
drop words that start inside the overlap. That makes the merge exact and also
enables subtitle output (SRT/VTT).

## Open questions

- Language: pass the user's choice through, or rely on the model's detection
  for each chunk? Per-chunk detection can disagree between chunks.
- Prompting the model with the end of the previous chunk's text (supported by
  some endpoints) could improve continuity, but forces chunks to run in order.
