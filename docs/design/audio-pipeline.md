# Audio pipeline: validation, extraction and chunking

Status: current for the first beta. Last updated 2026-11.

This covers everything between the user's input file and the audio chunks sent
to the STT endpoint. See the [overview](overview.md) for the full pipeline.

## Input validation (`ovox.media`)

Before any processing, the input must:

1. exist and be a regular file,
2. be readable and non-empty,
3. have at least one audio stream according to `ffprobe`.

Each failure has its own clear message: missing file, unreadable file, empty
file, unsupported or corrupt media, and video without audio. Validation runs
first so that a bad input fails in under a second instead of after extraction.

## Audio extraction (`ovox.audio`)

The first audio stream of the input is converted to a **constant-bitrate (CBR)
MP3**. The sample rate, channel count and bitrate are deployment settings in
the "Audio processing" block of `.env`; the built-in defaults are **mono,
16 kHz, 64 kbps** (about 28.8 MB per hour of audio).

| Setting | Env var | Default | Reason |
| --- | --- | --- | --- |
| First audio stream only (`-map 0:a:0`) | — | — | Ignores video, subtitles and cover art. Files with several audio tracks (for example several languages) use the first; choosing a track is a possible later option. |
| Channels | `AUDIO_CHANNELS` | 1 (mono) | Speech recognition does not use stereo; halves the data. |
| Sample rate | `AUDIO_SAMPLE_RATE` | 16000 | The native sample rate of common STT models (for example Whisper). Higher rates add size without improving accuracy. Also selects which MP3 bitrates are valid. |
| MP3 bitrate (CBR) | `AUDIO_BITRATE_KBPS` | 64 | Widely accepted by STT endpoints. A constant bitrate makes file size predictable from duration, which chunking relies on. 64 kbps gives about 28.8 MB/hour. |

The sample rate and bitrate are validated together at startup: MP3 only allows
certain bitrates per sample rate (16/22.05/24 kHz: 8–160 kbps; 32/44.1/48 kHz:
32–320 kbps). An out-of-table value is rejected with a clear error, because
ffmpeg would otherwise quietly pick a different bitrate and the size
calculations would be wrong without anyone noticing. The bitrate is expressed in
kilobits per second in the environment (`-b:a 64k` is how people and ffmpeg
state it) and converted to bits per second internally.

Alternatives considered: WAV/PCM (lossless but about 115 MB per hour at 16 kHz
mono, so many more chunks), FLAC (smaller than WAV but not accepted everywhere),
Opus (smallest, but endpoint support varies). MP3 is the safe default; the
format could become configurable per endpoint.

### Temporary files

Planning gets one private temp directory (`workdir()`, based on
`tempfile.TemporaryDirectory`) holding the extracted audio (used for silence
detection). It is removed when the plan is built, also on errors and Ctrl-C.
Chunk files go straight to the output directory. For SIGTERM, the CLI installs
a handler that raises `SystemExit` so the same cleanup runs. Nothing is written
next to the user's input file.

## Chunking (`ovox.chunking`)

STT endpoints limit upload size and often audio duration, so long audio is split
into chunks. The goals, in order:

1. Every chunk fits the endpoint's limits.
2. Cuts land in pauses, so no word is split.
3. No more chunks than needed (each chunk is a request and loses context at its
   edges).

### Chunk length limit

The target endpoint's limits are kept together as per-target settings
(`CHUNK_MAX_SIZE_BYTES`, `CHUNK_MAX_DURATION_S` and `CHUNK_SAFETY` in `.env`),
so pointing ovox at a different endpoint is a config change, not a code change.
`max_chunk_seconds` turns the size limit into seconds using the *configured*
MP3 bitrate (the same `AUDIO_BITRATE_KBPS` the audio is encoded at), takes the
smaller of that and the duration limit, and applies a safety margin:

```
max_len = min(max_bytes * 8 / bitrate, max_seconds) * safety
```

Defaults: 25 MiB, 600 s and a safety factor of 0.8, which gives 480 s chunks
(the duration limit wins). Point these at the real endpoint's limits.

The extraction and the planning must read the *same* bitrate, so the size
estimate matches the file actually written. That is why the bitrate is a single
config value passed to both, not a constant in each module.

The format you send matters a lot: mono 16 kHz MP3 at 64 kbps fits about
28.8 MB per hour, so several times more minutes sit under a given size limit
than uncompressed stereo would. The safety margin (0.8 by default,
`CHUNK_SAFETY`) leaves room for container overhead and variable-bitrate drift,
because estimated sizes are based on the CBR rate and the endpoint's real limit
is hard.

An upload-*time* limit will be added later as another input to the same
maximum-length calculation (based on an assumed upload speed), without changing
how splitting works.

### Where to cut

Planning is a pure function (`plan_chunks`) over the duration and a list of
silences, so it is fully unit tested without ffmpeg. Silences come from
ffmpeg's `silencedetect` filter (`detect_silence`; threshold -30 dB, minimum
0.2 s).

For each chunk starting at `start`, with planned end `start + max_len`:

1. **Search window.** Look for silence between `planned_end - search_window`
   (default 30 s) and `planned_end`, but never before `start + max_len / 2`.
   The lower bound stops short pauses from producing many tiny chunks.
2. **Clip silences to the window.** A long pause that starts before the window
   or runs past the planned end still counts, clipped to the window.
3. **Cut on silence.** Choose the longest clipped silence (at least 0.2 s) and
   cut at its middle. The next chunk starts exactly there, with no overlap,
   because nothing is cut mid-word.
4. **Hard cut otherwise.** With no usable silence, cut at the planned end and
   start the next chunk `overlap` seconds earlier (default 1 s), so a word split
   by the cut is heard whole at least once. The merge step removes the
   duplicated words (see [Transcription](transcription.md#merging-chunk-texts)).
5. **No sliver at the end.** If the cut would leave less than 1 s
   (`MIN_TAIL_S`), cut earlier instead. Very short audio is often rejected by
   endpoints or produces invented text.

Each chunk records `overlap_before`: how many seconds at its start repeat the
previous chunk (0 after a silence cut).

Alternatives considered:

- **Fixed-length chunks with overlap everywhere**: simplest, but every boundary
  splits words and every boundary needs deduplication.
- **Voice activity detection (VAD) models**: better pause detection, but adds a
  heavy dependency. `silencedetect` is good enough for speech recordings.

### Cutting the audio

Each chunk is produced by **decoding the planned range from the original source
file and re-encoding** it to the upload format (mono, 16 kHz, CBR MP3). The cut
is positioned with an input seek (`-ss` before `-i`) plus an exact `-t`
duration, so it lands at the planned sample on every ffmpeg version.

This is deliberately **not** a stream copy (`-c copy`) of the extracted MP3.
Stream copy can only cut on whole MP3 frames, and because an MP3's trailing
bit-reservoir and encoder padding hold no decodable samples once you cut into
them, a copy near the end of the file decodes to *less* audio than its header
claims (measured ~0.25 s short at 16 kHz). That loss varied between ffmpeg
versions and would have shifted transcript timestamps and broken resume.
Re-encoding removes the version dependence; 16 kHz mono MP3 is fast, and
cutting from the *source* (not the intermediate MP3) keeps the chunk to a
single generation of lossy compression.

A chunk's real length is measured by **decoding it and counting samples**
(`decoded_duration`), not by ffprobe's reported (header) duration: for an MP3
the header includes the padding above, and how much it overstates differs
between ffmpeg versions. The measured length is recorded in the manifest as
`actual_duration` so the manifest matches what is actually on disk.

Each chunk file is checked against the size limit. Audio shorter than `max_len`
is sent as one chunk without cutting.

## Known limitations and possible improvements

- Chunks that are entirely silence are still sent; skipping them would save
  requests and avoid invented text.
- Music or constant background noise may contain no silence below -30 dB, so
  cuts become hard cuts with overlap. The threshold could adapt to the
  recording's loudness.
- Only the first audio track is used.
