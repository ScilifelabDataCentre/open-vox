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

The first audio stream of the input is converted to **mono, 16 kHz, 32 kbps
CBR MP3**.

| Choice | Reason |
| --- | --- |
| First audio stream only (`-map 0:a:0`) | Ignores video, subtitles and cover art. Files with several audio tracks (for example several languages) use the first; choosing a track is a possible later option. |
| Mono | Speech recognition does not use stereo; halves the data. |
| 16 kHz | The native sample rate of common STT models (for example Whisper). Higher rates add size without improving accuracy. |
| MP3, 32 kbps, constant bitrate | Widely accepted by STT endpoints and small (about 14 MB per hour). A constant bitrate makes file size predictable from duration, which chunking relies on. |

Alternatives considered: WAV/PCM (lossless but about 115 MB per hour at 16 kHz
mono, so many more chunks), FLAC (smaller than WAV but not accepted everywhere),
Opus (smallest, but endpoint support varies). MP3 is the safe default; the
format could become configurable per endpoint.

### Temporary files

Each run gets one private temp directory (`workdir()`, based on
`tempfile.TemporaryDirectory`) holding the extracted audio and the chunks. It is
removed when the run ends, also on errors and Ctrl-C. For SIGTERM, the CLI
installs a handler that raises `SystemExit` so the same cleanup runs. Nothing is
written next to the user's input file.

## Chunking (`ovox.chunking`)

STT endpoints limit upload size and often audio duration, so long audio is split
into chunks. The goals, in order:

1. Every chunk fits the endpoint's limits.
2. Cuts land in pauses, so no word is split.
3. No more chunks than needed (each chunk is a request and loses context at its
   edges).

### Chunk length limit

`max_chunk_seconds` turns the endpoint's size limit into seconds using the fixed
MP3 bitrate, takes the smaller of that and the duration limit, and applies a
safety margin:

```
max_len = min(max_bytes * 8 / bitrate, max_seconds) * safety
```

Defaults: 25 MiB, 600 s and a safety factor of 0.8, which gives 480 s chunks
(the duration limit wins). Point these at the real endpoint's limits.

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

Chunks are cut from the extracted MP3 with stream copy (`-c copy`), which is
fast and needs no re-encoding. MP3 frames are 36 ms at 16 kHz (576 samples), so cut points
are accurate to a frame, well within the overlap and silence margins. Each chunk
file is checked against the size limit. Audio shorter than `max_len` is sent as
one chunk without cutting.

## Known limitations and possible improvements

- Chunks that are entirely silence are still sent; skipping them would save
  requests and avoid invented text.
- Music or constant background noise may contain no silence below -30 dB, so
  cuts become hard cuts with overlap. The threshold could adapt to the
  recording's loudness.
- Only the first audio track is used.
