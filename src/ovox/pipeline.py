"""The direct-mode transcription pipeline.

:func:`transcribe_file` is the end-to-end core: validate input, plan the
chunks, cut them into the work folder, send the missing ones to the STT
endpoint (bounded concurrency, retries), and stitch the replies into the
transcript files. The CLI (and, later, a server job layer) are thin wrappers
around this function: progress is reported through a callback and failures
come back as exceptions, so nothing here prints or exits except a resume
note and an interruption note on stderr.

Work folder
-----------
Chunks, their manifest and the per-chunk transcripts live together in one
folder, so a run can resume from it:

- ``.ovox-<stem>-<hash>/`` next to the input file by default, or ``--workdir``.
- ``ovox-chunk-manifest.json``: the plan, written once and never changed
  afterwards.
- the chunk audio files: exactly what gets uploaded.
- ``chunk_NNN.json``: one stored reply per chunk; **its existence is the
  transcription progress**, checked per chunk, so the manifest stays stable.

Resume means: read the manifest, check the source and the settings match, and
transcribe the chunks without a transcript file.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ovox import plan as plan_mod
from ovox import stitch
from ovox.api import STTClient, SttError, interrupted, transcribe_chunk
from ovox.config import AudioConfig, ChunkConfig, SttConfig, TranscribeConfig
from ovox.media import MediaError, MediaInfo, validate_input
from ovox.plan import ChunkEntry, ChunkPlan

# The settings a chunk's reply was asked with; stored in each chunk_NNN.json so
# a later run can tell whether the deployment changed.
TRANSCRIPT_SCHEMA_VERSION = 1


class PipelineError(Exception):
    """A transcription run failed; the message is user-actionable."""


class WorkdirError(PipelineError):
    """The work folder is missing, incomplete or does not match this run."""


def _setting_mismatch(kind: str, expected: object, actual: object) -> str:
    return (
        f"settings mismatch: the work folder was created with {kind}={expected!r}, "
        f"but the current setting is {kind}={actual!r}. Chunks made with different "
        f"settings would not match; delete the work folder to start over."
    )


# --- work folder ---------------------------------------------------------------


def default_workdir(source: Path) -> Path:
    """The default work folder for *source*: ``.ovox-<stem>-<hash>`` next to it.

    The 8-character hash of the resolved path keeps two files with the same
    name in different folders apart, and makes the folder name stable no
    matter which relative path the user runs from.
    """
    resolved = source.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return source.parent / f".ovox-{resolved.stem}-{digest}"


def chunk_plan_path(workdir: Path) -> Path:
    return Path(workdir) / "ovox-chunk-manifest.json"


def _clean_workdir(workdir: Path, plan: ChunkPlan) -> None:
    """Remove ovox's own files from *workdir*: the manifest, the chunk audio of
    *plan* and the per-chunk result files (and any stray temp files).

    Used to start fresh when a stored work folder does not match the current
    file or settings. Only files ovox itself wrote are removed; anything else
    in the directory (e.g. in a user-chosen ``--workdir``) is left alone.
    """
    for name in [e.name for e in plan.chunks] + ["ovox-chunk-manifest.json"]:
        (workdir / name).unlink(missing_ok=True)
    for pattern in ("chunk_*.json", "chunk_*.json.tmp"):
        for old in workdir.glob(pattern):
            old.unlink(missing_ok=True)


def plan_and_cut(
    info: MediaInfo,
    chunk_config: ChunkConfig,
    audio_config: AudioConfig,
    stt: SttConfig,
    workdir_path: Path,
) -> tuple[ChunkPlan, bool]:
    """Plan the chunks and cut them into *workdir_path*, or reuse the stored
    plan when it is still valid.

    A previous run's work folder may be here. Its chunks (and any finished
    transcripts) are still valid when the source and the settings have not
    changed, so a plain re-run is also a resume: read the manifest, check it,
    and skip the re-plan and re-cut. Otherwise start fresh, removing only
    ovox's own files (so stale transcripts for a different plan cannot be
    reused). Used by both the real run and the ``--dry-run`` local stage, so
    the two always agree on what a work folder contains.

    Returns:
        The plan, and whether a stored plan was reused (so the caller can
        tell a fresh start from a re-run that picks the folder up).
    """
    candidate: ChunkPlan | None = None
    if chunk_plan_path(workdir_path).exists():
        try:
            stored = load_plan(workdir_path)
            check_resume_matches(stored, chunk_config, audio_config, stt)
            if not any(not (workdir_path / e.name).exists() for e in stored.chunks):
                candidate = stored
        except WorkdirError:
            pass
    if candidate is None:
        plan = plan_mod.build_plan(info, chunk_config, audio_config, model=stt.model)
        if chunk_plan_path(workdir_path).exists():
            _clean_workdir(workdir_path, plan)
        workdir_path.mkdir(parents=True, exist_ok=True)
        plan_mod.write_chunks(plan, workdir_path, audio_config)
        chunk_plan_path(workdir_path).write_text(plan_mod.plan_manifest_text(plan))
        return plan, False
    return candidate, True


def load_plan(workdir: Path) -> ChunkPlan:
    """Read the manifest a previous run wrote into *workdir*."""
    path = chunk_plan_path(workdir)
    if not path.exists():
        raise WorkdirError(
            f"No chunk manifest in {path}. Run 'ovox transcribe' (without --resume) "
            f"to create the work folder first."
        )
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkdirError(f"The chunk manifest is not valid JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise WorkdirError(f"The chunk manifest is malformed: {path}")
    try:
        return plan_mod.plan_from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkdirError(f"The chunk manifest is malformed: {path} ({exc})") from exc


def _settings_value(plan: ChunkPlan, name: str) -> object:
    setting = plan.settings.get(name)
    return setting.value if setting is not None else None


def check_resume_matches(
    plan: ChunkPlan,
    chunk_config: ChunkConfig,
    audio_config: AudioConfig,
    stt: SttConfig,
) -> None:
    """Check the manifest against the source file and the current settings.

    A mismatch means the stored chunks (and their transcripts) were made for a
    different file or a different deployment, and reusing them would produce a
    wrong transcript. The error tells the user what to do (delete the work
    folder) rather than silently re-cutting or mismerging work.

    Raises:
        WorkdirError: the source is gone or a setting that shapes the chunks
            (limits, upload format, split method, STT model) changed.
    """
    source_raw = plan.input.get("path") or plan.input.get("file")
    if not source_raw:
        raise WorkdirError(
            "The chunk manifest does not record its source file; delete the work folder."
        )
    source = Path(str(source_raw)).resolve()
    if not source.exists():
        raise WorkdirError(
            f"The source file {source} no longer exists. If the file was moved or "
            f"renamed, delete the work folder and transcribe the new file."
        )
    if plan.input.get("model") not in (None, stt.model):
        raise WorkdirError(_setting_mismatch("STT model", plan.input.get("model"), stt.model))
    if plan.split_method not in ("silence", "fixed"):
        raise WorkdirError(f"Unknown split method in the manifest: {plan.split_method!r}")

    # The stored chunks were cut from the source as it was then. Any change to
    # the file (a re-record, an edit) invalidates them: the chunk audio and
    # the transcripts would not correspond to the audio being transcribed.
    recorded_size = plan.input.get("size")
    recorded_mtime = plan.input.get("mtime")
    if recorded_size is not None or recorded_mtime is not None:
        current = source.stat()
        if (recorded_size is not None and current.st_size != recorded_size) or (
            recorded_mtime is not None and current.st_mtime != recorded_mtime
        ):
            raise WorkdirError(
                f"The source file {source} has changed since the work folder was "
                f"created (its size or modification time no longer matches the "
                f"manifest). Delete the work folder and transcribe again."
            )

    expected: dict[str, object] = {
        "max_chunk_size_bytes": chunk_config.max_chunk_size_bytes,
        "max_chunk_duration_s": chunk_config.max_chunk_duration_s,
        "safety": chunk_config.safety,
        "sample_rate": audio_config.sample_rate,
        "channels": audio_config.channels,
        "bitrate_kbps": audio_config.bitrate_bps // 1000,
    }
    for name, value in expected.items():
        actual = _settings_value(plan, name)
        if actual is not None and actual != value:
            raise WorkdirError(_setting_mismatch(name, actual, value))


# --- the pipeline ----------------------------------------------------------------


def transcribe_file(
    input_path: str | Path,
    *,
    chunk_config: ChunkConfig,
    audio_config: AudioConfig,
    stt: SttConfig,
    transcribe_config: TranscribeConfig,
    workdir: str | Path | None = None,
    output: str | Path | None = None,
    resume: bool = False,
    client: STTClient | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Transcribe *input_path* end to end and write the transcript files.

    Stages: validate -> plan -> cut -> transcribe (the missing chunks only)
    -> merge. With ``resume`` a previous run's work folder is reused (its
    manifest and chunks, after a source/settings check); otherwise the work
    folder is (re)created and its chunks cut, which also serves as the
    ``--dry-run`` local stage (a dry run stops before the STT stage).

    The transcript is written as ``<stem>.txt`` next to the input, or to
    *output* when given. The beta is text-only, so no segments JSON is
    written: if the endpoint did return segment data, it is kept untouched in
    the work folder's per-chunk result files until timestamp output exists.

    The target transcript is not overwritten silently: a run that starts
    fresh (no matching work folder) fails if the transcript already exists
    (before any STT request is sent) and the message offers deleting it or
    choosing another path with ``-o``. A run that reuses a matching work
    folder (``resume``, or a plain re-run that picks the folder up)
    regenerates the transcript that work folder produced, so it overwrites
    it.

    Args:
        input_path: The media file to transcribe.
        chunk_config, audio_config, stt, transcribe_config: Resolved settings.
        workdir: Where chunks, manifest and per-chunk transcripts live.
            Defaults to a hidden folder next to the input.
        output: Where to write the ``.txt`` transcript.
        resume: Reuse an existing work folder instead of planning and cutting.
        client: An open STTClient; one is created and closed when not given.
        on_progress: Called with ``(done, total)`` after each chunk completes.

    Returns:
        A result dict: ``text_file``, ``chunks_transcribed`` (STT calls made
        by this run), ``chunks_total``, ``resumed`` (the work folder was
        reused), ``workdir`` and ``timings`` (per-stage seconds: probe, plan,
        cut, stt, merge).

    Raises:
        MediaError: the input is not a usable media file (or ffmpeg is missing).
        WorkdirError: a resume could not be matched to the stored work.
        PipelineError: the run failed; the message is actionable (auth
            problems point at ``ovox check``, and a partial run reports how
            many chunks are already done, so re-running resumes it).
    """
    source = Path(input_path)
    workdir_path = Path(workdir) if workdir is not None else default_workdir(source)
    base = Path(output) if output is not None else source
    text_path = base.with_name(f"{base.stem}.txt")
    stages: dict[str, float] = {}

    to_do: list[tuple[int, ChunkEntry, Path]] = []
    text = ""
    reused = False
    close_at_end: STTClient | None = None
    if client is None:
        client = STTClient(config=stt)
        close_at_end = client
    try:
        # -- validate -----------------------------------------------------
        t0 = time.monotonic()
        info = validate_input(source)
        stages["probe"] = time.monotonic() - t0

        # -- plan + cut (or reuse the stored plan) --------------------------
        t0 = time.monotonic()
        plan: ChunkPlan
        if resume:
            reused = True
            plan = load_plan(workdir_path)
            check_resume_matches(plan, chunk_config, audio_config, stt)
            missing_audio = [e for e in plan.chunks if not (workdir_path / e.name).exists()]
            if missing_audio:
                names = ", ".join(e.name for e in missing_audio[:5])
                more = f" (+{len(missing_audio) - 5} more)" if len(missing_audio) > 5 else ""
                raise WorkdirError(
                    f"The work folder {workdir_path} is missing chunk file(s): "
                    f"{names}{more}. Delete the work folder and transcribe again."
                )
        else:
            plan, reused = plan_and_cut(info, chunk_config, audio_config, stt, workdir_path)
        stages["cut"] = time.monotonic() - t0

        # -- output check ----------------------------------------------------
        # A run that starts fresh (no matching work folder) never overwrites
        # an existing transcript: that file may be the output of an earlier,
        # different run that the user still wants, or something else entirely.
        # A run that reuses a matching work folder (a resume, or a plain
        # re-run that picks the folder up) regenerates the transcript that
        # work folder produced, so it may overwrite it.
        if not reused and text_path.exists():
            raise WorkdirError(
                f"Transcript file {text_path} already exists. Delete it, or choose "
                f"another path with -o."
            )

        # -- transcribe the chunks without a transcript file -----------------
        t0 = time.monotonic()
        replies = stitch.load_replies(workdir_path, plan)
        to_do = stitch.pending_chunks(workdir_path, plan, replies)
        done = len(plan.chunks) - len(to_do)
        if done:
            print(
                f"[ovox] resuming: {done}/{len(plan.chunks)} chunks already transcribed",
                file=sys.stderr,
            )
        if to_do:
            _run_transcription(
                workdir_path,
                plan,
                to_do,
                client,
                transcribe_config,
                on_progress=on_progress,
                base_done=done,
            )
        stages["stt"] = time.monotonic() - t0

        # -- merge -----------------------------------------------------------
        t0 = time.monotonic()
        payloads = stitch.load_replies(workdir_path, plan)
        missing = [e.name for e, p in zip(plan.chunks, payloads, strict=True) if p is None]
        if missing:
            raise PipelineError(
                f"Transcription stopped with {len(missing)} of {len(plan.chunks)} chunks "
                f"untranscribed (first: {missing[0]}). The {len(plan.chunks) - len(missing)} "
                f"finished chunks are kept in {workdir_path}; run the same command again "
                f"to resume."
            )
        solid = stitch.replies_from_payloads([p for p in payloads if p is not None])
        text = stitch.build_text(plan, solid)
        stages["merge"] = time.monotonic() - t0
    except SttError as exc:
        raise PipelineError(f"STT endpoint error: {exc}") from exc
    except WorkdirError:
        raise
    except MediaError:
        raise
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Transcription failed: {exc}") from exc
    finally:
        if close_at_end is not None:
            close_at_end.close_http()

    # -- outputs -----------------------------------------------------------
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text + "\n", encoding="utf-8")

    return {
        "text_file": text_path,
        "chunks_transcribed": len(to_do),
        "chunks_total": len(plan.chunks),
        # The work folder was reused (a --resume, or a plain re-run that
        # picked the folder up), so the cut stage skipped.
        "resumed": reused,
        "workdir": str(workdir_path),
        "timings": {name: round(seconds, 3) for name, seconds in stages.items()},
    }


def prepare_for_dry_run(
    input_path: str | Path,
    *,
    chunk_config: ChunkConfig,
    audio_config: AudioConfig,
    stt: SttConfig,
    workdir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the local stages (validate, plan, cut) for ``--dry-run``: no STT.

    Cuts the chunks into the work folder and writes the manifest, so a later
    real run resumes from it. Returns the plan as a dict plus the progress
    count (how many chunks already have a transcript file, on a resume).

    Raises:
        MediaError, WorkdirError, PipelineError: as in :func:`transcribe_file`.
    """
    source = Path(input_path)
    workdir_path = Path(workdir) if workdir is not None else default_workdir(source)

    t0 = time.monotonic()
    info = validate_input(source)
    probe = time.monotonic() - t0

    t0 = time.monotonic()
    if resume:
        plan = load_plan(workdir_path)
        check_resume_matches(plan, chunk_config, audio_config, stt)
        reused = True
    else:
        plan, reused = plan_and_cut(info, chunk_config, audio_config, stt, workdir_path)
    cut = time.monotonic() - t0

    replies = stitch.load_replies(workdir_path, plan)
    pending = stitch.pending_chunks(workdir_path, plan, replies)
    return {
        "plan": plan.to_dict(),
        "workdir": str(workdir_path),
        "chunks_transcribed": len(plan.chunks) - len(pending),
        "chunks_pending": len(pending),
        "resumed": reused,
        "timings": {"probe": round(probe, 3), "cut": round(cut, 3)},
    }


# --- bounded-concurrency transcription of the pending chunks ----------------------


def _run_transcription(
    workdir: Path,
    plan: ChunkPlan,
    to_do: Sequence[tuple[int, ChunkEntry, Path]],
    client: STTClient,
    config: TranscribeConfig,
    *,
    on_progress: Callable[[int, int], None] | None,
    base_done: int,
) -> None:
    """Send *to_do* chunks to the endpoint with bounded concurrency.

    The first chunk that fails for good sets a stop event, so no chunk not yet
    started is sent and no retry goes out; a request already in flight is let
    run to completion (it is cut short only by the shutdown bound in
    :mod:`ovox.api`). Chunks that finished stay on disk (each result is
    written ``.tmp`` then atomically renamed), so a later run resumes from
    them.

    An interrupt signal (Ctrl-C, SIGTERM) stops the run the same way — it is
    folded into the same "stop requested" check (see :mod:`ovox.api`). Because
    every result is written atomically, any chunk whose result file exists is
    complete and will not be re-transcribed by a resume.

    Raises:
        PipelineError: a chunk could not be transcribed (or the run was
            interrupted); the message names the chunk and says the work
            folder, so a resume is one command away.
    """
    stop = threading.Event()
    failed: list[SttError] = []
    total = len(plan.chunks)
    state = {"done": base_done}
    lock = threading.Lock()

    # True when the run must stop: another chunk failed for good (the stop
    # event), or an interrupt signal arrived (see ovox.api.interrupted).
    def _stop_requested() -> bool:
        return stop.is_set() or interrupted()

    def _work(item: tuple[int, ChunkEntry, Path]) -> None:
        _index, entry, path = item
        if _stop_requested():
            return
        try:
            reply = transcribe_chunk(client, path, entry.name, config, stop_event=stop)
        except SttError as exc:
            if not _stop_requested():
                stop.set()
                failed.append(exc)
            return
        payload = {
            "schema": TRANSCRIPT_SCHEMA_VERSION,
            "model": client.model,
            "endpoint": client.endpoint,
            "reply": reply,
        }
        stitch.write_chunk_transcript(workdir, entry.name, payload)
        with lock:
            state["done"] += 1
        if on_progress is not None:
            on_progress(state["done"], total)

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as pool:
        futures = [pool.submit(_work, item) for item in to_do]
        for future in futures:
            future.result()  # surfaces thread bugs; SttError is handled in _work

    if failed:
        first = failed[0]
        done_now = state["done"]
        raise PipelineError(
            f"Chunk {first.chunk_name} could not be transcribed: {first} "
            f"({done_now}/{total} chunks are done in {workdir}; "
            f"run the same command again to resume)."
        ) from first

    if _stop_requested():
        # An interrupt stopped the run (nothing else failed): the main
        # thread, waiting on the pool, is the one that unwinds, so this is
        # where the interruption is reported.
        print(
            f"Interrupted: {state['done']}/{total} chunks are done in {workdir}; "
            "run the same command again to resume.",
            file=sys.stderr,
        )
        raise PipelineError(
            f"Transcription was interrupted ({state['done']}/{total} chunks done "
            f"in {workdir}); run the same command again to resume."
        )
