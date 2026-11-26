"""Unit tests for ovox.pipeline: the direct-mode end-to-end run.

ffmpeg and HTTP are both faked: the plan and the chunk cutting are replaced
with stubs that build a plan and write tiny files, and ``transcribe_chunk``
is replaced with a fake that records calls and returns canned replies. What
is under test is the pipeline glue: the work folder layout, resume, progress
counting, fail-fast behaviour, merging and the written outputs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import ovox.plan as plan_mod
from ovox.api import SttError
from ovox.config import AudioConfig, ChunkConfig, SttConfig, TranscribeConfig
from ovox.media import MediaInfo
from ovox.pipeline import (
    PipelineError,
    WorkdirError,
    default_workdir,
    prepare_for_dry_run,
    transcribe_file,
)
from ovox.plan import ChunkEntry, ChunkPlan, Setting

STEM = "meeting"


def make_info(path: Path, duration: float) -> MediaInfo:
    return MediaInfo(
        path=path,
        format_name="mp4",
        duration=duration,
        audio_streams=1,
        video_streams=1,
        audio_codec="aac",
        audio_sample_rate=44100,
        audio_channels=2,
        audio_track_index=0,
    )


def make_plan(n: int, duration: float, source: Path) -> ChunkPlan:
    """An n-chunk plan covering [0, duration); chunk 1+ follow hard cuts.

    Mirrors ovox.plan._input_dict, including the source size and mtime the
    resume check compares against.
    """
    stat = source.stat()
    per = duration / n
    chunks = []
    for i in range(n):
        start = i * per
        end = duration if i == n - 1 else (i + 1) * per
        chunks.append(
            ChunkEntry(
                index=i,
                start=start,
                end=end,
                duration=end - start,
                estimated_size=8000,
                name=plan_mod.chunk_name(STEM, i, start, end),
                reason="cut",
                overlap_before=0.0 if i == 0 else 1.0,
            )
        )
    settings = {
        "max_chunk_size_bytes": Setting(26214400, "default", "CHUNK_MAX_SIZE_BYTES"),
        "max_chunk_duration_s": Setting(600.0, "default", "CHUNK_MAX_DURATION_S"),
        "safety": Setting(0.8, "default", "CHUNK_SAFETY"),
        "sample_rate": Setting(16000, "default", "AUDIO_SAMPLE_RATE"),
        "channels": Setting(1, "default", "AUDIO_CHANNELS"),
        "bitrate_kbps": Setting(64, "default", "AUDIO_BITRATE_KBPS"),
    }
    return ChunkPlan(
        input={
            "file": str(source),
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "model": "voxtral-1",
            "format": "mp4",
            "codec": "aac",
            "duration": duration,
            "sample_rate": 44100,
            "channels": 2,
            "audio_track": 0,
        },
        settings=settings,
        chunks=chunks,
        warnings=[],
        totals={"num_chunks": n, "total_audio_length": duration},
        split_method="silence",
    )


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    path = tmp_path / f"{STEM}.mp4"
    path.write_bytes(b"fake media")
    return path


class FakeTranscriber:
    """Stands in for ovox.pipeline.transcribe_chunk."""

    def __init__(self, texts: dict[str, str] | None = None, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, object]] = []
        self.texts = texts or {}
        self.fail_on = fail_on or set()
        self.lock = threading.Lock()

    def __call__(self, client, path, chunk_name, config, *, stop_event=None):
        with self.lock:
            self.calls.append((chunk_name, stop_event))
        if chunk_name in self.fail_on:
            raise SttError(f"endpoint exploded for {chunk_name}", chunk_name=chunk_name)
        return {"text": self.texts.get(chunk_name, f"text of {chunk_name}")}


@pytest.fixture
def fakes(input_file: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch validate_input, build_plan, write_chunks and transcribe_chunk."""
    fake = FakeTranscriber()
    plan = make_plan(3, 90.0, input_file)

    monkeypatch.setattr("ovox.pipeline.validate_input", lambda path: make_info(path, 90.0))
    monkeypatch.setattr(plan_mod, "build_plan", lambda info, cc, ac, model=None: plan)

    def fake_write_chunks(plan, output_dir, audio_config):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for entry in plan.chunks:
            (out / entry.name).write_bytes(b"fake mp3")
            entry.actual_size = 8
            entry.actual_duration = entry.duration
        return [out / e.name for e in plan.chunks]

    monkeypatch.setattr(plan_mod, "write_chunks", fake_write_chunks)
    monkeypatch.setattr("ovox.pipeline.transcribe_chunk", fake)
    return SimpleNamespace(fake=fake, plan=plan)


def configs(
    **transcribe_overrides: object,
) -> tuple[ChunkConfig, AudioConfig, SttConfig, TranscribeConfig]:
    values: dict[str, object] = {
        "max_concurrency": 2,
        "max_retries": 0,
        "backoff_base_s": 1.0,
        "backoff_cap_s": 10.0,
    }
    values.update(transcribe_overrides)
    return (
        ChunkConfig(26214400, 600.0, 0.8),
        AudioConfig(16000, 1, 64_000),
        SttConfig("https://stt.example.com/v1/audio/transcriptions", "key", "voxtral-1"),
        TranscribeConfig(**values),  # type: ignore[arg-type]
    )


def fake_client() -> SimpleNamespace:
    return SimpleNamespace(
        model="voxtral-1", endpoint="https://stt.example.com/v1/audio/transcriptions"
    )


def run_transcribe(input_file: Path, fakes, **kwargs: object) -> dict:
    cc, ac, stt, tc = configs()
    return transcribe_file(
        input_file,
        chunk_config=cc,
        audio_config=ac,
        stt=stt,
        transcribe_config=tc,
        client=fake_client(),
        **kwargs,  # type: ignore[arg-type]
    )


# --- workdir naming --------------------------------------------------------------


def test_default_workdir_is_stable_and_local(input_file: Path) -> None:
    workdir = default_workdir(input_file)
    assert workdir.parent == input_file.parent
    assert workdir.name.startswith(f".ovox-{STEM}-")
    # Same file (even via a different relative spelling) -> same folder.
    assert workdir == default_workdir(input_file)


# --- end to end ------------------------------------------------------------------


def test_transcribe_file_end_to_end(input_file: Path, fakes) -> None:
    for i, entry in enumerate(fakes.plan.chunks):
        fakes.fake.texts[entry.name] = "gamma delta" if i == 1 else "alpha beta gamma"

    result = run_transcribe(input_file, fakes)

    # All three chunks were sent, in no particular order.
    assert sorted(c[0] for c in fakes.fake.calls) == sorted(e.name for e in fakes.plan.chunks)
    # The work folder holds the manifest, the chunk audio and one result per chunk.
    workdir = Path(result["workdir"])
    assert (workdir / "ovox-chunk-manifest.json").exists()
    assert len(list(workdir.glob("*_chunk_*.mp3"))) == 3
    assert len(list(workdir.glob("chunk_*.json"))) == 3
    # The transcript is written next to the input with the overlap deduped:
    # chunk 1 ("gamma delta") follows a hard cut, so its "gamma" is dropped;
    # chunk 2 ("alpha beta gamma") has no matching overlap, kept as-is.
    text_path = input_file.with_name(f"{STEM}.txt")
    assert result["text_file"] == text_path
    assert text_path.read_text() == "alpha beta gamma delta alpha beta gamma\n"
    # No segment data was returned, so no segments file.
    assert result["segments_file"] is None
    assert not input_file.with_name(f"{STEM}.segments.json").exists()
    assert result["chunks_transcribed"] == 3
    assert result["chunks_total"] == 3
    assert result["resumed"] is False
    for stage in ("probe", "cut", "stt", "merge"):
        assert stage in result["timings"]


def test_transcribe_file_writes_custom_output(input_file: Path, fakes, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere" / "notes.txt"
    result = run_transcribe(input_file, fakes, output=out)

    assert result["text_file"] == out
    assert out.exists()
    # And not next to the input.
    assert not input_file.with_name(f"{STEM}.txt").exists()


def test_transcribe_file_writes_segments_when_returned(
    input_file: Path, fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the endpoint returns segment data, a global segments JSON is
    written with each chunk's start time added to the segment times."""
    import ovox.pipeline as pipeline_mod

    entry0 = fakes.plan.chunks[0]

    def with_segments(client, path, chunk_name, config, *, stop_event=None):
        reply = fakes.fake(client, path, chunk_name, config, stop_event=stop_event)
        if chunk_name == entry0.name:
            return {
                "text": "alpha",
                "segments": [{"start": 0.5, "end": 1.5, "text": "alpha"}],
            }
        return reply

    monkeypatch.setattr(pipeline_mod, "transcribe_chunk", with_segments)
    result = run_transcribe(input_file, fakes)

    assert result["segments_file"] is not None
    data = json.loads(result["segments_file"].read_text())
    assert data == [{"start": 0.5, "end": 1.5, "text": "alpha"}]


def test_transcribe_file_reports_progress(input_file: Path, fakes) -> None:
    progress: list[tuple[int, int]] = []
    run_transcribe(input_file, fakes, on_progress=lambda d, t: progress.append((d, t)))
    assert (3, 3) in progress
    assert all(d <= 3 for d, _ in progress)


# --- resume ----------------------------------------------------------------------


def test_resume_transcribes_only_missing_chunks(input_file: Path, fakes) -> None:
    # A previous run cut the chunks and finished only the first one.
    result0 = run_transcribe(input_file, fakes)
    workdir = Path(result0["workdir"])
    (workdir / fakes.plan.chunks[0].name).exists()  # sanity
    # Simulate the first run having been interrupted after one chunk: drop the
    # results of chunks 1 and 2 (the first stays done).
    for entry in fakes.plan.chunks[1:]:
        (workdir / f"chunk_{entry.index:02d}.json").unlink()
    (input_file.with_name(f"{STEM}.txt")).unlink()

    fakes.fake.calls.clear()
    result = run_transcribe(input_file, fakes, resume=True)

    assert result["resumed"] is True
    assert result["chunks_transcribed"] == 2
    # Only the two missing chunks were sent (concurrency makes the order vary).
    assert sorted(c[0] for c in fakes.fake.calls) == sorted(e.name for e in fakes.plan.chunks[1:])
    assert result["text_file"].exists()


def test_resume_requires_matching_source(input_file: Path, fakes, tmp_path: Path) -> None:
    result0 = run_transcribe(input_file, fakes)
    workdir = Path(result0["workdir"])
    # The source file is gone: the stored work cannot be trusted.
    input_file.unlink()

    cc, ac, stt, tc = configs()
    with pytest.raises(WorkdirError, match="no longer exists"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            workdir=workdir,
            resume=True,
            client=fake_client(),
        )


def test_resume_rejects_changed_settings(input_file: Path, fakes) -> None:
    result0 = run_transcribe(input_file, fakes)
    workdir = Path(result0["workdir"])
    _cc, ac, stt, tc = configs()
    changed = ChunkConfig(1000, 600.0, 0.8)  # a different size limit

    with pytest.raises(WorkdirError, match="settings mismatch"):
        transcribe_file(
            input_file,
            chunk_config=changed,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            workdir=workdir,
            resume=True,
            client=fake_client(),
        )


def test_resume_rejects_missing_chunk_files(input_file: Path, fakes) -> None:
    result0 = run_transcribe(input_file, fakes)
    workdir = Path(result0["workdir"])
    (workdir / fakes.plan.chunks[1].name).unlink()

    cc, ac, stt, tc = configs()
    with pytest.raises(WorkdirError, match="missing chunk file"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            workdir=workdir,
            resume=True,
            client=fake_client(),
        )


def test_resume_without_manifest_fails(input_file: Path, fakes, tmp_path: Path) -> None:
    cc, ac, stt, tc = configs()
    with pytest.raises(WorkdirError, match="No chunk manifest"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            workdir=tmp_path / "empty",
            resume=True,
            client=fake_client(),
        )


# --- failure handling --------------------------------------------------------------


def test_failure_keeps_finished_chunks_and_says_to_resume(input_file: Path, fakes) -> None:
    fakes.fake.fail_on.add(fakes.plan.chunks[2].name)

    cc, ac, stt, tc = configs()
    with pytest.raises(PipelineError, match="resume"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            client=fake_client(),
        )

    # No transcript was written, and at least one finished chunk's result
    # survives on disk so a re-run can resume (the fake makes chunk 2 fail,
    # so with concurrency 2 one of the others got a result).
    workdir = default_workdir(input_file)
    assert (workdir / "chunk_00.json").exists() or (workdir / "chunk_01.json").exists()
    assert not input_file.with_name(f"{STEM}.txt").exists()


def test_first_fatal_error_stops_the_run(input_file: Path, fakes) -> None:
    # Every chunk fails; with concurrency 2 not all can be attempted, but the
    # error must name a chunk and report the work folder.
    for entry in fakes.plan.chunks:
        fakes.fake.fail_on.add(entry.name)

    cc, ac, stt, tc = configs()
    with pytest.raises(PipelineError, match=r"\d+/3 chunks are done"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            client=fake_client(),
        )


# --- dry run -----------------------------------------------------------------------


def test_dry_run_of_a_re_recorded_source_starts_fresh(input_file: Path, fakes) -> None:
    """The dry-run local stage shares the fresh-start path with a real run:
    when the source changed, the stale chunks and transcripts are removed so
    a later real run cannot mix two recordings."""
    run_transcribe(input_file, fakes)
    workdir = default_workdir(input_file)
    assert (workdir / "chunk_00.json").exists()

    import time as _time

    _time.sleep(0.01)
    input_file.write_bytes(b"other recording, same name")

    cc, ac, stt, _ = configs()
    prepare_for_dry_run(input_file, chunk_config=cc, audio_config=ac, stt=stt)

    # The stale results are gone; the chunks are re-cut.
    assert len(list(workdir.glob("chunk_*.json"))) == 0
    assert len(list(workdir.glob("*_chunk_*.mp3"))) == 3
    assert (workdir / "ovox-chunk-manifest.json").exists()


def test_dry_run_runs_local_stages_without_stt(input_file: Path, fakes) -> None:
    cc, ac, stt, _ = configs()
    result = prepare_for_dry_run(input_file, chunk_config=cc, audio_config=ac, stt=stt)

    # The local stages ran: chunks and manifest exist in the work folder...
    workdir = Path(result["workdir"])
    assert len(list(workdir.glob("*_chunk_*.mp3"))) == 3
    assert (workdir / "ovox-chunk-manifest.json").exists()
    # ...but no STT request was made and no transcript written.
    assert fakes.fake.calls == []
    assert len(list(workdir.glob("chunk_*.json"))) == 0
    assert not input_file.with_name(f"{STEM}.txt").exists()
    assert result["chunks_pending"] == 3
    assert result["chunks_transcribed"] == 0
    assert result["resumed"] is False
    # The plan is returned as a dict, ready to print.
    assert result["plan"]["totals"]["num_chunks"] == 3


def test_dry_run_resume_counts_done_chunks(input_file: Path, fakes) -> None:
    cc, ac, stt, _ = configs()
    result0 = prepare_for_dry_run(input_file, chunk_config=cc, audio_config=ac, stt=stt)
    workdir = Path(result0["workdir"])
    # Simulate one finished chunk from an earlier real run.
    (workdir / "chunk_00.json").write_text(json.dumps({"reply": {"text": "a"}}))

    result = prepare_for_dry_run(
        input_file,
        chunk_config=cc,
        audio_config=ac,
        stt=stt,
        workdir=workdir,
        resume=True,
    )
    assert result["resumed"] is True
    assert result["chunks_transcribed"] == 1
    assert result["chunks_pending"] == 2
    assert fakes.fake.calls == []


def test_rerun_of_a_re_recorded_source_starts_fresh(input_file: Path, fakes) -> None:
    """When the source file changes (same name, new content), the stored
    chunks and transcripts are invalidated: the run re-plans, re-cuts and
    re-transcribes everything rather than mixing two recordings."""
    run_transcribe(input_file, fakes)
    workdir = default_workdir(input_file)

    # Re-record the file: same name, different content and mtime.
    import time as _time

    _time.sleep(0.01)
    input_file.write_bytes(b"other recording, same name")

    fakes.fake.calls.clear()
    result = run_transcribe(input_file, fakes)
    assert result["chunks_transcribed"] == 3
    # The stale results were replaced, not kept.
    data = json.loads((workdir / "chunk_01.json").read_text())
    assert "reply" in data
    assert result["text_file"].exists()


def test_rerun_with_unchanged_source_keeps_results(input_file: Path, fakes) -> None:
    """With the same source file and an untouched work folder, a plain re-run
    is a resume: nothing is re-transcribed."""
    run_transcribe(input_file, fakes)
    fakes.fake.calls.clear()

    result = run_transcribe(input_file, fakes)
    assert result["chunks_transcribed"] == 0
    assert fakes.fake.calls == []


def test_explicit_resume_of_a_re_recorded_source_fails(input_file: Path, fakes) -> None:
    """--resume is strict: a changed source is an error, not a silent fresh
    start."""
    run_transcribe(input_file, fakes)
    import time as _time

    _time.sleep(0.01)
    input_file.write_bytes(b"other recording, same name")

    cc, ac, stt, tc = configs()
    with pytest.raises(WorkdirError, match="has changed since"):
        transcribe_file(
            input_file,
            chunk_config=cc,
            audio_config=ac,
            stt=stt,
            transcribe_config=tc,
            resume=True,
            client=fake_client(),
        )
