# Open Vox

![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)

Open Vox, also known as ovox, is a service for audio and video transcription to text, supporting custom transcription models.

## Branch Strategy

This repository used trunk-based development, meaning we skip a development branch.
Work should be performed in short-lived feature branches, then merged into the main branch after PR review.

## Setup for local development

```bash
# Clone the git repository:
git clone https://github.com/ScilifelabDataCentre/open-vox.git
cd ./open-vox

# Create .venv on Python 3.14, install ovox in editable mode, and install the dev tools
uv sync --python 3.14 --extra dev
# Pin the python environment
uv python pin 3.14

# Option 1: run through uv without activating
uv run ovox -V
uv run ovox version

# Option 2: activate the environment and run directly
source .venv/bin/activate
ovox -V
```

## Development

### Versioning

This project uses semantic versioning: "major.minor.patch"

When starting new work, bump the appropriate version in:
- ./pyproject.toml, version = "major.minor.patch"
- ./src/ovox/__init__.py
Keep these two in sync (checked by tests/test_version.py)

### Git workflow

Create feature branch from main:
```bash
git checkout main
git fetch origin
git pull

git checkout -b AIS-<nr>-<branch-task>

# ... do work ...

# run quality gate checks:
chmod +x ./scripts/check.sh
./scripts/check.sh

# or run each check individually:
# format
ruff format .    # rewrites files in place
# lint
ruff check .
uv run ruff format --check src tests
# mypy checks
uv run mypy

# run tests
pytest tests/unit -v

git add .
git commit -S -m "<commit-message>"
git push -u origin <branch name>

# Create PR
```

### Managing the virtual environment and maintaining dependencies

The python virtual environment is managed with uv together with a pyproject.toml file.

```bash
# Packages are added and removed using uv:
uv add <pkg>
uv remove <pkg>

# Add a dev tool into the [dependency-groups]:
uv add --dev <pkg>
```

uv add updates pyproject.toml, updates uv.lock, and installs the package into .venv.
uv add fills in a sensible lower bound. Commit both files whenever they change.
uv.lock records the exact versions installed.

Optionals, --optional, are to be reserved for optional backends that the user can install.

## Testing

The project contains both automated unit and integration tests. These reside under the tests/unit and tests/integration directories respectively.

Unit tests are fast and light and each developer should run them often as part of active development in the git flow as described above.
Unit tests are also run as part of CI.

To run the unit tests:
```bash
pytest tests/unit -v
```

Integration tests work with real files and are therefore a valuable assurance that it works in a more realistic context.

Integration tests that use media files require the ffmpeg tool.

Install ffmpeg:
```bash
sudo apt install ffmpeg
```

ovox calls ffmpeg/ffprobe through `subprocess` and does not pin a version: it
should work on whatever you have installed. CI runs the integration tests
against the distro ffmpeg on `ubuntu-24.04` (ffmpeg 6.1) and `ubuntu-latest`
(ffmpeg 8.0). Chunk lengths are measured by decoding, so they do not depend on
the ffmpeg version.

To run the integration tests:
```bash
pytest tests/integration -v
```

You can and should run the integration test against real media files.
The project repository contains some in the tests/data directory.
```bash
OVOX_TEST_MEDIA_DIR=./tests/data pytest -m local_media -v
```

In this example, we have copied some real media files into the ~/media directory.
Then run:
```bash
OVOX_TEST_MEDIA_DIR=~/media pytest -m local_media -v
```

The live tests transcribe the committed short speech clips (and, with
`OVOX_TEST_MEDIA_DIR`, your own files) with the real STT endpoint configured
in `.env`, and check the transcript against the committed expected `.txt`.
They are excluded from the default run because they cost real API calls;
run `ovox check` first, then:
```bash
pytest -m live                                  # real transcription of the committed files
OVOX_TEST_MEDIA_DIR=~/media pytest -m "live and local_media"   # your own long files
```
A local file is picked up only when it has a sibling `.txt` with the same
name and the expected transcript in it.

## Usage

The first usage of open-vox is as a CLI.

Copy the .env.example file to .env and configure:
```bash
cp .env.example .env
```

### CLI Usage

```bash
ovox --version (or -V) prints the version of ovox and Python.

ovox version
prints ovox, Python, implementation and platform details. With --json it prints them as JSON, which is useful for recording the environment alongside performance test results.

ovox check
Check if the STT API is reachable and credentials are valid.

ovox chunk <file> --output-dir <dir>
Split an audio or video file into chunks that fit the STT endpoint's size and
duration limits, writing them (and a JSON manifest) to the output directory. No
transcription is done.
  --dry-run   Print the plan and write nothing (safe; no files, no directory).
  --plan-only Write the manifest to the output directory, but no chunk files.
  --json      Print the plan as JSON (its content is the manifest).
--dry-run and --plan-only are mutually exclusive.

ovox transcribe <file>
Transcribe an audio or video file end to end: split it into chunks that fit
the STT endpoint's limits, send each chunk with retries and bounded
concurrency, and merge the results into a .txt transcript next to the input.
The chunk audio, the manifest and one result file per chunk are kept in a
work folder next to the input (.ovox-<name>-<hash>/), so an interrupted run
can be resumed.
  -o, --output  Write the transcript to this .txt file (default: next to the input).
  --workdir DIR Use this directory as the work folder.
  --dry-run     Run the local steps (validate, cut) into the work folder and
                print the plan, without making any STT requests.
  --resume      Reuse the chunks a previous run cut (checking that the source
                and settings still match) and only transcribe the chunks
                without a result.
  --json        Print the result (or the plan, with --dry-run) as JSON,
                including per-stage timings; progress goes to stderr.
--dry-run and --resume are mutually exclusive.

ovox with no arguments shows help.
```
