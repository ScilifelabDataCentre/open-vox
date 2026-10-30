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

### Git flow

Create feature branch from main:
```bash
git checkout main
git fetch origin
git pull

git checkout -b AIS-<nr>-<branch-task>

# ... do work ...

# lint
ruff check .

# run tests
pytest -v

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

ovox with no arguments shows help.
```
