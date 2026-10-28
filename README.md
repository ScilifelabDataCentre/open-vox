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

# Option 1: run through uv without activating
uv run ovox -V
uv run ovox version

# Option 2: activate the environment and run directly
source .venv/bin/activate
ovox -V
```

## Development

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

## Usage

The first usage of open-vox is as a CLI.

### CLI Usage

```bash
ovox --version (or -V) prints the version of ovox and Python.

ovox version prints ovox, Python, implementation and platform details. With --json it prints them as JSON, which is useful for recording the environment alongside performance test results.

ovox with no arguments shows help.
```
