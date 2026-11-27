#!/usr/bin/env bash
# Quality gate: lint, format, type-check.
#   scripts/check.sh         verify only, never modify files (local + CI)
#   scripts/check.sh --fix   apply lint fixes and formatting, then type-check
set -euo pipefail
cd "$(dirname "$0")/.."   # run from repo root regardless of cwd

PATHS=(src tests)

if [[ "${1:-}" == "--fix" ]]; then
  uv run ruff check --fix "${PATHS[@]}"   # lint fixes first...
  uv run ruff format "${PATHS[@]}"        # ...then format (Ruff's recommended order)
else
  uv run ruff check "${PATHS[@]}"
  uv run ruff format --check "${PATHS[@]}"
fi

uv run mypy
