"""Check that the version is consistent everywhere it appears."""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

import ovox
from ovox.cli import main

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Simplified PEP 440: release segment plus optional pre/post/dev suffixes.
PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as f:
        version: str = tomllib.load(f)["project"]["version"]
    return version


def test_version_is_valid_pep440() -> None:
    assert PEP440.match(ovox.__version__), ovox.__version__


def test_package_version_matches_pyproject() -> None:
    assert ovox.__version__ == _pyproject_version(), (
        "Version mismatch: update both src/ovox/__init__.py and pyproject.toml"
    )


def test_installed_metadata_matches_package() -> None:
    # Fails if the environment holds a stale install; re-run `uv sync` to fix.
    assert importlib.metadata.version("open-vox") == ovox.__version__


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert ovox.__version__ in capsys.readouterr().out
