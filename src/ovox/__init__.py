"""Open Vox (ovox): audio and video transcription to text.

ovox transcribes audio and video files to text and supports plugging in
custom transcription models.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Package version. Keep in sync with `version` in pyproject.toml;
# tests/test_version.py fails if the two differ.
__version__ = "0.1.0"
