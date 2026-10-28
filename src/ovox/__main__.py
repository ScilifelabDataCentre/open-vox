"""Allow running ovox as `python -m ovox`."""

import sys

from ovox.cli import main

sys.exit(main())
