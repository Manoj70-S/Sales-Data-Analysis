"""Entry point for running the Sales Data Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly via `python main.py` without editable install
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sales_analyzer.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
