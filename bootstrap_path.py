"""Ensure ``filter_pipeline``, ``nifty50_symbols``, and ``patterns`` are importable.

Vendored copies live in this directory for Streamlit Cloud / standalone deploy.
Falls back to parent ``test/`` when running inside the monorepo without local copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if not (_ROOT / "filter_pipeline.py").exists():
    _TEST_ROOT = _ROOT.parent
    if str(_TEST_ROOT) not in sys.path:
        sys.path.insert(0, str(_TEST_ROOT))
