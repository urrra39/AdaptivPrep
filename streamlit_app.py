"""Streamlit Cloud entry point — keeps imports on the repo root, not /mount/src/."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.quiz_app import main

main()
