"""
scripts/run_gui.py
Khởi chạy giao diện GUI chính của FaceAttend.

Chạy:
    .venv\\Scripts\\python.exe scripts/run_gui.py

Yêu cầu:
    pip install pillow
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.gui.main_window import main

if __name__ == "__main__":
    main()
