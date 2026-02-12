"""Simple JSON state store for live controller restart safety."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "last_15m_open_time": None,
                "last_processed_1h_open_time": None,
                "position": None,
                "last_exit_hour_open_time": None,
            }
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "last_15m_open_time": None,
                "last_processed_1h_open_time": None,
                "position": None,
                "last_exit_hour_open_time": None,
            }

    def save(self, state: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
