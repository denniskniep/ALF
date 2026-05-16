from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from app.schemas import IngestResponse, ScoreResponse


class TrainDetectLogger:
    """Records extracted training and scoring events to .train_and_detect/<date>/<n>/."""

    def __init__(self, enabled: bool = False, base_path: str = ".train_and_detect") -> None:
        self._enabled = enabled
        self._base = Path(base_path)
        self._lock = threading.Lock()
        self._session_dir: Path | None = None
        self._started_files: set[Path] = set()

    def log_train(self, response: IngestResponse) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._write(self._session() / "input.json", response.model_dump())

    def log_score(self, response: ScoreResponse) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._write(self._session() / "output.json", response.model_dump())

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _session(self) -> Path:
        # Called while self._lock is held.
        if self._session_dir is None:
            date_str = datetime.now().strftime("%Y_%m_%d")
            date_dir = self._base / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(
                int(d.name) for d in date_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
            n = (existing[-1] if existing else 0) + 1
            self._session_dir = date_dir / f"{n:03d}"
            self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir

    def _write(self, path: Path, obj: dict) -> None:
        content = json.dumps(obj, indent=2)
        if path not in self._started_files:
            self._started_files.add(path)
            path.write_text(f"[\n{content}\n]")
        else:
            with open(path, "r+b") as f:
                f.seek(-2, 2)  # before the closing \n]
                f.write(f",\n{content}\n]".encode())
