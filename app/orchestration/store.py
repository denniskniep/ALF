from __future__ import annotations

import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config import DatabaseConfig

class ModelStore(ABC):
    @abstractmethod
    def save(self, model_type: str, model_key: str, blob: bytes, n_learned: int) -> None: ...

    @abstractmethod
    def load(self, model_type: str, model_key: str) -> bytes | None: ...

    @abstractmethod
    def delete(self, model_type: str, model_key: str) -> None: ...

    @abstractmethod
    def get_stats(self, model_type: str, model_key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_keys(self, model_type: str) -> list[str]: ...


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS models (
    model_type   TEXT NOT NULL,
    model_key    TEXT NOT NULL,
    model_blob   BLOB,
    sample_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_type, model_key)
);
"""

class SQLiteModelStore(ModelStore):
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, model_type: str, model_key: str, blob: bytes, n_learned: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO models (model_type, model_key, model_blob, sample_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model_type, model_key) DO UPDATE SET
                    model_blob   = excluded.model_blob,
                    sample_count = excluded.sample_count,
                    last_updated = excluded.last_updated
                """,
                (model_type, model_key, blob, n_learned),
            )

    def load(self, model_type: str, model_key: str) -> bytes | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT model_blob FROM models WHERE model_type=? AND model_key=?",
                (model_type, model_key),
            ).fetchone()
        return row[0] if row is not None else None

    def delete(self, model_type: str, model_key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM models WHERE model_type=? AND model_key=?",
                (model_type, model_key),
            )

    def get_stats(self, model_type: str, model_key: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT sample_count, last_updated FROM models WHERE model_type=? AND model_key=?",
                (model_type, model_key),
            ).fetchone()
        return {"sample_count": row[0], "last_updated": row[1]} if row else None

    def list_keys(self, model_type: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model_key FROM models WHERE model_type=?", (model_type,)
            ).fetchall()
        return [r[0] for r in rows]


class InMemoryModelStore(ModelStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], tuple[bytes, int]] = {}
        self._lock = threading.Lock()

    def save(self, model_type: str, model_key: str, blob: bytes, n_learned: int) -> None:
        with self._lock:
            self._data[(model_type, model_key)] = (blob, n_learned)

    def load(self, model_type: str, model_key: str) -> bytes | None:
        entry = self._data.get((model_type, model_key))
        return entry[0] if entry is not None else None

    def delete(self, model_type: str, model_key: str) -> None:
        with self._lock:
            self._data.pop((model_type, model_key), None)

    def get_stats(self, model_type: str, model_key: str) -> dict[str, Any] | None:
        entry = self._data.get((model_type, model_key))
        return {"sample_count": entry[1], "last_updated": None} if entry is not None else None

    def list_keys(self, model_type: str) -> list[str]:
        return [key for t, key in self._data if t == model_type]


def create_store(config: DatabaseConfig) -> ModelStore:
    if config.type == "memory":
        return InMemoryModelStore()
    if config.type == "sqlite":
        if not config.path:
            raise ValueError("database.path is required for type 'sqlite'")
        return SQLiteModelStore(config.path)
    raise ValueError(f"Unknown database.type {config.type!r}. Expected 'sqlite' or 'memory'.")
