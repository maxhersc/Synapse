from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any


class SharedMemory:
    """Thread-safe async key/value store shared across all agents."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any) -> None:
        """Write a value to the shared store."""

        async with self._lock:
            self._store[key] = value
            self._history.append(
                {
                    "action": "set",
                    "key": key,
                    "timestamp": datetime.now(timezone.utc),
                    "value": value,
                }
            )

    async def get(self, key: str, default: Any = None) -> Any:
        """Read a value from the shared store or return the default."""

        async with self._lock:
            return self._store.get(key, default)

    async def delete(self, key: str) -> None:
        """Remove a key from the shared store."""

        async with self._lock:
            self._store.pop(key, None)
            self._history.append(
                {
                    "action": "delete",
                    "key": key,
                    "timestamp": datetime.now(timezone.utc),
                    "value": None,
                }
            )

    async def update(self, data: dict) -> None:
        """Atomically write multiple keys to the shared store."""

        async with self._lock:
            self._store.update(data)
            self._history.append(
                {
                    "action": "update",
                    "key": "multiple",
                    "timestamp": datetime.now(timezone.utc),
                    "value": dict(data),
                }
            )

    async def all(self) -> dict:
        """Return a copy of the entire shared store."""

        async with self._lock:
            return dict(self._store)

    async def has(self, key: str) -> bool:
        """Check whether a key exists in the shared store."""

        async with self._lock:
            return key in self._store

    async def history(self, limit: int | None = None) -> list[dict]:
        """Return the write history in insertion order with an optional limit."""

        async with self._lock:
            if limit is None:
                return list(self._history)
            return list(self._history[-limit:])

    def __repr__(self) -> str:
        """Return a compact representation showing the current key count."""

        return f"<SharedMemory keys={len(self._store)}>"
