"""
Synapse v0.2 — Shared memory store.

Async-safe key/value store that any agent can read from or write to.
All access is lock-protected so agents running concurrently won't corrupt data.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class SharedMemory:
    """Thread-safe (asyncio-safe) shared key/value store for agents."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = value

    async def append(self, key: str, value: Any) -> None:
        """Append a value to a list stored at key. Creates the list if needed."""
        async with self._lock:
            if key not in self._store:
                self._store[key] = []
            self._store[key].append(value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())

    async def all(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._store)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def has(self, key: str) -> bool:
        async with self._lock:
            return key in self._store
