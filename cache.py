from __future__ import annotations

"""
Personal Health — tiny in-process LRU cache with TTL.

Used by /progress, /injury-risk, /weak-joints, and /coach to keep
hot reads cheap. Thread-safe.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    def __init__(self, maxsize: int = 256, default_ttl: float = 300.0):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            expires_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


progress_cache = TTLCache(maxsize=512, default_ttl=120.0)
coach_cache = TTLCache(maxsize=256, default_ttl=3600.0)
