"""
Cache utilities for Code Evaluator
Provides content-based caching for analysis results
"""

import hashlib
from typing import Any, Dict, Optional


class ContentCache:
    """
    A content-based cache that uses SHA-256 hash of content combined with file path
    to create unique cache keys. This prevents stale cache results when file content
    changes but path/modification time remain the same.
    """

    def __init__(self):
        """Initialize an empty cache"""
        self._cache: Dict[str, Any] = {}

    def generate_key(self, file_path: str, content: str) -> str:
        """
        Generate a cache key based on file path and content hash

        Args:
            file_path: Path to the file
            content: Content of the file

        Returns:
            Cache key string combining file path and content hash
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"{file_path}_{content_hash}"

    def get(self, file_path: str, content: str) -> Optional[Any]:
        """
        Get cached result for the given file path and content

        Args:
            file_path: Path to the file
            content: Current content of the file

        Returns:
            Cached result if found, None otherwise
        """
        key = self.generate_key(file_path, content)
        return self._cache.get(key)

    def set(self, file_path: str, content: str, result: Any) -> None:
        """
        Store a result in the cache

        Args:
            file_path: Path to the file
            content: Content of the file
            result: Result to cache
        """
        key = self.generate_key(file_path, content)
        self._cache[key] = result

    def invalidate(self, file_path: str = None) -> None:
        """
        Invalidate cache entries

        Args:
            file_path: If provided, invalidate entries for this file path only.
                      If None, clear the entire cache.
        """
        if file_path is None:
            self._cache.clear()
        else:
            # Remove all entries that start with the file path
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{file_path}_")]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached entries"""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache"""
        return key in self._cache
