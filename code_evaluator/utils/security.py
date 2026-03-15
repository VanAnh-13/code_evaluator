"""
Security utilities for Code Evaluator
Provides path validation, content type checking, and other security measures
"""

import os
import tempfile
from typing import List, Optional, Tuple


def validate_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a file path for security (path traversal protection)

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directories. If None, uses cwd and temp directory.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    # Check for path traversal attempts using '..'
    normalized_path = os.path.normpath(file_path)
    if '..' in normalized_path.split(os.sep):
        return False, "Path traversal not allowed: file path contains '..'"

    # Get the real (absolute) path
    real_path = os.path.realpath(file_path)

    # Define allowed directories
    if allowed_dirs is None:
        allowed_dirs = [
            os.path.realpath(os.getcwd()),
            os.path.realpath(tempfile.gettempdir()),
        ]
    else:
        allowed_dirs = [os.path.realpath(d) for d in allowed_dirs]

    # Check if the real path is within an allowed directory
    is_allowed = any(
        real_path.startswith(allowed_dir + os.sep) or real_path == allowed_dir
        for allowed_dir in allowed_dirs
    )

    if not is_allowed:
        return False, "Path traversal not allowed: file path is outside the allowed directory"

    return True, None


def is_text_file(file_path: str, chunk_size: int = 8192) -> bool:
    """
    Check if a file contains text content by reading the first chunk
    and looking for null bytes (binary indicator)

    Args:
        file_path: Path to the file
        chunk_size: Number of bytes to read for checking

    Returns:
        True if the file appears to be text, False if binary
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size)
        return b'\x00' not in chunk
    except (IOError, OSError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing potentially dangerous characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', '..', ':', '*', '?', '"', '<', '>', '|']
    result = filename
    for char in dangerous_chars:
        result = result.replace(char, '_')
    return result


def check_file_size(file_path: str, max_size_bytes: int) -> Tuple[bool, Optional[str]]:
    """
    Check if a file is within the allowed size limit

    Args:
        file_path: Path to the file
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_size_bytes:
            max_mb = max_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return False, f"File too large ({actual_mb:.1f}MB). Maximum size is {max_mb:.1f}MB."
        return True, None
    except OSError as e:
        return False, f"Could not check file size: {str(e)}"
