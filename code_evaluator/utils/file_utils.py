"""
File utilities for Code Evaluator
Handles language detection, file I/O, and path operations
"""

import os
from typing import Tuple, Optional


# Map file extensions to languages
LANGUAGE_MAP = {
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.py': 'python',
    '.js': 'javascript',
    '.html': 'html',
    '.css': 'css',
    '.java': 'java',
    '.c': 'c',
    '.cs': 'csharp',
    '.php': 'php',
    '.rb': 'ruby',
    '.go': 'go',
    '.rs': 'rust',
    '.ts': 'typescript',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
}

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {
    # C++
    '.cpp', '.cc', '.cxx', '.h', '.hpp',
    # Python
    '.py',
    # JavaScript
    '.js',
    # HTML/CSS
    '.html', '.css',
    # Java
    '.java',
    # C#
    '.cs',
    # PHP
    '.php',
    # Ruby
    '.rb',
    # Go
    '.go',
    # Rust
    '.rs',
    # TypeScript
    '.ts',
    # Swift
    '.swift',
    # Kotlin
    '.kt',
    # Scala
    '.scala',
    # C
    '.c'
}


def detect_language(file_path: str) -> str:
    """
    Detect the programming language based on file extension

    Args:
        file_path: Path to the code file

    Returns:
        Detected language name (lowercase), or 'unknown' if not recognized
    """
    ext = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_MAP.get(ext, 'unknown')


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path

    Args:
        file_path: Path to the file

    Returns:
        File extension (lowercase, including the dot)
    """
    return os.path.splitext(file_path)[1].lower()


def is_allowed_extension(filename: str) -> bool:
    """
    Check if the file has an allowed extension

    Args:
        filename: Name of the file

    Returns:
        True if the extension is allowed, False otherwise
    """
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def read_file_content(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read file content with proper encoding handling

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (content, error_message). If successful, error_message is None.
        If failed, content is None and error_message contains the error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content, None
    except IOError as e:
        error_msg = f"IO Error: {str(e)}"
        suggestion = "Make sure the file exists and you have permission to read it."
        return None, f"{error_msg}. {suggestion}"
    except UnicodeDecodeError as e:
        error_msg = f"Encoding error: {str(e)}"
        suggestion = "The file might be binary or use an unsupported encoding. Try saving it with UTF-8 encoding."
        return None, f"{error_msg}. {suggestion}"


def count_lines(content: str) -> int:
    """
    Count the number of lines in content

    Args:
        content: Text content

    Returns:
        Number of lines
    """
    return len(content.splitlines())
