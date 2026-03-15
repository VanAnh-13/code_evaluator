"""
File validators for web application
Handles file upload validation, content type checking, and security
"""

import os
import uuid
from typing import Tuple, Optional

from code_evaluator.utils.file_utils import ALLOWED_EXTENSIONS, get_file_extension
from code_evaluator.utils.security import is_text_file


# Maximum file size (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Maximum lines of code
MAX_CODE_LINES = 10000


def allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed extension
    
    Args:
        filename: Name of the file
        
    Returns:
        True if the extension is allowed, False otherwise
    """
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def validate_upload(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded file
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File not found"
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size is {max_mb:.0f}MB."
    except OSError as e:
        return False, f"Could not check file size: {str(e)}"
    
    # Check if file is text (not binary)
    if not is_text_file(file_path):
        return False, "File appears to contain binary content. Please upload a valid text/code file."
    
    return True, None


def validate_code_content(content: str) -> Tuple[bool, Optional[str]]:
    """
    Validate code content
    
    Args:
        content: Code content to validate
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    # Check if content is empty
    if not content or not content.strip():
        return False, "The file is empty. Please upload a file with code content to analyze."
    
    # Check number of lines
    lines = content.splitlines()
    if len(lines) > MAX_CODE_LINES:
        return False, f"Code file exceeds the maximum limit of {MAX_CODE_LINES:,} lines. Please upload a smaller file."
    
    return True, None


def get_unique_filename(filename: str) -> str:
    """
    Generate a unique filename to prevent overwriting
    
    Args:
        filename: Original filename
        
    Returns:
        Unique filename with UUID
    """
    base, ext = os.path.splitext(filename)
    return f"{base}_{uuid.uuid4().hex}{ext}"
