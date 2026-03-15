"""Tests for file utilities"""

import pytest
import os
from code_evaluator.utils.file_utils import (
    detect_language,
    read_file_content,
    ALLOWED_EXTENSIONS
)


class TestFileUtils:
    """Test file utility functions"""

    def test_detect_language_python(self):
        """Test language detection for Python files"""
        assert detect_language("test.py") == "python"
        assert detect_language("/path/to/script.py") == "python"

    def test_detect_language_javascript(self):
        """Test language detection for JavaScript files"""
        assert detect_language("app.js") == "javascript"
        assert detect_language("module.mjs") == "javascript"

    def test_detect_language_cpp(self):
        """Test language detection for C++ files"""
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("header.hpp") == "cpp"
        assert detect_language("code.cc") == "cpp"
        assert detect_language("impl.cxx") == "cpp"

    def test_detect_language_c(self):
        """Test language detection for C files"""
        assert detect_language("program.c") == "c"
        assert detect_language("header.h") == "c"

    def test_detect_language_java(self):
        """Test language detection for Java files"""
        assert detect_language("Main.java") == "java"

    def test_detect_language_typescript(self):
        """Test language detection for TypeScript files"""
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "typescript"

    def test_detect_language_go(self):
        """Test language detection for Go files"""
        assert detect_language("main.go") == "go"

    def test_detect_language_rust(self):
        """Test language detection for Rust files"""
        assert detect_language("main.rs") == "rust"

    def test_detect_language_unknown(self):
        """Test language detection for unknown extensions"""
        result = detect_language("file.unknown")
        # Should return a default or 'unknown'
        assert result is not None

    def test_detect_language_case_insensitive(self):
        """Test language detection is case insensitive"""
        assert detect_language("Script.PY") == "python"
        assert detect_language("Main.CPP") == "cpp"

    def test_read_file_content_success(self, temp_code_file):
        """Test reading file content"""
        content = read_file_content(temp_code_file)
        assert content is not None
        assert len(content) > 0
        assert "def divide" in content

    def test_read_file_content_nonexistent(self):
        """Test reading non-existent file"""
        with pytest.raises((FileNotFoundError, IOError)):
            read_file_content("/nonexistent/path/file.py")

    def test_allowed_extensions_includes_common_languages(self):
        """Test ALLOWED_EXTENSIONS includes common file types"""
        assert 'py' in ALLOWED_EXTENSIONS or '.py' in ALLOWED_EXTENSIONS
        assert 'js' in ALLOWED_EXTENSIONS or '.js' in ALLOWED_EXTENSIONS
        assert 'cpp' in ALLOWED_EXTENSIONS or '.cpp' in ALLOWED_EXTENSIONS
        assert 'java' in ALLOWED_EXTENSIONS or '.java' in ALLOWED_EXTENSIONS

    def test_detect_language_with_path(self):
        """Test language detection with full path"""
        path = "/home/user/projects/myapp/src/main.py"
        assert detect_language(path) == "python"
