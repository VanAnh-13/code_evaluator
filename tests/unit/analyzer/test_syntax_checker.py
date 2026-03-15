"""Tests for syntax checker module"""

import pytest
from code_evaluator.analyzer.syntax_checker import check_syntax


class TestSyntaxChecker:
    """Test syntax checking functionality"""

    def test_valid_python_syntax(self):
        """Test valid Python code passes syntax check"""
        code = "def hello():\n    print('Hello, World!')"
        result = check_syntax(code, "python")
        assert result["valid"] is True
        assert len(result.get("errors", [])) == 0

    def test_invalid_python_syntax(self):
        """Test invalid Python code fails syntax check"""
        code = "def hello(\n    print('Missing closing paren')"
        result = check_syntax(code, "python")
        assert result["valid"] is False
        assert len(result.get("errors", [])) > 0

    def test_valid_javascript_basic(self):
        """Test basic JavaScript syntax validation"""
        code = "function greet() { console.log('Hello'); }"
        result = check_syntax(code, "javascript")
        # Should return valid or handle gracefully
        assert "valid" in result

    def test_cpp_syntax_check(self):
        """Test C++ syntax checking (if implemented)"""
        code = "#include <iostream>\nint main() { return 0; }"
        result = check_syntax(code, "cpp")
        assert "valid" in result

    def test_unsupported_language(self):
        """Test handling of unsupported languages"""
        code = "some code"
        result = check_syntax(code, "unsupported_lang")
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "valid" in result

    def test_empty_code(self):
        """Test handling of empty code"""
        result = check_syntax("", "python")
        assert isinstance(result, dict)
        assert "valid" in result

    def test_multiline_python_with_indentation(self):
        """Test Python code with proper indentation"""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
"""
        result = check_syntax(code, "python")
        assert result["valid"] is True
