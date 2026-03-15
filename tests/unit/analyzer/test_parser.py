"""Tests for parser module"""

import pytest
import json
from code_evaluator.analyzer.parser import parse_json_response, parse_analysis


class TestParser:
    """Test response parsing functionality"""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response"""
        response = """{
            "summary": "Code has potential issues",
            "overall_score": 75,
            "issues": [
                {
                    "type": "bug",
                    "severity": "high",
                    "line": 5,
                    "description": "Division by zero",
                    "suggestion": "Add zero check"
                }
            ]
        }"""
        result = parse_json_response(response)
        assert result is not None
        assert result["overall_score"] == 75
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "bug"

    def test_parse_json_with_markdown_wrapper(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        response = """```json
{
    "summary": "Test summary",
    "overall_score": 90,
    "issues": []
}
```"""
        result = parse_json_response(response)
        assert result is not None
        assert result["overall_score"] == 90
        assert result["issues"] == []

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with surrounding text"""
        response = """Here is the analysis:

{"summary": "Good code", "overall_score": 95, "issues": []}

Hope this helps!"""
        result = parse_json_response(response)
        assert result is not None
        assert result["overall_score"] == 95

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON"""
        response = "This is not valid JSON at all"
        result = parse_json_response(response)
        # Should return None or default structure
        assert result is None or isinstance(result, dict)

    def test_parse_analysis_with_complete_data(self, mock_openai_response):
        """Test parsing complete analysis data"""
        result = parse_analysis(json.dumps(mock_openai_response))
        assert result is not None
        assert "summary" in result
        assert "overall_score" in result
        assert "issues" in result

    def test_parse_empty_issues_list(self):
        """Test parsing response with empty issues"""
        response = '{"summary": "Perfect code", "overall_score": 100, "issues": []}'
        result = parse_json_response(response)
        assert result["issues"] == []
        assert result["overall_score"] == 100

    def test_parse_malformed_json(self):
        """Test handling of malformed JSON"""
        response = '{"summary": "Test", "overall_score": 80, "issues": [}'
        result = parse_json_response(response)
        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    def test_parse_nested_structures(self):
        """Test parsing JSON with nested structures"""
        response = """{
            "summary": "Complex analysis",
            "overall_score": 70,
            "issues": [{
                "type": "security",
                "severity": "critical",
                "details": {
                    "cwe": "CWE-89",
                    "owasp": "A1"
                }
            }]
        }"""
        result = parse_json_response(response)
        assert result is not None
        assert result["issues"][0]["details"]["cwe"] == "CWE-89"
