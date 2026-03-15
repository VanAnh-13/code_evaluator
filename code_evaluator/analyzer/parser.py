"""
Parser for Code Evaluator
Parses LLM JSON output into structured analysis results
"""

import re
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Category mapping from API output to internal result keys
CATEGORY_MAP = {
    "bug": "bugs",
    "bugs": "bugs",
    "memory": "memory_issues",
    "memory_issues": "memory_issues",
    "security": "security_vulnerabilities",
    "security_vulnerabilities": "security_vulnerabilities",
    "performance": "performance_issues",
    "performance_issues": "performance_issues",
    "style": "style_issues",
    "style_issues": "style_issues",
}


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the JSON response from the LLM API into structured results.

    Args:
        response_text: Raw JSON text from the LLM

    Returns:
        Dictionary containing structured analysis results
    """
    results = {
        "bugs": [],
        "memory_issues": [],
        "security_vulnerabilities": [],
        "performance_issues": [],
        "style_issues": [],
        "summary": "",
        "overall_score": 0,
        "suggested_fixes": {},
    }

    try:
        # Try to extract JSON from response (handle markdown code fences)
        json_text = _extract_json(response_text)
        data = json.loads(json_text)

        # Extract summary and score
        results["summary"] = data.get("summary", "")
        results["overall_score"] = max(0, min(100, int(data.get("overall_score", 0))))

        # Parse issues into categories
        for issue in data.get("issues", []):
            category = issue.get("category", "bug").lower()
            result_key = CATEGORY_MAP.get(category, "bugs")

            parsed_issue = {
                "line": int(issue.get("line", 0)),
                "severity": issue.get("severity", "medium").lower(),
                "description": issue.get("description", ""),
                "recommendation": issue.get("recommendation", "No recommendation provided."),
            }

            # Validate severity
            if parsed_issue["severity"] not in ("critical", "high", "medium", "low", "info"):
                parsed_issue["severity"] = "medium"

            results[result_key].append(parsed_issue)

        # Parse suggested fixes
        for fix in data.get("suggested_fixes", []):
            line = str(fix.get("line", 0))
            fix_text = fix.get("fixed", "")
            original = fix.get("original", "")
            explanation = fix.get("explanation", "")

            if fix_text:
                fix_content = ""
                if original:
                    fix_content += f"// Original:\n{original}\n\n// Fixed:\n"
                fix_content += fix_text
                if explanation:
                    fix_content += f"\n\n// {explanation}"
                results["suggested_fixes"][line] = fix_content

        # Extract detected language if present
        if "language" in data:
            results["detected_language"] = data["language"]

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning(f"Failed to parse JSON response: {e}. Falling back to text parser.")
        # Fallback to legacy text parser
        text_results = parse_analysis(response_text)
        results.update(text_results)

    return results


def _extract_json(text: str) -> str:
    """
    Extract JSON from text that may contain markdown code fences or extra content.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Cleaned JSON string
    """
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return text


def parse_analysis(analysis_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy parser: Parse the model's text output into structured analysis results.
    Used as fallback when JSON parsing fails.

    Args:
        analysis_text: Text output from the model

    Returns:
        Dictionary containing parsed analysis results
    """
    results = {
        "bugs": [],
        "memory_issues": [],
        "security_vulnerabilities": [],
        "performance_issues": [],
        "style_issues": []
    }

    try:
        lines = analysis_text.split('\n')
        current_section = "bugs"
        issue_buffer = {}

        # Map section titles to result keys
        section_mapping = {
            "bugs": "bugs",
            "logical errors": "bugs",
            "memory": "memory_issues",
            "memory management": "memory_issues",
            "security": "security_vulnerabilities",
            "performance": "performance_issues",
            "style": "style_issues",
            "readability": "style_issues"
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            lower_line = line.lower()
            for section_key, result_key in section_mapping.items():
                if section_key in lower_line and (line.startswith('#') or line.startswith('**')):
                    current_section = result_key
                    break

            # Check if this is an issue line
            if (line.startswith('- ') or line.startswith('* ') or
                line.startswith('Line ') or re.match(r'^\d+\.', line)):

                if issue_buffer and 'description' in issue_buffer and current_section:
                    _finalize_issue(issue_buffer, results, current_section)

                issue_buffer = {'description': line}

                line_match = re.search(r'Line\s+(\d+)', line)
                if line_match:
                    issue_buffer['line'] = int(line_match.group(1))

                issue_buffer['severity'] = _extract_severity(line)

            elif current_section and issue_buffer and ('recommendation' in line.lower() or 'suggestion' in line.lower()):
                recommendation_text = line.split(':', 1)[1].strip() if ':' in line else line
                issue_buffer['recommendation'] = recommendation_text

            elif current_section and issue_buffer:
                if 'recommendation' not in issue_buffer and line.startswith('  '):
                    issue_buffer['recommendation'] = line.strip()
                else:
                    issue_buffer['description'] += " " + line

        if issue_buffer and 'description' in issue_buffer and current_section:
            _finalize_issue(issue_buffer, results, current_section)

    except Exception as e:
        logger.error(f"Failed to parse analysis: {str(e)}")

    return results


def _finalize_issue(issue_buffer: Dict[str, Any], results: Dict[str, List], current_section: str) -> None:
    """Finalize an issue by adding default values and appending to results"""
    if 'line' not in issue_buffer:
        issue_buffer['line'] = 0
    if 'severity' not in issue_buffer:
        issue_buffer['severity'] = 'medium'
    if 'recommendation' not in issue_buffer:
        issue_buffer['recommendation'] = "No specific recommendation provided."
    results[current_section].append(issue_buffer.copy())


def _extract_severity(line: str) -> str:
    """Extract severity from a line of text"""
    severity_match = re.search(r'\((critical|high|medium|low|info)\)', line, re.IGNORECASE)
    if severity_match:
        return severity_match.group(1).lower()

    lower_line = line.lower()
    if any(word in lower_line for word in ['critical', 'crash', 'exploit', 'vulnerability', 'security']):
        return 'critical'
    elif any(word in lower_line for word in ['high', 'error', 'bug', 'memory leak']):
        return 'high'
    elif any(word in lower_line for word in ['medium', 'warning', 'performance']):
        return 'medium'
    elif any(word in lower_line for word in ['low', 'style', 'readability']):
        return 'low'
    else:
        return 'info'
