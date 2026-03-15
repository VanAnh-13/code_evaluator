"""
Report exporter for Code Evaluator
Handles saving analysis results to various formats
"""

import os
import json
from typing import Dict, Any, List, Optional

from code_evaluator.report.generator import generate_report, generate_summary_report


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis results to a JSON file

    Args:
        results: Analysis results dictionary
        output_path: Path to save the results
    """
    # Ensure directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Results saved to {output_path}")


def save_report(results: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis report as Markdown file

    Args:
        results: Analysis results dictionary
        output_path: Path to save the report
    """
    report = generate_report(results)

    # Ensure directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[INFO] Report saved to {output_path}")


def save_summary(all_results: List[Dict[str, Any]], output_path: str, format: str = "json") -> None:
    """
    Save summary of multiple analysis results

    Args:
        all_results: List of analysis results
        output_path: Path to save the summary
        format: Output format ('json' or 'markdown')
    """
    # Ensure directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    if format == "markdown":
        summary = generate_summary_report(all_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    else:
        # JSON format
        summary = {
            "total_files": len(all_results),
            "files_with_errors": sum(1 for r in all_results if "error" in r),
            "total_issues": sum(
                len(r.get("syntax_errors", [])) +
                len(r.get("bugs", [])) +
                len(r.get("memory_issues", [])) +
                len(r.get("security_vulnerabilities", [])) +
                len(r.get("performance_issues", [])) +
                len(r.get("style_issues", []))
                for r in all_results if "error" not in r
            ),
            "file_results": all_results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Summary saved to {output_path}")


def export_html(results: Dict[str, Any], output_path: str) -> None:
    """
    Export analysis results as an HTML file

    Args:
        results: Analysis results dictionary
        output_path: Path to save the HTML file
    """
    # Generate basic HTML report
    language = results.get('language', 'Unknown')
    file_path = results.get('file_path', 'Unknown')

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>Code Analysis Report - {file_path}</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
        "        h1 { color: #333; }",
        "        h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }",
        "        .issue { margin: 15px 0; padding: 10px; border-radius: 5px; }",
        "        .critical { background-color: #ffcdd2; border-left: 4px solid #f44336; }",
        "        .high { background-color: #ffe0b2; border-left: 4px solid #ff9800; }",
        "        .medium { background-color: #fff9c4; border-left: 4px solid #ffeb3b; }",
        "        .low { background-color: #c8e6c9; border-left: 4px solid #4caf50; }",
        "        .info { background-color: #e3f2fd; border-left: 4px solid #2196f3; }",
        "        .recommendation { color: #555; font-style: italic; margin-top: 5px; }",
        "        code { background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Code Analysis Report</h1>",
        f"    <p><strong>File:</strong> {file_path}</p>",
        f"    <p><strong>Language:</strong> {language}</p>",
    ]

    if "error" in results:
        html_parts.append(f"    <p><strong>Error:</strong> {results['error']}</p>")
    else:
        # Add sections
        sections = [
            ("Syntax Errors", "syntax_errors"),
            ("Bugs and Logical Errors", "bugs"),
            ("Memory Management Issues", "memory_issues"),
            ("Security Vulnerabilities", "security_vulnerabilities"),
            ("Performance Issues", "performance_issues"),
            ("Code Style and Readability", "style_issues")
        ]

        for title, key in sections:
            issues = results.get(key, [])
            html_parts.append(f"    <h2>{title} ({len(issues)})</h2>")

            if issues:
                for issue in issues:
                    severity = issue.get('severity', 'info')
                    line = issue.get('line', 0)
                    description = issue.get('description', 'No description')
                    recommendation = issue.get('recommendation', 'No recommendation')

                    html_parts.append(f"    <div class='issue {severity}'>")
                    html_parts.append(f"        <strong>Line {line}</strong> ({severity}): {description}")
                    html_parts.append(f"        <div class='recommendation'>Recommendation: {recommendation}</div>")
                    html_parts.append("    </div>")
            else:
                html_parts.append("    <p>No issues found.</p>")

    html_parts.extend([
        "</body>",
        "</html>"
    ])

    # Ensure directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    print(f"[INFO] HTML report saved to {output_path}")
