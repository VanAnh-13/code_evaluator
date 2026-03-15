"""
Report generator for Code Evaluator
Generates human-readable reports from analysis results
"""

from typing import Dict, Any


def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from analysis results

    Args:
        results: Analysis results dictionary

    Returns:
        Formatted report as string (Markdown format)
    """
    language = results.get('language', 'Unknown')
    report = [
        f"# Code Analysis Report\n",
        f"File: {results.get('file_path', 'Unknown')}\n",
        f"Language: {language}\n"
    ]

    # Score and summary (new API format)
    score = results.get('overall_score')
    if score is not None:
        report.append(f"Overall Score: **{score}/100**\n")

    summary = results.get('summary')
    if summary:
        report.append(f"> {summary}\n")

    if "error" in results:
        report.append(f"Error: {results['error']}\n")
        return "\n".join(report)

    # Add summary
    total_issues = (
        len(results.get("syntax_errors", [])) +
        len(results.get("bugs", [])) +
        len(results.get("memory_issues", [])) +
        len(results.get("security_vulnerabilities", [])) +
        len(results.get("performance_issues", [])) +
        len(results.get("style_issues", []))
    )
    report.append(f"Total issues found: {total_issues}\n")

    # Add syntax errors section
    syntax_errors = results.get("syntax_errors", [])
    report.append(_format_section("Syntax Errors", syntax_errors))

    # Add detailed sections
    sections = [
        ("Bugs and Logical Errors", "bugs"),
        ("Memory Management Issues", "memory_issues"),
        ("Security Vulnerabilities", "security_vulnerabilities"),
        ("Performance Issues", "performance_issues"),
        ("Code Style and Readability", "style_issues")
    ]

    for title, key in sections:
        issues = results.get(key, [])
        report.append(_format_section(title, issues))

    # Add suggested fixes section if available
    fixes = results.get("suggested_fixes")
    if fixes:
        report.append(f"## Suggested Fixes\n")
        if isinstance(fixes, list):
            for fix in fixes:
                line = fix.get("line", "?")
                explanation = fix.get("explanation", "")
                original = fix.get("original", "")
                fixed = fix.get("fixed", "")
                report.append(f"### Line {line}: {explanation}\n")
                if original:
                    report.append(f"**Original:**\n```\n{original}\n```\n")
                if fixed:
                    report.append(f"**Fixed:**\n```\n{fixed}\n```\n")
        elif isinstance(fixes, dict):
            for line_num, fix in fixes.items():
                report.append(f"### Line {line_num}:\n")
                report.append("```\n")
                report.append(f"{fix}\n")
                report.append("```\n")

    return "\n".join(report)


def _format_section(title: str, issues: list) -> str:
    """
    Format a section of the report

    Args:
        title: Section title
        issues: List of issues

    Returns:
        Formatted section string
    """
    result = [f"## {title} ({len(issues)})\n"]

    if issues:
        for issue in issues:
            line = issue.get('line', 0)
            severity = issue.get('severity', 'unknown')
            description = issue.get('description', 'No description')
            recommendation = issue.get('recommendation', 'No recommendation')

            result.append(f"- Line {line} ({severity}): {description}")
            result.append(f"  Recommendation: {recommendation}\n")
    else:
        result.append("No issues found.\n")

    return "\n".join(result)


def generate_summary_report(all_results: list) -> str:
    """
    Generate a summary report for multiple files

    Args:
        all_results: List of analysis results for multiple files

    Returns:
        Formatted summary report as string
    """
    report = ["# Code Analysis Summary\n"]

    total_files = len(all_results)
    files_with_errors = sum(1 for r in all_results if "error" in r)

    total_issues = sum(
        len(r.get("syntax_errors", [])) +
        len(r.get("bugs", [])) +
        len(r.get("memory_issues", [])) +
        len(r.get("security_vulnerabilities", [])) +
        len(r.get("performance_issues", [])) +
        len(r.get("style_issues", []))
        for r in all_results if "error" not in r
    )

    report.append(f"**Total files analyzed:** {total_files}\n")
    report.append(f"**Files with errors:** {files_with_errors}\n")
    report.append(f"**Total issues found:** {total_issues}\n")

    # Per-file summary
    report.append("\n## Files Summary\n")
    for result in all_results:
        file_path = result.get("file_path", "Unknown")
        if "error" in result:
            report.append(f"- {file_path}: **Error** - {result['error']}\n")
        else:
            file_issues = (
                len(result.get("syntax_errors", [])) +
                len(result.get("bugs", [])) +
                len(result.get("memory_issues", [])) +
                len(result.get("security_vulnerabilities", [])) +
                len(result.get("performance_issues", [])) +
                len(result.get("style_issues", []))
            )
            report.append(f"- {file_path}: {file_issues} issues\n")

    return "\n".join(report)
