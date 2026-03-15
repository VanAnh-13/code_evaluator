"""
Built-in Agent Tools
Registers all available tools that the AI agent can call during analysis.
Wraps existing functions from the codebase as agent-callable tools.
"""

import json
import os
import re
import glob
import logging
from typing import Dict, List, Any

from code_evaluator.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Global default registry
default_registry = ToolRegistry()


def create_default_registry() -> ToolRegistry:
    """Create and return a ToolRegistry with all built-in tools registered."""
    registry = ToolRegistry()
    _register_all_tools(registry)
    return registry


def _register_all_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools on the given registry."""

    # ── 1. check_syntax ──────────────────────────────────────────────────
    registry.register(
        name="check_syntax",
        description=(
            "Run syntax checking on code using language-specific tools "
            "(g++ for C/C++, pylint for Python, etc.). "
            "Returns a list of syntax errors with line numbers and descriptions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to check for syntax errors.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (e.g., 'cpp', 'python', 'javascript').",
                },
            },
            "required": ["code", "language"],
        },
        handler=_tool_check_syntax,
    )

    # ── 2. analyze_patterns ──────────────────────────────────────────────
    registry.register(
        name="analyze_patterns",
        description=(
            "Perform static pattern-based analysis on code to detect common "
            "issues like null pointer dereference, buffer overflow, assignment "
            "in conditions, off-by-one errors, etc. Works without external compilers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to analyze.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code.",
                },
            },
            "required": ["code", "language"],
        },
        handler=_tool_analyze_patterns,
    )

    # ── 3. suggest_fixes ─────────────────────────────────────────────────
    registry.register(
        name="suggest_fixes",
        description=(
            "Generate fix suggestions for identified issues in the code. "
            "Provide the code, a list of issues, and the language."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Original source code.",
                },
                "issues": {
                    "type": "string",
                    "description": "JSON array of issue objects, each with 'line', 'description', and 'severity' fields.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language.",
                },
            },
            "required": ["code", "issues", "language"],
        },
        handler=_tool_suggest_fixes,
    )

    # ── 4. apply_fix ─────────────────────────────────────────────────────
    registry.register(
        name="apply_fix",
        description=(
            "Apply a text replacement fix to code. "
            "Replace specific text on a given line with new text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Original source code.",
                },
                "line_number": {
                    "type": "integer",
                    "description": "1-based line number to modify.",
                },
                "original_text": {
                    "type": "string",
                    "description": "The text on that line to replace.",
                },
                "replacement_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["code", "line_number", "original_text", "replacement_text"],
        },
        handler=_tool_apply_fix,
    )

    # ── 5. verify_fix ────────────────────────────────────────────────────
    registry.register(
        name="verify_fix",
        description=(
            "Verify a fixed version of code by running syntax checking on it. "
            "Returns whether the fix is valid (no new syntax errors introduced)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fixed_code": {
                    "type": "string",
                    "description": "The code after applying the fix.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language.",
                },
            },
            "required": ["fixed_code", "language"],
        },
        handler=_tool_verify_fix,
    )

    # ── 6. read_file ─────────────────────────────────────────────────────
    registry.register(
        name="read_file",
        description=(
            "Read the contents of a source code file from the project. "
            "Returns the file content or an error message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
            },
            "required": ["file_path"],
        },
        handler=_tool_read_file,
    )

    # ── 7. list_directory ────────────────────────────────────────────────
    registry.register(
        name="list_directory",
        description=(
            "List files and directories in the given directory path. "
            "Returns a structured listing with file types."
        ),
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to list.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter (e.g., '*.py'). Default lists all.",
                },
            },
            "required": ["directory"],
        },
        handler=_tool_list_directory,
    )

    # ── 8. search_code ───────────────────────────────────────────────────
    registry.register(
        name="search_code",
        description=(
            "Search for a text pattern (regex supported) across files in a directory. "
            "Returns matching lines with file paths and line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob file pattern (e.g., '*.py'). Default: all text files.",
                },
            },
            "required": ["pattern", "directory"],
        },
        handler=_tool_search_code,
    )

    # ── 9. detect_language ───────────────────────────────────────────────
    registry.register(
        name="detect_language",
        description="Detect the programming language from a file path based on extension.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file.",
                },
            },
            "required": ["file_path"],
        },
        handler=_tool_detect_language,
    )

    # ── 10. analyze_dependencies ─────────────────────────────────────────
    registry.register(
        name="analyze_dependencies",
        description=(
            "Analyze import/include statements in code to understand dependencies. "
            "Returns a list of imported modules or headers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Source code to analyze.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language.",
                },
            },
            "required": ["code", "language"],
        },
        handler=_tool_analyze_dependencies,
    )

    # ── 11. get_analysis_summary ─────────────────────────────────────────
    registry.register(
        name="get_analysis_summary",
        description=(
            "Compile a summary of all tool results gathered so far during this agent session. "
            "Pass in the collected results as a JSON string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "collected_results": {
                    "type": "string",
                    "description": "JSON string of collected analysis results from previous tool calls.",
                },
            },
            "required": ["collected_results"],
        },
        handler=_tool_get_analysis_summary,
    )


# =====================================================================
# Tool handler implementations
# =====================================================================


def _tool_check_syntax(code: str, language: str) -> str:
    """Run syntax checking on code."""
    from code_evaluator.analyzer.syntax_checker import check_syntax

    errors = check_syntax(code, language)
    if not errors:
        return json.dumps({"status": "ok", "message": "No syntax errors found.", "errors": []})
    return json.dumps({"status": "errors_found", "error_count": len(errors), "errors": errors})


def _tool_analyze_patterns(code: str, language: str) -> str:
    """Run static pattern analysis."""
    from code_evaluator.analyzer.syntax_checker import analyze_cpp_code

    if language in ("cpp", "c"):
        issues = analyze_cpp_code(code)
    else:
        # Basic pattern analysis for other languages
        issues = _basic_pattern_analysis(code, language)

    if not issues:
        return json.dumps({"status": "ok", "message": "No pattern-based issues detected.", "issues": []})
    return json.dumps({"status": "issues_found", "issue_count": len(issues), "issues": issues})


def _basic_pattern_analysis(code: str, language: str) -> List[Dict]:
    """Basic pattern analysis for non-C++ languages."""
    issues = []
    lines = code.splitlines()

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()

        # Common patterns across languages
        if "TODO" in line or "FIXME" in line or "HACK" in line:
            issues.append({
                "line": line_num,
                "severity": "info",
                "description": f"Found TODO/FIXME/HACK comment: {stripped[:80]}",
                "recommendation": "Address or remove the TODO/FIXME comment.",
            })

        if language == "python":
            # Bare except
            if re.match(r'\s*except\s*:', line):
                issues.append({
                    "line": line_num,
                    "severity": "medium",
                    "description": "Bare except clause catches all exceptions including SystemExit and KeyboardInterrupt.",
                    "recommendation": "Use 'except Exception:' or catch specific exceptions.",
                })
            # eval() usage
            if "eval(" in line:
                issues.append({
                    "line": line_num,
                    "severity": "high",
                    "description": "Use of eval() is a security risk.",
                    "recommendation": "Use ast.literal_eval() or avoid eval entirely.",
                })

        elif language == "javascript":
            # == instead of ===
            if re.search(r'[^=!]==[^=]', line):
                issues.append({
                    "line": line_num,
                    "severity": "medium",
                    "description": "Use of == instead of === for comparison.",
                    "recommendation": "Use === for strict equality comparison.",
                })
            # var usage
            if re.match(r'\s*var\s+', line):
                issues.append({
                    "line": line_num,
                    "severity": "low",
                    "description": "Use of 'var' instead of 'let' or 'const'.",
                    "recommendation": "Use 'const' for values that don't change, 'let' otherwise.",
                })

    return issues


def _tool_suggest_fixes(code: str, issues: str, language: str) -> str:
    """Generate fix suggestions."""
    from code_evaluator.analyzer.fix_suggester import suggest_fixes

    try:
        issues_list = json.loads(issues) if isinstance(issues, str) else issues
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in issues parameter."})

    fixes = suggest_fixes(code, issues_list, language)
    if not fixes:
        return json.dumps({"status": "no_fixes", "message": "No automatic fixes available.", "fixes": {}})
    return json.dumps({"status": "fixes_generated", "fix_count": len(fixes), "fixes": fixes})


def _tool_apply_fix(code: str, line_number: int, original_text: str, replacement_text: str) -> str:
    """Apply a fix to code at a specific line."""
    lines = code.splitlines()

    if line_number < 1 or line_number > len(lines):
        return json.dumps({"error": f"Line number {line_number} out of range (1-{len(lines)})."})

    target_line = lines[line_number - 1]
    if original_text not in target_line:
        return json.dumps({
            "error": f"Original text not found on line {line_number}.",
            "actual_line": target_line,
        })

    lines[line_number - 1] = target_line.replace(original_text, replacement_text, 1)
    fixed_code = "\n".join(lines)
    return json.dumps({"status": "fix_applied", "fixed_code": fixed_code})


def _tool_verify_fix(fixed_code: str, language: str) -> str:
    """Verify that a fix doesn't introduce new syntax errors."""
    from code_evaluator.analyzer.syntax_checker import check_syntax

    errors = check_syntax(fixed_code, language)
    if not errors:
        return json.dumps({
            "status": "verified",
            "message": "Fix is valid — no syntax errors in the fixed code.",
        })
    return json.dumps({
        "status": "verification_failed",
        "message": f"Fix introduced {len(errors)} syntax error(s).",
        "errors": errors,
    })


def _tool_read_file(file_path: str) -> str:
    """Read a file with security validation."""
    from code_evaluator.utils.security import validate_path
    from code_evaluator.utils.file_utils import read_file_content

    is_valid, error_msg = validate_path(file_path)
    if not is_valid:
        return json.dumps({"error": error_msg})

    content, error = read_file_content(file_path)
    if error:
        return json.dumps({"error": error})

    # Limit output size to avoid token overflow
    if len(content) > 50000:
        content = content[:50000] + "\n... (truncated, file too large)"

    return json.dumps({
        "file_path": file_path,
        "content": content,
        "line_count": len(content.splitlines()),
    })


def _tool_list_directory(directory: str, pattern: str = "*") -> str:
    """List directory contents."""
    from code_evaluator.utils.security import validate_path

    is_valid, error_msg = validate_path(directory)
    if not is_valid:
        return json.dumps({"error": error_msg})

    if not os.path.isdir(directory):
        return json.dumps({"error": f"Not a directory: {directory}"})

    entries = []
    try:
        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            is_dir = os.path.isdir(full_path)
            # Skip hidden and __pycache__ dirs
            if entry.startswith(".") or entry == "__pycache__":
                continue
            if is_dir or (pattern == "*" or glob.fnmatch.fnmatch(entry, pattern)):
                entries.append({
                    "name": entry,
                    "type": "directory" if is_dir else "file",
                    "size": os.path.getsize(full_path) if not is_dir else None,
                })
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({
        "directory": directory,
        "entry_count": len(entries),
        "entries": entries[:200],  # Limit to 200 entries
    })


def _tool_search_code(pattern: str, directory: str, file_pattern: str = "*") -> str:
    """Search for a pattern across files in a directory."""
    from code_evaluator.utils.security import validate_path

    is_valid, error_msg = validate_path(directory)
    if not is_valid:
        return json.dumps({"error": error_msg})

    matches = []
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {str(e)}"})

    code_extensions = {".py", ".cpp", ".c", ".h", ".hpp", ".js", ".ts", ".java", ".go", ".rs", ".rb"}

    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden and cache dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in code_extensions:
                    continue
                if file_pattern != "*" and not glob.fnmatch.fnmatch(filename, file_pattern):
                    continue

                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if compiled.search(line):
                                matches.append({
                                    "file": os.path.relpath(file_path, directory),
                                    "line": line_num,
                                    "text": line.rstrip()[:200],
                                })
                                if len(matches) >= 50:
                                    break
                except (IOError, OSError):
                    continue

            if len(matches) >= 50:
                break
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
        "truncated": len(matches) >= 50,
    })


def _tool_detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    from code_evaluator.utils.file_utils import detect_language

    language = detect_language(file_path)
    return json.dumps({"file_path": file_path, "language": language})


def _tool_analyze_dependencies(code: str, language: str) -> str:
    """Analyze import/include statements."""
    dependencies = []
    lines = code.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        line_num = i + 1

        if language in ("cpp", "c"):
            match = re.match(r'#include\s*[<"]([^>"]+)[>"]', stripped)
            if match:
                dependencies.append({
                    "line": line_num,
                    "type": "include",
                    "module": match.group(1),
                    "is_system": "<" in stripped.split("#include")[1] if "#include" in stripped else False,
                })

        elif language == "python":
            # import X / from X import Y
            match = re.match(r'(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)', stripped)
            if match:
                from_module = match.group(1) or ""
                imports = [s.strip() for s in match.group(2).split(",")]
                for imp in imports:
                    dependencies.append({
                        "line": line_num,
                        "type": "import",
                        "module": f"{from_module}.{imp}" if from_module else imp,
                        "from_module": from_module,
                    })

        elif language == "javascript":
            # import ... from 'module' / require('module')
            match = re.search(r"(?:import\s+.*\s+from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", stripped)
            if match:
                module_name = match.group(1) or match.group(2)
                dependencies.append({
                    "line": line_num,
                    "type": "import",
                    "module": module_name,
                    "is_local": module_name.startswith("."),
                })

        elif language == "java":
            match = re.match(r'import\s+([\w.]+);', stripped)
            if match:
                dependencies.append({
                    "line": line_num,
                    "type": "import",
                    "module": match.group(1),
                })

    return json.dumps({
        "language": language,
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
    })


def _tool_get_analysis_summary(collected_results: str) -> str:
    """Compile a summary from collected results."""
    try:
        results = json.loads(collected_results) if isinstance(collected_results, str) else collected_results
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in collected_results."})

    summary = {
        "total_items": len(results) if isinstance(results, list) else 1,
        "summary": "Analysis results compiled.",
        "results": results,
    }
    return json.dumps(summary)


# Initialize the default registry
_register_all_tools(default_registry)
