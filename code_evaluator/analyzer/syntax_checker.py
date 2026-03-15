"""
Syntax checker for Code Evaluator
Provides language-specific syntax checking using external tools
"""

import os
import re
import subprocess
import tempfile
from typing import Dict, List, Any


def analyze_cpp_code(code: str) -> List[Dict[str, Any]]:
    """
    Perform basic static analysis on C++ code to identify common errors

    Args:
        code: C++ code to analyze

    Returns:
        List of detected issues
    """
    issues = []
    lines = code.splitlines()

    for i, line in enumerate(lines):
        line_num = i + 1

        # Check for missing type in function parameters
        if re.search(r'(\w+)\s*\([^)]*\b\w+\b(?!\s*:|\s*\w+)[^)]*\)', line):
            param_match = re.search(r'\(([^)]+)\)', line)
            if param_match:
                params = param_match.group(1).split(',')
                for param in params:
                    param = param.strip()
                    if param and not re.match(r'^\s*\w+\s+\w+', param) and not param.startswith('...'):
                        issues.append({
                            "line": line_num,
                            "severity": "high",
                            "description": "Missing type for parameter in function declaration",
                            "recommendation": "Add the appropriate type for the parameter"
                        })

        # Check for string assigned to int
        if re.search(r'int\s+\w+\s*=\s*"', line):
            issues.append({
                "line": line_num,
                "severity": "high",
                "description": "Assigning string literal to integer variable",
                "recommendation": "Use an integer value or convert the string to an integer"
            })

        # Check for potential buffer overflow in array declarations
        if re.search(r'char\s+\w+\s*\[\s*\d+\s*\]', line) and 'cin' in lines[min(i+1, len(lines)-1)]:
            issues.append({
                "line": line_num,
                "severity": "high",
                "description": "Potential buffer overflow with fixed-size char array",
                "recommendation": "Use std::string instead of char array or ensure input doesn't exceed buffer size"
            })

        # Check for assignment in if condition (common mistake)
        if re.search(r'if\s*\(\s*\w+\s*=\s*\w+\s*\)', line):
            issues.append({
                "line": line_num,
                "severity": "high",
                "description": "Assignment operator used in if condition instead of equality operator",
                "recommendation": "Use == for comparison instead of = (assignment)"
            })

        # Check for off-by-one errors in for loops
        if re.search(r'for\s*\([^;]*;\s*\w+\s*<=\s*\d+\s*;', line):
            issues.append({
                "line": line_num,
                "severity": "medium",
                "description": "Potential off-by-one error in for loop condition using <=",
                "recommendation": "Consider using < instead of <= to avoid off-by-one errors"
            })

        # Check for null pointer dereference
        if re.search(r'\*\s*\w+', line) and 'nullptr' in line:
            issues.append({
                "line": line_num,
                "severity": "critical",
                "description": "Potential null pointer dereference",
                "recommendation": "Add a null check before dereferencing the pointer"
            })

    return issues


def generate_fix_suggestion(code: str, line_num: int, error_message: str, language: str) -> str:
    """
    Generate a suggestion to fix the syntax error

    Args:
        code: The original code
        line_num: Line number with the error
        error_message: Error message from the compiler/linter
        language: Programming language of the code

    Returns:
        A suggestion to fix the error
    """
    lines = code.splitlines()
    if line_num <= 0 or line_num > len(lines):
        return "Review the code structure"

    error_line = lines[line_num - 1]

    # Language-specific error patterns and suggested fixes
    if language == "cpp":
        if "expected ';'" in error_message:
            return f"Add a semicolon at the end of the line: `{error_line};`"
        elif "undeclared identifier" in error_message:
            identifier = re.search(r"'(\w+)'", error_message)
            if identifier:
                return f"Declare variable '{identifier.group(1)}' before using it"
        elif "expected '}'" in error_message:
            return "Add a closing curly brace '}' to match an opening brace"
        elif "expected ')'" in error_message:
            return "Add a closing parenthesis ')' to match an opening parenthesis"
        elif "no matching function for call to" in error_message:
            return "Check function arguments or ensure the function is declared before use"
        elif "invalid operands" in error_message or "no operator" in error_message:
            return "Ensure operand types are compatible or add appropriate type conversion"
        elif "redefinition of" in error_message:
            identifier = re.search(r"'(\w+)'", error_message)
            if identifier:
                return f"Remove duplicate definition of '{identifier.group(1)}' or use a different name"
        elif "cannot convert" in error_message:
            return "Add explicit type conversion or modify the types to be compatible"
        elif "expected initializer" in error_message:
            return "Add an initializer to the variable declaration"

    elif language == "python":
        if "SyntaxError: invalid syntax" in error_message:
            return "Check for missing colons, parentheses, or brackets"
        elif "IndentationError" in error_message:
            return "Fix the indentation to use consistent spaces (usually 4 spaces per level)"
        elif "NameError: name" in error_message and "is not defined" in error_message:
            identifier = re.search(r"'(\w+)'", error_message)
            if identifier:
                return f"Define variable '{identifier.group(1)}' before using it"
        elif "TypeError: unsupported operand type" in error_message:
            return "Ensure operand types are compatible or add appropriate type conversion"
        elif "ImportError" in error_message:
            return "Check that the module is installed and the import statement is correct"

    # Default suggestion
    return f"Review the line and fix the syntax according to {language.upper()} rules"


def check_syntax(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Check code for syntax errors using appropriate tools for the language

    Args:
        code: Code to check
        language: Programming language of the code

    Returns:
        List of syntax errors with line numbers and descriptions
    """
    syntax_errors = []

    # Create a temporary file to store the code
    suffix_map = {
        'cpp': '.cpp',
        'python': '.py',
        'javascript': '.js',
        'java': '.java',
        'c': '.c',
    }
    suffix = suffix_map.get(language, f'.{language}')
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))

    try:
        if language == "cpp":
            syntax_errors = _check_cpp_syntax(code, temp_file_path)
        elif language == "python":
            syntax_errors = _check_python_syntax(code, temp_file_path)
        # Add more language-specific syntax checking as needed

    except subprocess.TimeoutExpired:
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "Syntax check timed out",
            "recommendation": "The code might be too complex or contain infinite loops"
        })
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return syntax_errors


def _check_cpp_syntax(code: str, temp_file_path: str) -> List[Dict[str, Any]]:
    """
    Check C++ code for syntax errors using g++

    Args:
        code: C++ code
        temp_file_path: Path to temporary file containing the code

    Returns:
        List of syntax errors
    """
    syntax_errors = []

    try:
        result = subprocess.run(
            ['g++', '-fsyntax-only', '-Wall', '-Wextra', temp_file_path],
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            error_pattern = re.compile(r'([^:]+):(\d+):(\d+):\s+(warning|error):\s+(.+)')
            for line in result.stderr.splitlines():
                match = error_pattern.match(line)
                if match:
                    _, line_num, column, error_type, message = match.groups()
                    severity = "high" if error_type == "error" else "medium"
                    syntax_errors.append({
                        "line": int(line_num),
                        "column": int(column),
                        "severity": severity,
                        "description": f"Syntax {error_type}: {message}",
                        "recommendation": generate_fix_suggestion(code, int(line_num), message, "cpp")
                    })

    except FileNotFoundError:
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "G++ compiler not found",
            "recommendation": "Install g++ to enable C++ syntax checking"
        })
        # Fallback to basic static analysis for C++ if compiler not found
        cpp_issues = analyze_cpp_code(code)
        syntax_errors.extend(cpp_issues)

    return syntax_errors


def _check_python_syntax(code: str, temp_file_path: str) -> List[Dict[str, Any]]:
    """
    Check Python code for syntax errors using pylint

    Args:
        code: Python code
        temp_file_path: Path to temporary file containing the code

    Returns:
        List of syntax errors
    """
    syntax_errors = []

    try:
        result = subprocess.run(
            ['pylint', '--output-format=text', temp_file_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=10
        )

        error_pattern = re.compile(r'([^:]+):(\d+):(\d+):\s+([CEFIRW]\d+):\s+(.+)')
        for line in result.stdout.splitlines():
            match = error_pattern.match(line)
            if match:
                _, line_num, column, error_code, message = match.groups()

                # Determine severity based on pylint error code
                severity = "info"
                if error_code.startswith('E') or error_code.startswith('F'):
                    severity = "high"
                elif error_code.startswith('W'):
                    severity = "medium"

                syntax_errors.append({
                    "line": int(line_num),
                    "column": int(column),
                    "severity": severity,
                    "description": f"Syntax issue: {message} ({error_code})",
                    "recommendation": generate_fix_suggestion(code, int(line_num), message, "python")
                })

    except FileNotFoundError:
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "Pylint not found",
            "recommendation": "Install pylint to enable Python syntax checking"
        })

    return syntax_errors
