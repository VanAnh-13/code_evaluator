"""
Fix suggester for Code Evaluator
Generates suggested fixes for identified code issues
"""

import re
from typing import Dict, List, Any


def suggest_fixes(code: str, issues: List[Dict[str, Any]], language: str) -> Dict[int, str]:
    """
    Generate suggested fixes for the identified issues

    Args:
        code: Original code
        issues: List of issues identified in the code
        language: Programming language of the code

    Returns:
        Dictionary mapping line numbers to suggested fixed code
    """
    lines = code.splitlines()
    fixes = {}

    if language == "cpp":
        fixes = _suggest_cpp_fixes(lines, issues)
    elif language == "python":
        fixes = _suggest_python_fixes(lines, issues)
    # Add more language-specific fixes as needed

    return fixes


def _suggest_cpp_fixes(lines: List[str], issues: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Generate C++ specific fix suggestions

    Args:
        lines: List of code lines
        issues: List of issues identified in the code

    Returns:
        Dictionary mapping line numbers to suggested fixed code
    """
    fixes = {}

    for issue in issues:
        line_num = issue.get("line", 0)
        if line_num <= 0 or line_num > len(lines):
            continue

        original_line = lines[line_num - 1]
        description = issue.get("description", "")

        # Generate fixes based on issue type
        if "null pointer dereference" in description.lower():
            if "ptr" in original_line or "*" in original_line:
                var_match = re.search(r'(\w+)\s*->', original_line) or re.search(r'(\w+)\s*\*', original_line)
                if var_match:
                    var_name = var_match.group(1)
                    indentation = len(original_line) - len(original_line.lstrip())
                    fixed_line = " " * indentation + f"if ({var_name} != nullptr) {{\n{original_line}\n" + " " * indentation + "}"
                    fixes[line_num] = fixed_line

        elif "memory leak" in description.lower():
            if "new" in original_line and "delete" not in original_line:
                var_match = re.search(r'(\w+)\s*=\s*new', original_line)
                if var_match:
                    var_name = var_match.group(1)
                    indentation = len(original_line) - len(original_line.lstrip())
                    if "[]" in original_line:
                        fixes[line_num + 1] = " " * indentation + f"delete[] {var_name};"
                    else:
                        fixes[line_num + 1] = " " * indentation + f"delete {var_name};"

        elif "buffer overflow" in description.lower():
            if "strcpy" in original_line:
                fixed_line = original_line.replace("strcpy", "strncpy")
                if "," in fixed_line and ");" in fixed_line:
                    fixed_line = fixed_line.replace(");", ", sizeof(destination));")
                fixes[line_num] = fixed_line

        elif "inefficient loop" in description.lower():
            if "for" in original_line and "i++" in original_line and ".size()" in original_line:
                container_match = re.search(r'(\w+)\.size\(\)', original_line)
                if container_match:
                    container_name = container_match.group(1)
                    indentation = len(original_line) - len(original_line.lstrip())
                    fixed_line = " " * indentation + f"for (const auto& element : {container_name}) {{"
                    fixes[line_num] = fixed_line

    return fixes


def _suggest_python_fixes(lines: List[str], issues: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Generate Python specific fix suggestions

    Args:
        lines: List of code lines
        issues: List of issues identified in the code

    Returns:
        Dictionary mapping line numbers to suggested fixed code
    """
    fixes = {}

    for issue in issues:
        line_num = issue.get("line", 0)
        if line_num <= 0 or line_num > len(lines):
            continue

        original_line = lines[line_num - 1]
        description = issue.get("description", "")

        # Python-specific fixes
        if "undefined variable" in description.lower():
            var_match = re.search(r'variable\s+\'(\w+)\'', description)
            if var_match:
                var_name = var_match.group(1)
                indentation = len(original_line) - len(original_line.lstrip())
                fixes[line_num] = " " * indentation + f"{var_name} = None  # Initialize variable"

        elif "indentation error" in description.lower():
            indentation = len(original_line) - len(original_line.lstrip())
            if indentation % 4 != 0:
                new_indentation = (indentation // 4 + 1) * 4
                fixes[line_num] = " " * new_indentation + original_line.lstrip()

    return fixes


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
