"""
C++ Code Analyzer using Qwen model
This program analyzes C++ code for potential issues, code quality, and security vulnerabilities
"""

import os
import sys
import argparse
import json
import subprocess
import tempfile
import re
from typing import Dict, List, Any, Optional, Tuple


def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from analysis results

    Args:
        results: Analysis results

    Returns:
        Formatted report as string
    """
    report = [f"# C++ Code Analysis Report\n", f"File: {results.get('file_path', 'Unknown')}\n"]

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
    if syntax_errors:
        report.append(f"## Syntax Errors ({len(syntax_errors)})\n")
        for issue in syntax_errors:
            report.append(f"- Line {issue['line']} ({issue['severity']}): {issue['description']}")
            report.append(f"  Recommendation: {issue['recommendation']}\n")
    else:
        report.append(f"## Syntax Errors (0)\n")
        report.append("No syntax errors found.\n")

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
        if issues:
            report.append(f"## {title} ({len(issues)})\n")
            for issue in issues:
                report.append(f"- Line {issue['line']} ({issue['severity']}): {issue['description']}")
                report.append(f"  Recommendation: {issue['recommendation']}\n")
        else:
            report.append(f"## {title} (0)\n")
            report.append("No issues found.\n")

    # Add suggested fixes section if available
    if "suggested_fixes" in results and results["suggested_fixes"]:
        report.append(f"## Suggested Fixes\n")
        for line_num, fix in results["suggested_fixes"].items():
            report.append(f"### Line {line_num}:\n")
            report.append("```cpp\n")
            report.append(f"{fix}\n")
            report.append("```\n")

    return "\n".join(report)


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save analysis results to a file

    Args:
        results: Analysis results
        output_path: Path to save the results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {output_path}")


def suggest_fixes(code: str, issues: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Generate suggested fixes for the identified issues

    Args:
        code: Original C++ code
        issues: List of issues identified in the code

    Returns:
        Dictionary mapping line numbers to suggested fixed code
    """
    lines = code.splitlines()
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

        elif "inconsistent naming" in description.lower():
            # This would require more context to fix properly
            pass

    return fixes


def generate_fix_suggestion(code: str, line_num: int, error_message: str) -> str:
    """
    Generate a suggestion to fix the syntax error

    Args:
        code: The original C++ code
        line_num: Line number with the error
        error_message: Error message from the compiler

    Returns:
        A suggestion to fix the error
    """
    lines = code.splitlines()
    if line_num <= 0 or line_num > len(lines):
        return "Review the code structure"

    error_line = lines[line_num - 1]

    # Common error patterns and suggested fixes
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

    # Default suggestion
    return "Review the line and fix the syntax according to C++ rules"


def check_syntax(code: str) -> List[Dict[str, Any]]:
    """
    Check C++ code for syntax errors using g++ compiler

    Args:
        code: C++ code to check

    Returns:
        List of syntax errors with line numbers and descriptions
    """
    syntax_errors = []

    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))

    try:
        # Run g++ with -fsyntax-only to check syntax without generating output
        result = subprocess.run(
            ['g++', '-fsyntax-only', '-Wall', '-Wextra', temp_file_path],
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        # Parse compiler errors
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
                        "recommendation": generate_fix_suggestion(code, int(line_num), message)
                    })
    except subprocess.TimeoutExpired:
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "Syntax check timed out",
            "recommendation": "The code might be too complex or contain infinite loops"
        })
    except FileNotFoundError:
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "G++ compiler not found",
            "recommendation": "Install g++ to enable syntax checking"
        })
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return syntax_errors


class CppCodeAnalyzer:
    """
    A class to analyze C++ code using the Qwen model
    """

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat"):
        """
        Initialize the analyzer with the specified model

        Args:
            model_name: Name of the Qwen model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._cache = {}  # Cache for analyzed files
        self.prompt_template = """
        You are an expert C++ code analyzer. Analyze the following C++ code for:
        1. Potential bugs and logical errors
        2. Memory management issues (leaks, dangling pointers, etc.)
        3. Security vulnerabilities
        4. Performance issues
        5. Code style and readability issues

        Provide a detailed analysis with specific line numbers and recommendations for improvement.

        C++ CODE:
        ```cpp
        {code}
        ```

        ANALYSIS:
        """

    def load_model(self):
        """
        Load the Qwen model (placeholder for actual implementation)

        Note: This would be implemented with transformers or modelscope in the full version
        """
        print(f"[INFO] Loading model: {self.model_name}")
        # In the full implementation, this would load the model using:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # return model, tokenizer

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze the given C++ code

        Args:
            code: C++ code to analyze

        Returns:
            Dictionary containing analysis results
        """
        # Check for syntax errors first
        syntax_errors = check_syntax(code)

        # Prepare the prompt
        formatted_prompt = self.prompt_template.format(code=code)

        # In the full implementation, this would:
        # 1. Tokenize the prompt
        # 2. Generate a response using the model
        # 3. Parse the response into structured analysis

        # For this placeholder, we'll return a sample analysis
        analysis_results = {
            "syntax_errors": syntax_errors,
            "bugs": [
                {"line": 15, "severity": "high", "description": "Potential null pointer dereference",
                 "recommendation": "Add null check before dereferencing pointer"}
            ],
            "memory_issues": [
                {"line": 23, "severity": "critical", "description": "Memory leak: allocated memory not freed",
                 "recommendation": "Add delete[] or use smart pointers"}
            ],
            "security_vulnerabilities": [
                {"line": 42, "severity": "medium", "description": "Buffer overflow risk in strcpy",
                 "recommendation": "Use strncpy or std::string instead"}
            ],
            "performance_issues": [
                {"line": 57, "severity": "low", "description": "Inefficient loop implementation",
                 "recommendation": "Consider using std::transform or range-based for loop"}
            ],
            "style_issues": [
                {"line": 10, "severity": "info", "description": "Inconsistent naming convention",
                 "recommendation": "Follow a consistent naming style (e.g., camelCase or snake_case)"}
            ]
        }

        return analysis_results

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a C++ file

        Args:
            file_path: Path to the C++ file

        Returns:
            Dictionary containing analysis results
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "file_path": file_path}

        # Check if file is a C++ file
        if not file_path.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
            print(f"[WARNING] File {file_path} does not have a C++ extension")

        # Get file modification time for cache validation
        try:
            mtime = os.path.getmtime(file_path)

            # Check cache first
            cache_key = f"{file_path}_{mtime}"
            if cache_key in self._cache:
                print(f"[INFO] Using cached analysis for {file_path}")
                return self._cache[cache_key]

            # Read and analyze the file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()

                results = self.analyze_code(code)
                results["file_path"] = file_path

                # Cache the results
                self._cache[cache_key] = results
                return results
            except IOError as e:
                return {"error": f"IO Error: {str(e)}", "file_path": file_path}
            except UnicodeDecodeError as e:
                return {"error": f"Encoding error: {str(e)}", "file_path": file_path}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}", "file_path": file_path}
        except Exception as e:
            return {"error": f"Error accessing file: {str(e)}", "file_path": file_path}


def main():
    """Main function to run the analyzer from command line"""
    parser = argparse.ArgumentParser(description="Analyze C++ code using Qwen model")
    parser.add_argument("files", nargs="*", default=None, help="Path to C++ file(s) to analyze")
    parser.add_argument("--model", default="Qwen/Qwen-7B-Chat", help="Qwen model to use")
    parser.add_argument("--output", help="Directory to save analysis results (JSON)")
    parser.add_argument("--report", help="Directory to save human-readable reports (Markdown)")
    parser.add_argument("--fix", action="store_true", help="Generate suggested fixes for identified issues")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of analysis results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Handle case when no files are provided
    if not args.files:
        sample_path = "examples/example.cpp"
        if os.path.exists(sample_path):
            args.files = [sample_path]
            print(f"[INFO] No files provided, using sample file: {sample_path}")
        else:
            parser.print_help()
            sys.exit(1)

    # Initialize analyzer
    analyzer = CppCodeAnalyzer(model_name=args.model)
    analyzer.load_model()

    # Process each file
    all_results = []
    for file_path in args.files:
        if args.verbose:
            print(f"[INFO] Analyzing {file_path}...")

        # Analyze the file
        results = analyzer.analyze_file(file_path)

        # Store the file content if we'll need it for fixes
        file_content = None

        # Generate suggested fixes if requested
        if args.fix and "error" not in results:
            try:
                # Only read the file if we haven't already
                if file_content is None:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            file_content = f.read()
                    except Exception as e:
                        print(f"[ERROR] Failed to read file for fixes: {str(e)}")
                        continue

                # Collect all issues
                all_issues = []
                for key in ["syntax_errors", "bugs", "memory_issues", "security_vulnerabilities", "performance_issues"]:
                    all_issues.extend(results.get(key, []))

                # Generate fixes
                suggested_fixes = suggest_fixes(file_content, all_issues)
                results["suggested_fixes"] = {str(k): v for k, v in suggested_fixes.items()}

                if args.verbose:
                    print(f"[INFO] Generated {len(suggested_fixes)} suggested fixes for {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to generate fixes for {file_path}: {str(e)}")

        all_results.append(results)

        # Save individual results if output directory is specified
        if args.output:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_analysis.json")
            save_results(results, output_path)

        # Generate and save individual report if report directory is specified
        report = generate_report(results)
        if args.report:
            report_dir = args.report
            os.makedirs(report_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            report_path = os.path.join(report_dir, f"{os.path.splitext(base_name)[0]}_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            if args.verbose:
                print(f"[INFO] Report saved to {report_path}")
        else:
            print(report)
            print("\n" + "-" * 80 + "\n")  # Separator between reports

    # If multiple files were analyzed, save a summary
    if len(all_results) > 1 and args.output:
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
        summary_path = os.path.join(args.output, "analysis_summary.json")
        save_results(summary, summary_path)
        if args.verbose:
            print(f"[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
