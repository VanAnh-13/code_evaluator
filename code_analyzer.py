"""
Code Analyzer using Ollama
This program analyzes code for potential issues, code quality, and security vulnerabilities
Supports multiple programming languages using local LLM models
"""

import os
import sys
import argparse
import json
import subprocess
import tempfile
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Try to import ollama
try:
    import ollama
except ImportError:
    print("[WARNING] Ollama package is not installed. Please install it using: pip install ollama")

# Import C++ specific analysis functions
try:
    from cpp_code_analyzer import check_syntax as cpp_check_syntax, generate_fix_suggestion as cpp_generate_fix_suggestion
except ImportError:
    print("[WARNING] Could not import C++ analyzer functions. Basic syntax checking will be used.")


def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from analysis results

    Args:
        results: Analysis results

    Returns:
        Formatted report as string
    """
    language = results.get('language', 'Unknown')
    report = [f"# Code Analysis Report\n", f"File: {results.get('file_path', 'Unknown')}\n", f"Language: {language}\n"]

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
            report.append("```\n")
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

    # Language-specific fix suggestions
    if language == "cpp":
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

    elif language == "python":
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

    # Add more language-specific fixes as needed

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
        # Common C++ error patterns
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
        # Common Python error patterns
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

    # Add more language-specific error patterns as needed

    # Default suggestion
    return f"Review the line and fix the syntax according to {language.upper()} rules"


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

    # Check each line for common issues
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

    if language == "cpp":
        # Use comprehensive C++ syntax checking from cpp_code_analyzer
        try:
            syntax_errors = cpp_check_syntax(code)
            print(f"[INFO] C++ syntax check completed. Found {len(syntax_errors)} syntax issues.")
        except NameError:
            print("[WARNING] C++ analyzer functions not available. Using basic syntax checking.")
            # Fallback to basic syntax checking
            syntax_errors = _basic_cpp_syntax_check(code)
    else:
        # For other languages, use basic syntax checking
        syntax_errors = _basic_syntax_check(code, language)

    return syntax_errors


def _basic_cpp_syntax_check(code: str) -> List[Dict[str, Any]]:
    """
    Basic C++ syntax checking using g++ compiler as fallback
    
    Args:
        code: C++ code to check
        
    Returns:
        List of syntax errors
    """
    syntax_errors = []
    
    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))

    try:
        # Use g++ for C++ syntax checking
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
                        "recommendation": generate_fix_suggestion(code, int(line_num), message, "cpp")
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
            "recommendation": "Install g++ to enable C++ syntax checking"
        })
        # Add basic static analysis as fallback
        cpp_issues = analyze_cpp_code(code)
        syntax_errors.extend(cpp_issues)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return syntax_errors


def _basic_syntax_check(code: str, language: str) -> List[Dict[str, Any]]:
    """
    Basic syntax checking for non-C++ languages
    
    Args:
        code: Code to check
        language: Programming language
        
    Returns:
        List of syntax errors
    """
    syntax_errors = []
    
    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(suffix=f'.{language}', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))

    try:
        if language == "python":
            # Use pylint for Python syntax checking
            try:
                result = subprocess.run(
                    ['pylint', '--output-format=text', temp_file_path],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    timeout=10
                )

                # Parse pylint output
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
                            "recommendation": generate_fix_suggestion(code, int(line_num), message, language)
                        })
            except FileNotFoundError:
                syntax_errors.append({
                    "line": 0,
                    "severity": "info",
                    "description": "Pylint not found",
                    "recommendation": "Install pylint to enable Python syntax checking"
                })

        # Add more language-specific syntax checking as needed

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
            "description": f"Syntax checker for {language} not found",
            "recommendation": f"Install appropriate tools to enable {language} syntax checking"
        })
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return syntax_errors


def detect_language(file_path: str) -> str:
    """
    Detect the programming language based on file extension

    Args:
        file_path: Path to the code file

    Returns:
        Detected language name (lowercase)
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Map file extensions to languages
    language_map = {
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.py': 'python',
        '.js': 'javascript',
        '.html': 'html',
        '.css': 'css',
        '.java': 'java',
        '.c': 'c',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.ts': 'typescript',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
    }

    return language_map.get(ext, 'unknown')


class CodeAnalyzer:
    """
    A class to analyze code using the Ollama models
    Supports multiple programming languages
    """
    
    def __init__(self, model_name: str = "qwen2:7b"):
        """
        Initialize the analyzer with the specified model

        Args:
            model_name: Name of the Ollama model to use (e.g., "qwen2:7b")
        """
        self.model_name = model_name
        self.model = None  # Not used with Ollama
        self.tokenizer = None  # Not used with Ollama
        self._cache = {}  # Cache for analyzed files

        # Language-specific prompt templates
        self.prompt_templates = {
            "cpp": """
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
            """,

            "python": """
            You are an expert Python code analyzer. Analyze the following Python code for:
            1. Potential bugs and logical errors
            2. Memory and resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues (PEP 8 compliance)

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            PYTHON CODE:
            ```python
            {code}
            ```

            ANALYSIS:
            """,

            "javascript": """
            You are an expert JavaScript code analyzer. Analyze the following JavaScript code for:
            1. Potential bugs and logical errors
            2. Memory leaks and resource management
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            JAVASCRIPT CODE:
            ```javascript
            {code}
            ```

            ANALYSIS:
            """,

            # Add more language-specific templates as needed

            "default": """
            You are an expert code analyzer. Analyze the following code for:
            1. Potential bugs and logical errors
            2. Resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            CODE:
            ```
            {code}
            ```            ANALYSIS:
            """        }
        
    def load_model(self):
        """
        Check if Ollama is available and the model exists
        
        Returns:
            True if Ollama is available, False otherwise
        """
        # Handle fallback mode
        if self.model_name == "none":
            print("[INFO] Using fallback mode - no AI model required.")
            return True
            
        print(f"[INFO] Checking Ollama availability for model: {self.model_name}")
        try:
            # Import Ollama
            import ollama
            
            try:
                # List available models to check if the specified model exists
                models = ollama.list()
                
                # Check if models response has the expected structure
                if not isinstance(models, dict) or 'models' not in models:
                    print(f"[WARNING] Unexpected Ollama models response structure: {models}")
                    # Try to pull the model anyway
                    print(f"[INFO] Attempting to pull model {self.model_name}...")
                    ollama.pull(self.model_name)
                    return True
                
                # Extract model names safely
                model_names = []
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    else:
                        print(f"[WARNING] Unexpected model structure: {model}")
                
                if self.model_name not in model_names:
                    print(f"[WARNING] Model {self.model_name} not found in Ollama.")
                    print(f"[INFO] Available models: {', '.join(model_names)}")
                    print(f"[INFO] Pulling model {self.model_name}...")
                    ollama.pull(self.model_name)
                else:
                    print(f"[INFO] Model {self.model_name} is available in Ollama.")
                
                return True
            except Exception as e:
                print(f"[WARNING] Error accessing Ollama models: {str(e)}")
                # Try to pull the model anyway
                try:
                    print(f"[INFO] Attempting to pull model {self.model_name} despite error...")
                    ollama.pull(self.model_name)
                    return True
                except Exception as pull_error:
                    print(f"[ERROR] Failed to pull model: {str(pull_error)}")
                    return False
                
        except ImportError:
            print("[ERROR] Ollama Python package not installed.") 
            print("[INFO] Please install the Ollama package:")
            print("pip install ollama")
            return False
            
    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze the given code using Ollama

        Args:
            code: Code to analyze
            language: Programming language of the code

        Returns:
            Dictionary containing analysis results
        """
        import traceback
        
        # Check for syntax errors first
        syntax_errors = check_syntax(code, language)

        # Get the appropriate prompt template for the language
        prompt_template = self.prompt_templates.get(language, self.prompt_templates["default"])

        # Prepare the prompt
        formatted_prompt = prompt_template.format(code=code)        # Initialize results structure
        analysis_results = {
            "language": language,
            "syntax_errors": syntax_errors,
            "bugs": [],
            "memory_issues": [],
            "security_vulnerabilities": [],
            "performance_issues": [],
            "style_issues": []
        }

        # Check if we're using fallback mode (no AI model)
        if self.model_name == "none":
            print("[INFO] Running in fallback mode with syntax-only analysis.")
            # Add basic static analysis for supported languages
            if language == "cpp":
                cpp_issues = analyze_cpp_code(code)
                # Categorize the issues
                for issue in cpp_issues:
                    if "bug" in issue["description"].lower() or "error" in issue["description"].lower():
                        analysis_results["bugs"].append(issue)
                    elif "memory" in issue["description"].lower() or "buffer" in issue["description"].lower():
                        analysis_results["memory_issues"].append(issue)
                    elif "security" in issue["description"].lower() or "overflow" in issue["description"].lower():
                        analysis_results["security_vulnerabilities"].append(issue)
                    elif "performance" in issue["description"].lower() or "inefficient" in issue["description"].lower():
                        analysis_results["performance_issues"].append(issue)
                    else:
                        analysis_results["style_issues"].append(issue)
            
            print(f"[INFO] Basic analysis completed. Found {sum(len(analysis_results[k]) for k in ['bugs', 'memory_issues', 'security_vulnerabilities', 'performance_issues', 'style_issues'])} issues.")
            return analysis_results

        # Ensure Ollama is available
        try:
            import ollama
            
            # Make sure the model is available
            if not self.load_model():
                print("[ERROR] Failed to load Ollama model. Returning only syntax analysis.")
                return analysis_results
              # Generate analysis using Ollama
            print(f"[INFO] Generating analysis using Ollama model {self.model_name}...")
            
            # Generate with Ollama using correct API parameters
            response = ollama.generate(
                model=self.model_name,
                prompt=formatted_prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 1024,
                }
            )
            
            # Extract the analysis part (after "ANALYSIS:")
            analysis_text = response['response'].split("ANALYSIS:")[-1].strip()
            
            # Parse the analysis into structured results
            parsed_results = self._parse_analysis(analysis_text)
            
            # Update the results with the parsed analysis
            for key in ["bugs", "memory_issues", "security_vulnerabilities", "performance_issues", "style_issues"]:
                if key in parsed_results and parsed_results[key]:
                    analysis_results[key] = parsed_results[key]
                    
            print(f"[INFO] Analysis with Ollama completed. Found {sum(len(analysis_results[k]) for k in ['bugs', 'memory_issues', 'security_vulnerabilities', 'performance_issues', 'style_issues'])} issues.")
            
        except ImportError:
            print("[ERROR] Ollama Python package not installed. Run: pip install ollama")
        except Exception as e:
            print(f"[ERROR] Failed to generate analysis using Ollama: {str(e)}")
            print(traceback.format_exc())

        return analysis_results

    def _parse_analysis(self, analysis_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse the model's text output into structured analysis results

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
            # Split the analysis into lines
            lines = analysis_text.split('\n')

            current_section = None
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

                # Check if this is an issue line (usually starts with a number, bullet point, or line reference)
                if (line.startswith('- ') or line.startswith('* ') or 
                    line.startswith('Line ') or re.match(r'^\d+\.', line)):

                    # If we have a previous issue in the buffer, add it to results
                    if issue_buffer and 'description' in issue_buffer and current_section:
                        if 'line' not in issue_buffer:
                            issue_buffer['line'] = 0
                        if 'severity' not in issue_buffer:
                            issue_buffer['severity'] = 'medium'
                        if 'recommendation' not in issue_buffer:
                            issue_buffer['recommendation'] = "No specific recommendation provided."

                        results[current_section].append(issue_buffer.copy())

                    # Start a new issue
                    issue_buffer = {'description': line}

                    # Try to extract line number
                    line_match = re.search(r'Line\s+(\d+)', line)
                    if line_match:
                        issue_buffer['line'] = int(line_match.group(1))

                    # Try to extract severity
                    severity_match = re.search(r'\((critical|high|medium|low|info)\)', line, re.IGNORECASE)
                    if severity_match:
                        issue_buffer['severity'] = severity_match.group(1).lower()
                    else:
                        # Guess severity based on keywords
                        if any(word in line.lower() for word in ['critical', 'crash', 'exploit', 'vulnerability', 'security']):
                            issue_buffer['severity'] = 'critical'
                        elif any(word in line.lower() for word in ['high', 'error', 'bug', 'memory leak']):
                            issue_buffer['severity'] = 'high'
                        elif any(word in line.lower() for word in ['medium', 'warning', 'performance']):
                            issue_buffer['severity'] = 'medium'
                        elif any(word in line.lower() for word in ['low', 'style', 'readability']):
                            issue_buffer['severity'] = 'low'
                        else:
                            issue_buffer['severity'] = 'info'

                # If this line starts with "Recommendation" or similar, it's a recommendation
                elif current_section and issue_buffer and ('recommendation' in line.lower() or 'suggestion' in line.lower()):
                    recommendation_text = line.split(':', 1)[1].strip() if ':' in line else line
                    issue_buffer['recommendation'] = recommendation_text

                # Otherwise, if we're in an issue, append to the description or set as recommendation
                elif current_section and issue_buffer:
                    if 'recommendation' not in issue_buffer and line.startswith('  '):
                        issue_buffer['recommendation'] = line.strip()
                    else:
                        issue_buffer['description'] += " " + line

            # Add the last issue if there is one
            if issue_buffer and 'description' in issue_buffer and current_section:
                if 'line' not in issue_buffer:
                    issue_buffer['line'] = 0
                if 'severity' not in issue_buffer:
                    issue_buffer['severity'] = 'medium'
                if 'recommendation' not in issue_buffer:
                    issue_buffer['recommendation'] = "No specific recommendation provided."

                results[current_section].append(issue_buffer.copy())

        except Exception as e:
            print(f"[ERROR] Failed to parse analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())

        return results

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a code file

        Args:
            file_path: Path to the code file

        Returns:
            Dictionary containing analysis results
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "file_path": file_path}

        # Detect language based on file extension
        language = detect_language(file_path)

        if language == "unknown":
            print(f"[WARNING] Could not determine language for file {file_path}. Using generic analysis.")
            # Instead of returning an error, use a generic language template
            language = "default"

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

                if not code.strip():
                    return {
                        "error": "The file is empty. Please upload a file with code content to analyze.",
                        "file_path": file_path
                    }

                results = self.analyze_code(code, language)
                results["file_path"] = file_path
                results["language"] = language

                # Cache the results
                self._cache[cache_key] = results
                return results
            except IOError as e:
                error_msg = f"IO Error: {str(e)}"
                suggestion = "Make sure the file exists and you have permission to read it."
                print(f"[ERROR] {error_msg} - {file_path}")
                return {"error": f"{error_msg}. {suggestion}", "file_path": file_path}
            except UnicodeDecodeError as e:
                error_msg = f"Encoding error: {str(e)}"
                suggestion = "The file might be binary or use an unsupported encoding. Try saving it with UTF-8 encoding."
                print(f"[ERROR] {error_msg} - {file_path}")
                return {"error": f"{error_msg}. {suggestion}", "file_path": file_path}
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"[ERROR] {error_msg} - {file_path}")
                return {"error": f"{error_msg}. Please try again or contact support if the issue persists.", "file_path": file_path}
        except Exception as e:
            return {"error": f"Error accessing file: {str(e)}", "file_path": file_path}


def main():
    """Main function to run the analyzer from command line"""
    parser = argparse.ArgumentParser(description="Analyze code using Ollama")
    parser.add_argument("files", nargs="*", default=None, help="Path to code file(s) to analyze")
    parser.add_argument("--model", default="qwen2:7b", help="Ollama model to use (e.g., qwen2:7b, llama3:8b)")
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

    # Check if Ollama is installed
    try:
        import ollama
    except ImportError:
        print("[ERROR] Ollama Python package not installed.")
        print("[INFO] Please install the Ollama package: pip install ollama")
        sys.exit(1)

    # Initialize analyzer
    analyzer = CodeAnalyzer(model_name=args.model)
    if not analyzer.load_model():
        print("[ERROR] Failed to load or find the specified model. Please ensure Ollama is running and the model is available.")
        print("[INFO] You may need to run: ollama pull " + args.model)
        sys.exit(1)

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
                language = results.get("language", "unknown")
                suggested_fixes = suggest_fixes(file_content, all_issues, language)
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
