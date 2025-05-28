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
import torch
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
            # Updated regex pattern to handle Windows paths with drive letters (C:\path\file.cpp)
            error_pattern = re.compile(r'(.+?):(\d+):(\d+):\s+(warning|error):\s+(.+)')
            for line in result.stderr.splitlines():
                match = error_pattern.match(line)
                if match:
                    file_path, line_num, column, error_type, message = match.groups()
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

    def __init__(self, model_name: str):
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
        You are an expert code analyzer. Analyze the following code for:
        1. Potential bugs and logical errors
        2. Memory management issues (leaks, dangling pointers, etc.)
        3. Security vulnerabilities
        4. Performance issues
        5. Code style and readability issues

        For each issue you find, you MUST provide:
        - The specific line number (as 'Line <number>') where the issue occurs
        - The severity (critical, high, medium, low, info)
        - A clear description of the issue
        - A recommendation for fixing it

        Format your analysis as a list, grouping by issue type, and always include the line number for every issue.

        CODE:
        ```
        {code}
        ```

        ANALYSIS:
        """

    def load_model(self):
        """
        Load the Qwen model or a fine-tuned model

        Returns:
            Tuple of (model, tokenizer) if successful, None otherwise
        """
        print(f"[INFO] Loading model: {self.model_name}")
        try:
            # Import required libraries
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Using device: {device}")

            # Load tokenizer
            print(f"[INFO] Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            print(f"[INFO] Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Check if this is a PEFT (LoRA) fine-tuned model
            if os.path.isdir(self.model_name) and any(file.endswith("adapter_config.json") for file in os.listdir(self.model_name) if os.path.isfile(os.path.join(self.model_name, file))):
                print(f"[INFO] Detected PEFT/LoRA fine-tuned model")
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, self.model_name)
                    print(f"[INFO] Successfully loaded PEFT/LoRA adapters")
                except ImportError:
                    print(f"[WARNING] PEFT library not found. Please install it to use LoRA fine-tuned models.")
                except Exception as e:
                    print(f"[WARNING] Failed to load PEFT/LoRA adapters: {str(e)}")

            self.model = model
            self.tokenizer = tokenizer
            return model, tokenizer

        except ImportError as e:
            print(f"[ERROR] Required libraries not found: {str(e)}")
            print("[INFO] Please install the required dependencies:")
            print("pip install transformers torch")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            return None

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze the given C++ code using the loaded model

        Args:
            code: C++ code to analyze

        Returns:
            Dictionary containing analysis results
        """
        # Check for syntax errors first
        syntax_errors = check_syntax(code)

        # Prepare the prompt
        formatted_prompt = self.prompt_template.format(code=code)

        # Initialize results structure
        analysis_results = {
            "syntax_errors": syntax_errors,
            "bugs": [],
            "memory_issues": [],
            "security_vulnerabilities": [],
            "performance_issues": [],
            "style_issues": []
        }

        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            print("[WARNING] Model not loaded. Loading model now...")
            model_loaded = self.load_model()
            if not model_loaded:
                print("[ERROR] Failed to load model. Returning only syntax analysis.")
                return analysis_results

        try:
            # Generate analysis using the model
            print("[INFO] Generating analysis...")

            # Tokenize the prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the analysis part (after "ANALYSIS:")
            analysis_text = response.split("ANALYSIS:")[-1].strip()

            # Parse the analysis into structured results
            parsed_results = self._parse_analysis(analysis_text)

            # Update the results with the parsed analysis
            for key in ["bugs", "memory_issues", "security_vulnerabilities", "performance_issues", "style_issues"]:
                if key in parsed_results and parsed_results[key]:
                    analysis_results[key] = parsed_results[key]

            print(f"[INFO] Analysis completed. Found {sum(len(analysis_results[k]) for k in ['bugs', 'memory_issues', 'security_vulnerabilities', 'performance_issues', 'style_issues'])} issues.")

        except Exception as e:
            print(f"[ERROR] Failed to generate analysis: {str(e)}")
            import traceback
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
                            issue_buffer['line'] = None
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
                        try:
                            parsed_line_num = int(line_match.group(1))
                            if parsed_line_num > 0: # Only accept positive line numbers
                                issue_buffer['line'] = parsed_line_num
                            # If parsed_line_num is 0 or negative, we don't set issue_buffer['line'] here.
                            # This allows it to be caught by 'if 'line' not in issue_buffer:' later
                            # and defaulted to None, or for the subsequent detailed parsing (if line is None)
                            # to attempt to find a valid line number in the description.
                        except ValueError:
                            # Should not happen with \\d+ but good practice for int conversion
                            pass

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
                    issue_buffer['line'] = None
                if 'severity' not in issue_buffer:
                    issue_buffer['severity'] = 'medium'
                if 'recommendation' not in issue_buffer:
                    issue_buffer['recommendation'] = "No specific recommendation provided."

                results[current_section].append(issue_buffer.copy())

            # After parsing all issues, try to fix missing line numbers and support multiple lines
            for section in results:
                for issue in results[section]:
                    if (issue.get('line', None) is None and 'description' in issue):
                        desc = issue['description']
                        # Find all line numbers (e.g., 'line 9', 'lines 37, 41', or 'main.cpp:98')
                        matches = re.findall(r"line[s]?\s*([\d,\s\-]+)", desc, re.IGNORECASE)
                        colon_matches = re.findall(r":(\d+)", desc)
                        line_nums = []
                        for match in matches:
                            # Support ranges like 10-12
                            for part in match.split(','):
                                part = part.strip()
                                if '-' in part:
                                    start, end = part.split('-')
                                    try:
                                        start, end = int(start), int(end)
                                        line_nums.extend(list(range(start, end+1)))
                                    except Exception:
                                        continue
                                else:
                                    try:
                                        n = int(part)
                                        if n > 0:
                                            line_nums.append(n)
                                    except Exception:
                                        continue
                        for num in colon_matches:
                            n = int(num)
                            if n > 0 and n not in line_nums:
                                line_nums.append(n)
                        if line_nums:
                            if len(line_nums) == 1:
                                issue['line'] = line_nums[0]
                            else:
                                issue['lines'] = line_nums
                        else:
                            issue['line'] = None

        except Exception as e:
            print(f"[ERROR] Failed to parse analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())

        return results

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
    parser.add_argument("--model", default="https://f6bf-34-69-56-104.ngrok-free.app/Qwen-27B", help="Qwen model to use")
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
