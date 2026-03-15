"""
Code Analyzer using Qwen model
This program analyzes code for potential issues, code quality, and security vulnerabilities
Supports multiple programming languages

NOTE: This is a backward-compatible wrapper. The main code is now in code_evaluator/

Usage:
    python code_analyzer.py <files> [options]
    
    Or use the new unified CLI:
    python -m code_evaluator.main analyze <files> [options]
"""

import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules that don't require torch
from code_evaluator.utils.file_utils import detect_language
from code_evaluator.analyzer.syntax_checker import check_syntax
from code_evaluator.analyzer.fix_suggester import suggest_fixes
from code_evaluator.report.generator import generate_report
from code_evaluator.report.exporter import save_results

# Lazy import for torch-dependent modules
_CodeAnalyzer = None

def __getattr__(name):
    """Lazy import for CodeAnalyzer which requires torch"""
    global _CodeAnalyzer
    if name == "CodeAnalyzer":
        if _CodeAnalyzer is None:
            from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
            _CodeAnalyzer = CodeAnalyzer
        return _CodeAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Export commonly used items for backward compatibility
__all__ = [
    'CodeAnalyzer',
    'generate_report',
    'save_results',
    'suggest_fixes',
    'detect_language',
    'check_syntax',
]


def main():
    """Main function to run the analyzer from command line"""
    import argparse
    
    # Import CodeAnalyzer at runtime to avoid torch dependency at module load
    from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
    
    parser = argparse.ArgumentParser(description="Analyze code using Qwen model")
    parser.add_argument("files", nargs="*", default=None, help="Path to code file(s) to analyze")
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
    analyzer = CodeAnalyzer(model_name=args.model)
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
