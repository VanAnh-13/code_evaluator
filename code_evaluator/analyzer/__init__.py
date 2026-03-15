"""
Analyzer module for Code Evaluator
"""

from code_evaluator.analyzer.syntax_checker import check_syntax
from code_evaluator.analyzer.fix_suggester import suggest_fixes, generate_fix_suggestion

def __getattr__(name):
    """Lazy import for CodeAnalyzer which requires torch"""
    if name == "CodeAnalyzer":
        from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
        return CodeAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CodeAnalyzer",
    "check_syntax",
    "suggest_fixes",
    "generate_fix_suggestion",
]
