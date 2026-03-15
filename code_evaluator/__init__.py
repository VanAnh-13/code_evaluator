"""
Code Evaluator Package
A multi-language code analyzer using LLM API providers
"""

from code_evaluator.utils.file_utils import detect_language

__version__ = "2.0.0"


def __getattr__(name):
    """Lazy import for main classes"""
    if name == "CodeAnalyzer":
        from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
        return CodeAnalyzer
    elif name == "generate_report":
        from code_evaluator.report.generator import generate_report
        return generate_report
    elif name == "save_results":
        from code_evaluator.report.exporter import save_results
        return save_results
    elif name == "APIConfig":
        from code_evaluator.model.config import APIConfig
        return APIConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeAnalyzer",
    "generate_report",
    "save_results",
    "APIConfig",
    "detect_language",
]
