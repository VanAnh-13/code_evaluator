"""
AI Agent Module for Code Evaluator
Provides multi-step reasoning, tool-calling, and autonomous code analysis.
"""


def __getattr__(name):
    """Lazy imports for agent components."""
    if name == "AgentExecutor":
        from code_evaluator.agent.executor import AgentExecutor
        return AgentExecutor
    elif name == "AgentSession":
        from code_evaluator.agent.session import AgentSession
        return AgentSession
    elif name == "SessionManager":
        from code_evaluator.agent.session import SessionManager
        return SessionManager
    elif name == "ToolRegistry":
        from code_evaluator.agent.tool_registry import ToolRegistry
        return ToolRegistry
    elif name == "create_default_registry":
        from code_evaluator.agent.tools import create_default_registry
        return create_default_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentExecutor",
    "AgentSession",
    "SessionManager",
    "ToolRegistry",
    "create_default_registry",
]
