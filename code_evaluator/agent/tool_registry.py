"""
Tool Registry for AI Agent
Manages tool definitions and execution for the agent system.
Provider-agnostic tool definitions are converted to provider-specific formats at runtime.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Describes a tool that the agent can call."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict,
        handler: Callable[..., str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema
        self.handler = handler

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Central registry for agent tools.
    Supports decorator-based registration and provider-specific format export.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def tool(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to register a function as an agent tool.

        Usage:
            @registry.tool("check_syntax", "Check code syntax", {...})
            def check_syntax_tool(code: str, language: str) -> str:
                ...
        """
        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        def decorator(func: Callable[..., str]) -> Callable[..., str]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
            )
            return func

        return decorator

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict,
        handler: Callable[..., str],
    ) -> None:
        """Programmatically register a tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_definitions(self, provider: str = "openai") -> List[Dict]:
        """
        Export tool definitions in provider-specific format.
        All providers accept the OpenAI format since we convert
        in each client's chat() method.

        Args:
            provider: Currently unused; all return OpenAI format.
                      Conversion happens in the client layer.

        Returns:
            List of tool definition dicts.
        """
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Dict) -> str:
        """
        Execute a registered tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dict of keyword arguments for the tool function.

        Returns:
            String result from the tool.

        Raises:
            ValueError: If tool is not found.
            Exception: If tool execution fails.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")

        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {json.dumps(arguments, default=str)[:500]}")

        try:
            result = tool.handler(**arguments)
            # Ensure result is a string
            if not isinstance(result, str):
                result = json.dumps(result, default=str, ensure_ascii=False)
            logger.debug(f"Tool result ({tool_name}): {result[:200]}...")
            return result
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
