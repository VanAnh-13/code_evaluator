"""
Abstract base class for LLM API clients
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ToolCall:
    """Represents a tool call requested by the LLM."""

    def __init__(self, id: str, name: str, arguments: Dict):
        self.id = id
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolCall":
        return cls(id=data["id"], name=data["name"], arguments=data["arguments"])


class ChatResponse:
    """
    Unified response from LLM chat, supporting both text and tool-call responses.
    """

    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        raw: Optional[Dict] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.raw = raw or {}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        return self.content


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM API clients.
    All provider implementations must inherit from this class.
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.timeout = kwargs.get("timeout", 120)
        self.base_url = kwargs.get("base_url", None)

    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> Union[str, ChatResponse]:
        """
        Send a chat completion request to the LLM API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles: 'system', 'user', 'assistant', 'tool'
            json_mode: If True, request JSON-formatted output.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice strategy ('auto', 'none', 'required').

        Returns:
            str if no tools are provided (backward compatible).
            ChatResponse if tools are provided (may contain tool_calls).

        Raises:
            APIError: If the API request fails.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic', 'gemini')"""
        pass

    def _log_request(self, messages: List[Dict]):
        """Log API request (without sensitive content)"""
        logger.debug(
            f"[{self.provider_name}] Sending request to {self.model} "
            f"with {len(messages)} messages"
        )

    def _log_response(self, response_length: int):
        """Log API response metadata"""
        logger.debug(
            f"[{self.provider_name}] Received response: {response_length} chars"
        )


class APIError(Exception):
    """Custom exception for API errors"""

    def __init__(self, message: str, provider: str = "", status_code: int = 0):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}" if provider else message)
