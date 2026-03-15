"""
Model loader for Code Evaluator
Handles LLM API client management for code analysis
Supports multiple providers: OpenAI, Anthropic, Google Gemini
"""

import logging
from typing import Optional, List, Dict, Any, Union

from code_evaluator.model.config import APIConfig
from code_evaluator.model.base_client import BaseLLMClient, APIError, ChatResponse
from code_evaluator.model.factory import create_client

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages LLM API client for code analysis.
    Wraps the provider-specific client with a unified interface.
    """

    def __init__(self, config: Optional[APIConfig] = None, model_name: str = ""):
        """
        Initialize the model loader.

        Args:
            config: API configuration. If None, loads from environment.
            model_name: Legacy parameter (ignored if config is provided).
        """
        if config is None:
            config = APIConfig.from_env()
        
        self.config = config
        self.model_name = config.model or model_name
        self._client: Optional[BaseLLMClient] = None

    @property
    def is_loaded(self) -> bool:
        """Check if the API client is initialized"""
        return self._client is not None

    @property
    def client(self) -> Optional[BaseLLMClient]:
        """Get the underlying API client"""
        return self._client

    def load(self) -> bool:
        """
        Initialize the API client.

        Returns:
            True if successful, False otherwise.
        """
        if self._client is not None:
            return True

        if not self.config.validate():
            logger.error(
                "Invalid API configuration. Please check your API_KEY and API_PROVIDER."
            )
            print("[ERROR] Invalid API configuration. Set API_KEY and API_PROVIDER environment variables.")
            return False

        try:
            self._client = create_client(self.config)
            print(
                f"[INFO] API client initialized: {self.config.provider_display_name} "
                f"({self.config.model})"
            )
            return True
        except ImportError as e:
            print(f"[ERROR] {str(e)}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to initialize API client: {str(e)}")
            return False

    def analyze(
        self,
        messages: List[Dict],
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> Union[str, ChatResponse]:
        """
        Send analysis request to the LLM API.

        Args:
            messages: Chat messages (system, user, assistant, tool roles).
            json_mode: Request JSON output format.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            tools: Optional tool definitions for agent-mode function calling.
            tool_choice: Optional tool choice strategy ('auto', 'none', 'required').

        Returns:
            str if no tools provided (backward compatible).
            ChatResponse if tools provided (may contain tool_calls).

        Raises:
            RuntimeError: If client is not initialized.
            APIError: If the API request fails.
        """
        if self._client is None:
            raise RuntimeError("API client not initialized. Call load() first.")

        return self._client.chat(
            messages=messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider (safe for display)"""
        return {
            "provider": self.config.provider,
            "provider_name": self.config.provider_display_name,
            "model": self.config.model,
            "is_loaded": self.is_loaded,
            "has_api_key": bool(self.config.api_key),
        }

    def unload(self) -> None:
        """Release the API client"""
        self._client = None

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
