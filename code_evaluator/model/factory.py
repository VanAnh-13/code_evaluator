"""
Factory for creating LLM API clients
"""

import logging
from typing import Optional

from code_evaluator.model.config import APIConfig
from code_evaluator.model.base_client import BaseLLMClient, APIError

logger = logging.getLogger(__name__)


def create_client(config: Optional[APIConfig] = None) -> BaseLLMClient:
    """
    Create an LLM API client based on the configuration.

    Args:
        config: API configuration. If None, loads from environment variables.

    Returns:
        An instance of BaseLLMClient for the specified provider.

    Raises:
        ValueError: If the provider is unknown or API key is missing.
        ImportError: If the required SDK is not installed.
    """
    if config is None:
        config = APIConfig.from_env()

    if not config.api_key:
        raise ValueError(
            f"API key is required for {config.provider_display_name}. "
            f"Set the API_KEY environment variable or pass it in the configuration."
        )

    client_kwargs = {
        "api_key": config.api_key,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "base_url": config.base_url,
    }

    if config.provider == "openai":
        from code_evaluator.model.openai_client import OpenAIClient
        return OpenAIClient(**client_kwargs)

    elif config.provider == "anthropic":
        from code_evaluator.model.anthropic_client import AnthropicClient
        return AnthropicClient(**client_kwargs)

    elif config.provider == "gemini":
        from code_evaluator.model.gemini_client import GeminiClient
        return GeminiClient(**client_kwargs)

    else:
        raise ValueError(
            f"Unknown provider: '{config.provider}'. "
            f"Supported providers: openai, anthropic, gemini"
        )
