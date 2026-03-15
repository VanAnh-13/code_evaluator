"""
API Configuration for Code Evaluator
Supports multiple LLM providers: OpenAI, Anthropic, Google Gemini
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "ollama": "codellama",
}


@dataclass
class APIConfig:
    """Configuration for LLM API access"""

    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    base_url: Optional[str] = None
    timeout: int = 120

    def __post_init__(self):
        self.provider = self.provider.lower().strip()
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")

    @classmethod
    def from_env(cls) -> "APIConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            API_PROVIDER: Provider name (openai, anthropic, gemini)
            API_KEY: API key for the provider
            API_MODEL: Model name (optional, uses provider default)
            API_TEMPERATURE: Temperature for generation (optional)
            API_MAX_TOKENS: Max tokens for generation (optional)
            API_BASE_URL: Custom base URL for API (optional)
            API_TIMEOUT: Request timeout in seconds (optional)
        """
        provider = os.environ.get("API_PROVIDER", "openai").lower().strip()
        api_key = os.environ.get("API_KEY", "")
        model = os.environ.get("API_MODEL", "")
        temperature = float(os.environ.get("API_TEMPERATURE", "0.3"))
        max_tokens = int(os.environ.get("API_MAX_TOKENS", "4096"))
        base_url = os.environ.get("API_BASE_URL", None)
        timeout = int(os.environ.get("API_TIMEOUT", "120"))

        if not api_key:
            logger.warning(
                "API_KEY not set. Please set the API_KEY environment variable "
                "or create a .env file with your API key."
            )

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=timeout,
        )

    def validate(self) -> bool:
        """Check if the configuration is valid"""
        # Ollama doesn't require an API key (local hosting)
        if self.provider == "ollama":
            return True
        if not self.api_key:
            return False
        if self.provider not in ("openai", "anthropic", "gemini", "ollama"):
            return False
        return True

    # Agent configuration
    agent_enabled: bool = True
    agent_max_steps: int = 15
    agent_max_fix_retries: int = 3
    agent_session_ttl: int = 3600  # seconds
    agent_streaming: bool = True

    @property
    def provider_display_name(self) -> str:
        """Human-readable provider name"""
        names = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Google Gemini",
            "ollama": "Ollama (Local)",
        }
        return names.get(self.provider, self.provider.title())
