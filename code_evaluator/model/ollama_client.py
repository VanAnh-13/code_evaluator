"""
Ollama API client for Code Evaluator
Supports locally-hosted Ollama models (llama2, codellama, mistral, etc.)
"""

import json
import logging
import requests
from typing import List, Dict, Optional, Union

from code_evaluator.model.base_client import BaseLLMClient, APIError, ChatResponse, ToolCall

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama API client for local LLM inference"""

    def __init__(self, api_key: str = "", model: str = "codellama", **kwargs):
        # Ollama doesn't require API key for local hosting
        super().__init__(api_key or "not-required", model, **kwargs)

        # Default Ollama base URL is localhost:11434
        self.base_url = kwargs.get("base_url", "http://localhost:11434")

        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip('/')

        # Ollama-specific parameters
        self.num_ctx = kwargs.get("num_ctx", 4096)  # Context window size
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 40)

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        Send a chat completion request to Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            json_mode: If True, request JSON-formatted output
            temperature: Override default temperature
            max_tokens: Override default max tokens (maps to num_predict)
            tools: Optional list of tool definitions (limited support in Ollama)
            tool_choice: Optional tool choice strategy

        Returns:
            str or ChatResponse depending on tool usage
        """
        self._log_request(messages)

        # Build Ollama API request
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_ctx": self.num_ctx,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
        }

        # Add num_predict (equivalent to max_tokens)
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        elif self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens

        # JSON mode support (some Ollama models support format parameter)
        if json_mode:
            payload["format"] = "json"

        # Tool calling support (experimental - not all Ollama models support this)
        if tools:
            logger.warning(
                "Tool calling with Ollama is experimental and may not work with all models. "
                "Consider using OpenAI-compatible models for better tool support."
            )
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()

            # Extract message content
            message = result.get("message", {})
            content = message.get("content", "")

            # Handle tool calls if present
            if tools and "tool_calls" in message:
                tool_calls = []
                for tc in message.get("tool_calls", []):
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", tc.get("function", {}).get("name", "")),
                        arguments=tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                    ))

                chat_resp = ChatResponse(
                    content=content,
                    tool_calls=tool_calls,
                    raw=message
                )
                self._log_response(len(content))
                return chat_resp

            # Standard response without tools
            self._log_response(len(content))
            return content

        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is running (try: ollama serve)"
            )
            logger.error(error_msg)
            raise APIError(error_msg, provider="Ollama", status_code=0)

        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timed out after {self.timeout}s"
            logger.error(error_msg)
            raise APIError(error_msg, provider="Ollama", status_code=0)

        except requests.exceptions.HTTPError as e:
            error_msg = f"Ollama HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise APIError(error_msg, provider="Ollama", status_code=e.response.status_code)

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from Ollama: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg, provider="Ollama")

        except Exception as e:
            error_msg = f"Ollama API error: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg, provider="Ollama")

    def list_models(self) -> List[str]:
        """
        List available models from Ollama.

        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            result = response.json()
            models = [model.get("name", "") for model in result.get("models", [])]
            return models

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name, "stream": False}

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            logger.info(f"Successfully pulled model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
