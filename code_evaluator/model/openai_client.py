"""
OpenAI API client for Code Evaluator
Supports GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.
"""

import json
import logging
from typing import List, Dict, Optional, Union

from code_evaluator.model.base_client import BaseLLMClient, APIError, ChatResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client using the openai SDK"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import openai
            client_kwargs = {"api_key": api_key, "timeout": self.timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAI provider. "
                "Install it with: pip install openai"
            )

    @property
    def provider_name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: List[Dict],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> Union[str, ChatResponse]:
        self._log_request(messages)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}

        # Add tool calling support
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        try:
            response = self._client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # If tools were provided, return ChatResponse
            if tools:
                tool_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        ))

                chat_resp = ChatResponse(
                    content=message.content or "",
                    tool_calls=tool_calls,
                    raw={"role": "assistant", "content": message.content, "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in (message.tool_calls or [])
                    ]} if message.tool_calls else {"role": "assistant", "content": message.content or ""},
                )
                self._log_response(len(chat_resp.content))
                return chat_resp

            # Backward compatible: return plain string
            content = message.content or ""
            self._log_response(len(content))
            return content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenAI API error: {error_msg}")
            raise APIError(error_msg, provider="OpenAI")
