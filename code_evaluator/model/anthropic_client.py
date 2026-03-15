"""
Anthropic API client for Code Evaluator
Supports Claude Sonnet, Claude Haiku, etc.
"""

import json
import logging
import uuid
from typing import List, Dict, Optional, Union

from code_evaluator.model.base_client import BaseLLMClient, APIError, ChatResponse, ToolCall

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client using the anthropic SDK"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import anthropic
            client_kwargs = {"api_key": api_key, "timeout": self.timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**client_kwargs)
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic provider. "
                "Install it with: pip install anthropic"
            )

    @property
    def provider_name(self) -> str:
        return "anthropic"

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

        # Anthropic separates system message from messages
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            elif msg["role"] == "tool":
                # Anthropic expects tool results as user messages with tool_result content
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }]
                })
            elif msg["role"] == "assistant" and isinstance(msg.get("content"), list):
                # Pass through assistant messages with tool_use blocks
                api_messages.append(msg)
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # If JSON mode, append instruction to system prompt
        if json_mode and system_content and not tools:
            system_content += (
                "\n\nIMPORTANT: You MUST respond with valid JSON only. "
                "Do not include any text outside the JSON object. "
                "Do not use markdown code fences."
            )

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if system_content.strip():
            kwargs["system"] = system_content.strip()

        # Add tool calling support
        if tools:
            # Convert OpenAI-format tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", tool)
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools
            if tool_choice:
                if tool_choice == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    kwargs["tool_choice"] = {"type": "none"}
                else:
                    kwargs["tool_choice"] = {"type": "auto"}

        try:
            response = self._client.messages.create(**kwargs)

            # If tools were provided, parse response for tool_use blocks
            if tools:
                content_text = ""
                tool_calls = []
                raw_content = []

                for block in response.content:
                    if block.type == "text":
                        content_text += block.text
                        raw_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        ))
                        raw_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                chat_resp = ChatResponse(
                    content=content_text,
                    tool_calls=tool_calls,
                    raw={"role": "assistant", "content": raw_content},
                )
                self._log_response(len(content_text))
                return chat_resp

            content = response.content[0].text if response.content else ""
            self._log_response(len(content))
            return content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Anthropic API error: {error_msg}")
            raise APIError(error_msg, provider="Anthropic")
