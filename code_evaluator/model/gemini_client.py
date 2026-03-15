"""
Google Gemini API client for Code Evaluator
Supports Gemini 2.0 Flash, Gemini Pro, etc.
"""

import json
import logging
import uuid
from typing import List, Dict, Optional, Union

from code_evaluator.model.base_client import BaseLLMClient, APIError, ChatResponse, ToolCall

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Google Gemini API client using google-generativeai SDK"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "The 'google-generativeai' package is required for Gemini provider. "
                "Install it with: pip install google-generativeai"
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

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

        # Build Gemini-compatible content
        system_parts = []
        gemini_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            elif msg["role"] == "tool":
                # Gemini expects function responses inline
                from google.generativeai import protos
                gemini_messages.append({
                    "role": "user",
                    "parts": [protos.Part(function_response=protos.FunctionResponse(
                        name=msg.get("name", "tool"),
                        response={"result": msg.get("content", "")},
                    ))]
                })
            else:
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})

        generation_config = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_output_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if json_mode and not tools:
            generation_config["response_mime_type"] = "application/json"

        model_kwargs = {"generation_config": generation_config}
        if system_parts:
            model_kwargs["system_instruction"] = "\n".join(system_parts)

        # Add tool support
        gemini_tools = None
        if tools:
            from google.generativeai import protos
            function_declarations = []
            for tool in tools:
                func = tool.get("function", tool)
                params = func.get("parameters", {"type": "object", "properties": {}})
                # Convert JSON Schema to Gemini Schema
                fd = protos.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=self._convert_to_gemini_schema(params),
                )
                function_declarations.append(fd)
            gemini_tools = [protos.Tool(function_declarations=function_declarations)]
            model_kwargs["tools"] = gemini_tools

        try:
            model = self._genai.GenerativeModel(self.model, **model_kwargs)
            response = model.generate_content(gemini_messages)

            # If tools were provided, check for function calls
            if tools:
                content_text = ""
                tool_calls = []

                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call.name:
                        fc = part.function_call
                        # Convert proto Map to dict
                        args = {}
                        if fc.args:
                            args = dict(fc.args)
                        tool_calls.append(ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=args,
                        ))
                    elif hasattr(part, "text") and part.text:
                        content_text += part.text

                chat_resp = ChatResponse(
                    content=content_text,
                    tool_calls=tool_calls,
                )
                self._log_response(len(content_text))
                return chat_resp

            content = response.text if response.text else ""
            self._log_response(len(content))
            return content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini API error: {error_msg}")
            raise APIError(error_msg, provider="Gemini")

    @staticmethod
    def _convert_to_gemini_schema(json_schema: Dict) -> Dict:
        """
        Convert a JSON Schema dict to Gemini-compatible schema.
        Gemini uses a subset of OpenAPI schema in protobuf format.
        """
        from google.generativeai import protos

        type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "array": protos.Type.ARRAY,
            "object": protos.Type.OBJECT,
        }

        schema_type = type_map.get(json_schema.get("type", "object"), protos.Type.OBJECT)
        properties = {}
        for prop_name, prop_schema in json_schema.get("properties", {}).items():
            prop_type = type_map.get(prop_schema.get("type", "string"), protos.Type.STRING)
            properties[prop_name] = protos.Schema(
                type=prop_type,
                description=prop_schema.get("description", ""),
            )

        return protos.Schema(
            type=schema_type,
            properties=properties,
            required=json_schema.get("required", []),
            description=json_schema.get("description", ""),
        )
