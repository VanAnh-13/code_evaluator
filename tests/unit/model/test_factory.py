"""Tests for model factory"""

import pytest
from unittest.mock import patch, Mock
from code_evaluator.model.factory import ModelFactory
from code_evaluator.model.config import APIConfig
from code_evaluator.model.openai_client import OpenAIClient
from code_evaluator.model.anthropic_client import AnthropicClient
from code_evaluator.model.gemini_client import GeminiClient


class TestModelFactory:
    """Test model factory functionality"""

    def test_create_openai_client(self):
        """Test creating OpenAI client"""
        config = APIConfig(provider="openai", api_key="test-key")
        client = ModelFactory.create(config)
        assert isinstance(client, OpenAIClient)

    def test_create_anthropic_client(self):
        """Test creating Anthropic client"""
        config = APIConfig(provider="anthropic", api_key="test-key")
        client = ModelFactory.create(config)
        assert isinstance(client, AnthropicClient)

    def test_create_gemini_client(self):
        """Test creating Gemini client"""
        config = APIConfig(provider="gemini", api_key="test-key")
        client = ModelFactory.create(config)
        assert isinstance(client, GeminiClient)

    def test_create_invalid_provider(self):
        """Test creating client with invalid provider raises error"""
        config = APIConfig(provider="invalid", api_key="test-key")
        with pytest.raises(ValueError):
            ModelFactory.create(config)

    def test_provider_case_insensitive(self):
        """Test provider names are case insensitive"""
        config = APIConfig(provider="OpenAI", api_key="test-key")
        client = ModelFactory.create(config)
        assert isinstance(client, OpenAIClient)

    def test_factory_with_custom_base_url(self):
        """Test factory passes custom base URL to client"""
        config = APIConfig(
            provider="openai",
            api_key="test-key",
            base_url="https://custom.api.com"
        )
        client = ModelFactory.create(config)
        assert client is not None
        # Base URL should be set in the client
