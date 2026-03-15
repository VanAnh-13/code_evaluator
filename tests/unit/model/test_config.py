"""Tests for APIConfig"""

import pytest
import os
from unittest.mock import patch
from code_evaluator.model.config import APIConfig, DEFAULT_MODELS


class TestAPIConfig:
    """Test API configuration functionality"""

    def test_default_initialization(self):
        """Test default config initialization"""
        config = APIConfig()
        assert config.provider == "openai"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.timeout == 120

    def test_custom_initialization(self):
        """Test custom config initialization"""
        config = APIConfig(
            provider="anthropic",
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            max_tokens=8192
        )
        assert config.provider == "anthropic"
        assert config.api_key == "test-key"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192

    def test_model_default_selection(self):
        """Test automatic model selection based on provider"""
        config_openai = APIConfig(provider="openai")
        assert config_openai.model == DEFAULT_MODELS["openai"]

        config_anthropic = APIConfig(provider="anthropic")
        assert config_anthropic.model == DEFAULT_MODELS["anthropic"]

        config_gemini = APIConfig(provider="gemini")
        assert config_gemini.model == DEFAULT_MODELS["gemini"]

    def test_from_env_with_all_variables(self):
        """Test loading config from environment variables"""
        env_vars = {
            'API_PROVIDER': 'anthropic',
            'API_KEY': 'sk-test-123',
            'API_MODEL': 'claude-3-haiku-20240307',
            'API_TEMPERATURE': '0.7',
            'API_MAX_TOKENS': '2048',
            'API_TIMEOUT': '60',
            'API_BASE_URL': 'https://custom.api.com'
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = APIConfig.from_env()
            assert config.provider == 'anthropic'
            assert config.api_key == 'sk-test-123'
            assert config.model == 'claude-3-haiku-20240307'
            assert config.temperature == 0.7
            assert config.max_tokens == 2048
            assert config.timeout == 60
            assert config.base_url == 'https://custom.api.com'

    def test_from_env_with_minimal_variables(self):
        """Test loading config with only required env vars"""
        env_vars = {
            'API_PROVIDER': 'openai',
            'API_KEY': 'test-key'
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = APIConfig.from_env()
            assert config.provider == 'openai'
            assert config.api_key == 'test-key'
            assert config.model == DEFAULT_MODELS['openai']

    def test_from_env_without_api_key(self):
        """Test loading config without API key (should warn but not fail)"""
        with patch.dict(os.environ, {'API_PROVIDER': 'openai'}, clear=False):
            with patch.dict(os.environ, {'API_KEY': ''}, clear=False):
                config = APIConfig.from_env()
                assert config.api_key == ""

    def test_validate_valid_config(self):
        """Test validation of valid config"""
        config = APIConfig(provider="openai", api_key="test-key")
        assert config.validate() is True

    def test_validate_missing_api_key(self):
        """Test validation fails without API key"""
        config = APIConfig(provider="openai", api_key="")
        assert config.validate() is False

    def test_validate_invalid_provider(self):
        """Test validation fails with invalid provider"""
        config = APIConfig(provider="invalid", api_key="test-key")
        assert config.validate() is False

    def test_provider_normalization(self):
        """Test provider name normalization"""
        config = APIConfig(provider="  OpenAI  ")
        assert config.provider == "openai"

    def test_provider_display_name(self):
        """Test provider display name property"""
        config_openai = APIConfig(provider="openai")
        assert config_openai.provider_display_name == "OpenAI"

        config_anthropic = APIConfig(provider="anthropic")
        assert config_anthropic.provider_display_name == "Anthropic"

        config_gemini = APIConfig(provider="gemini")
        assert config_gemini.provider_display_name == "Google Gemini"

    def test_agent_config_defaults(self):
        """Test agent-related config defaults"""
        config = APIConfig()
        assert config.agent_enabled is True
        assert config.agent_max_steps == 15
        assert config.agent_max_fix_retries == 3
        assert config.agent_session_ttl == 3600
        assert config.agent_streaming is True
