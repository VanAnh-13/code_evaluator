"""
Model module for Code Evaluator
Handles LLM API client management for multiple providers
"""

from code_evaluator.model.loader import ModelLoader
from code_evaluator.model.config import APIConfig
from code_evaluator.model.factory import create_client
from code_evaluator.model.base_client import BaseLLMClient, APIError

__all__ = [
    "ModelLoader",
    "APIConfig",
    "create_client",
    "BaseLLMClient",
    "APIError",
]
