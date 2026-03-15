"""Pytest configuration and shared fixtures"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def mock_api_config():
    """Mock APIConfig for testing"""
    from code_evaluator.model.config import APIConfig
    return APIConfig(
        provider="openai",
        api_key="test-key-12345",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=4096,
        timeout=120
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "summary": "Test code analysis summary",
        "overall_score": 85,
        "issues": [
            {
                "type": "bug",
                "severity": "medium",
                "line": 10,
                "description": "Potential null pointer",
                "suggestion": "Add null check"
            }
        ],
        "suggestions": ["Improve error handling"]
    }


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing"""
    return """def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""


@pytest.fixture
def sample_cpp_code():
    """Sample C++ code for testing"""
    return """#include <iostream>

int main() {
    int* ptr = nullptr;
    std::cout << *ptr << std::endl;
    return 0;
}
"""


@pytest.fixture
def sample_js_code():
    """Sample JavaScript code for testing"""
    return """function fetchData(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => console.log(data));
}

fetchData(userInput);
"""


@pytest.fixture
def mock_model_loader(mock_api_config):
    """Mock ModelLoader for testing"""
    from code_evaluator.model.loader import ModelLoader
    loader = ModelLoader(config=mock_api_config)
    loader._client = MagicMock()
    loader._client.chat.return_value = Mock(
        content='{"summary": "Test", "overall_score": 85, "issues": []}',
        model="gpt-4o-mini",
        usage={"prompt_tokens": 100, "completion_tokens": 50}
    )
    return loader


@pytest.fixture
def temp_code_file(tmp_path, sample_python_code):
    """Create a temporary code file for testing"""
    file_path = tmp_path / "test_code.py"
    file_path.write_text(sample_python_code)
    return str(file_path)
