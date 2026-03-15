"""Tests for CodeAnalyzer class"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from code_evaluator.analyzer.code_analyzer import CodeAnalyzer
from code_evaluator.model.config import APIConfig


class TestCodeAnalyzer:
    """Test CodeAnalyzer main functionality"""

    def test_initialization_with_config(self, mock_api_config):
        """Test analyzer initialization with config"""
        analyzer = CodeAnalyzer(config=mock_api_config)
        assert analyzer is not None
        assert analyzer.model_loader is not None

    def test_initialization_without_config(self):
        """Test analyzer initialization without explicit config"""
        with patch.dict('os.environ', {'API_KEY': 'test-key', 'API_PROVIDER': 'openai'}):
            analyzer = CodeAnalyzer()
            assert analyzer is not None

    def test_model_name_property(self, mock_api_config):
        """Test model_name property"""
        analyzer = CodeAnalyzer(config=mock_api_config)
        assert analyzer.model_name == mock_api_config.model

    def test_model_loaded_property(self, mock_api_config):
        """Test model_loaded property"""
        analyzer = CodeAnalyzer(config=mock_api_config)
        # Initially not loaded
        assert isinstance(analyzer.model_loaded, bool)

    @patch('code_evaluator.analyzer.code_analyzer.check_syntax')
    @patch('code_evaluator.model.loader.ModelLoader.load')
    @patch('code_evaluator.model.loader.ModelLoader.analyze_code')
    def test_analyze_code_success(self, mock_analyze, mock_load, mock_syntax,
                                  mock_api_config, sample_python_code):
        """Test successful code analysis"""
        # Setup mocks
        mock_load.return_value = True
        mock_syntax.return_value = {"valid": True, "errors": []}
        mock_analyze.return_value = {
            "summary": "Good code",
            "overall_score": 85,
            "issues": []
        }

        analyzer = CodeAnalyzer(config=mock_api_config)
        result = analyzer.analyze_code(sample_python_code, "python")

        assert result is not None
        assert "overall_score" in result
        mock_analyze.assert_called_once()

    @patch('code_evaluator.utils.file_utils.read_file_content')
    @patch('code_evaluator.utils.file_utils.detect_language')
    def test_analyze_file(self, mock_detect, mock_read, mock_api_config,
                         sample_python_code, temp_code_file):
        """Test file analysis"""
        mock_detect.return_value = "python"
        mock_read.return_value = sample_python_code

        analyzer = CodeAnalyzer(config=mock_api_config)

        with patch.object(analyzer, 'analyze_code') as mock_analyze_code:
            mock_analyze_code.return_value = {"overall_score": 85}
            result = analyzer.analyze_file(temp_code_file)

            assert result is not None
            mock_analyze_code.assert_called_once()

    def test_load_model(self, mock_api_config):
        """Test model loading"""
        analyzer = CodeAnalyzer(config=mock_api_config)

        with patch.object(analyzer.model_loader, 'load') as mock_load:
            mock_load.return_value = True
            result = analyzer.load_model()
            assert result is True
            mock_load.assert_called_once()

    @patch('code_evaluator.analyzer.code_analyzer.check_syntax')
    def test_analyze_code_with_syntax_errors(self, mock_syntax, mock_api_config):
        """Test analysis with syntax errors"""
        mock_syntax.return_value = {
            "valid": False,
            "errors": ["SyntaxError: invalid syntax"]
        }

        analyzer = CodeAnalyzer(config=mock_api_config)
        code = "def invalid(\n    print('error')"

        # Should still attempt analysis or handle gracefully
        with patch.object(analyzer.model_loader, 'analyze_code') as mock_analyze:
            mock_analyze.return_value = {"overall_score": 40}
            result = analyzer.analyze_code(code, "python")
            assert result is not None

    def test_cache_usage(self, mock_api_config, sample_python_code):
        """Test that caching is utilized"""
        analyzer = CodeAnalyzer(config=mock_api_config)

        with patch.object(analyzer._cache, 'get') as mock_get:
            with patch.object(analyzer._cache, 'set') as mock_set:
                mock_get.return_value = None  # Cache miss

                with patch.object(analyzer.model_loader, 'analyze_code') as mock_analyze:
                    mock_analyze.return_value = {"overall_score": 85}
                    analyzer.analyze_code(sample_python_code, "python")

                    # Cache should be checked and set
                    mock_get.assert_called()
