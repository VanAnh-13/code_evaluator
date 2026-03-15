"""Integration tests for CLI commands"""

import pytest
import os
import json
from unittest.mock import patch
from code_evaluator.main import cmd_analyze, cmd_serve


class TestCLIIntegration:
    """Test CLI command integration"""

    @pytest.mark.integration
    def test_analyze_command_basic(self, tmp_path, sample_python_code):
        """Test basic analyze command"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(sample_python_code)

        # Create mock args
        class Args:
            files = [str(test_file)]
            provider = None
            api_key = "test-key"
            api_model = None
            output = None
            report = None
            verbose = False

        with patch('code_evaluator.analyzer.code_analyzer.CodeAnalyzer') as mock_analyzer:
            # Setup mock
            mock_instance = mock_analyzer.return_value
            mock_instance.analyze_file.return_value = {
                "summary": "Test",
                "overall_score": 85,
                "issues": []
            }

            # Run command
            cmd_analyze(Args())

            # Verify analyzer was called
            mock_instance.analyze_file.assert_called()

    @pytest.mark.integration
    def test_analyze_with_output_directory(self, tmp_path, sample_python_code):
        """Test analyze command with output directory"""
        test_file = tmp_path / "test.py"
        test_file.write_text(sample_python_code)

        output_dir = tmp_path / "output"

        class Args:
            files = [str(test_file)]
            provider = None
            api_key = "test-key"
            api_model = None
            output = str(output_dir)
            report = None
            verbose = True

        with patch('code_evaluator.analyzer.code_analyzer.CodeAnalyzer') as mock_analyzer:
            mock_instance = mock_analyzer.return_value
            mock_instance.analyze_file.return_value = {
                "summary": "Test",
                "overall_score": 85,
                "issues": []
            }

            cmd_analyze(Args())

            # Output directory should be created
            assert output_dir.exists()
