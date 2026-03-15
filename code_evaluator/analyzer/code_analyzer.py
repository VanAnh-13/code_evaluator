"""
Code Analyzer for Code Evaluator
Main analyzer class that orchestrates code analysis using LLM API
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from code_evaluator.model.loader import ModelLoader
from code_evaluator.model.config import APIConfig
from code_evaluator.analyzer.syntax_checker import check_syntax
from code_evaluator.analyzer.parser import parse_json_response, parse_analysis
from code_evaluator.utils.cache import ContentCache
from code_evaluator.utils.file_utils import detect_language, read_file_content
from code_evaluator.utils.security import validate_path

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    A class to analyze code using LLM API providers.
    Supports multiple programming languages with auto-detection.
    """

    def __init__(self, config: Optional[APIConfig] = None, model_name: str = ""):
        """
        Initialize the analyzer.

        Args:
            config: API configuration. If None, loads from environment.
            model_name: Legacy parameter (ignored if config is provided).
        """
        if config is None:
            config = APIConfig.from_env()

        self.model_loader = ModelLoader(config=config)
        self._cache = ContentCache()

        # Load prompt template and output schema
        self._system_prompt = ""
        self._output_schema = ""
        self._load_prompts()

    @property
    def model_name(self) -> str:
        """Get the model name"""
        return self.model_loader.model_name

    @property
    def model_loaded(self) -> bool:
        """Check if the API client is initialized"""
        return self.model_loader.is_loaded

    def _load_prompts(self):
        """Load the universal prompt template and output schema."""
        # Find prompts directory
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "prompts"
        )
        prompts_dir = os.path.normpath(prompts_dir)

        if not os.path.exists(prompts_dir):
            prompts_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts"
            )
            prompts_dir = os.path.normpath(prompts_dir)

        # Load output schema
        schema_path = os.path.join(prompts_dir, "output_schema.json")
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                self._output_schema = f.read()
        except FileNotFoundError:
            logger.warning(f"Output schema not found: {schema_path}")
            self._output_schema = ""

        # Load universal prompt
        prompt_path = os.path.join(prompts_dir, "universal.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._system_prompt = f.read()
        except FileNotFoundError:
            logger.warning(f"Universal prompt not found: {prompt_path}. Using fallback.")
            self._system_prompt = self._get_fallback_prompt()

        # Inject output schema instruction into prompt
        schema_instruction = ""
        if self._output_schema:
            schema_instruction = (
                "\n\n**OUTPUT FORMAT**: You MUST respond with ONLY a valid JSON object "
                "following this exact schema (no markdown, no extra text):\n\n"
                f"```json\n{self._output_schema}\n```"
            )
        self._system_prompt = self._system_prompt.replace(
            "{output_schema}", schema_instruction
        )

    def _get_fallback_prompt(self) -> str:
        """Get a fallback prompt if the external file is not available."""
        return (
            "You are an expert code reviewer. Analyze the following code for bugs, "
            "memory issues, security vulnerabilities, performance problems, and style issues. "
            "Respond with a JSON object containing: language (string), summary (string), "
            "overall_score (0-100), issues (array of {line, category, severity, description, recommendation}), "
            "and suggested_fixes (array of {line, original, fixed, explanation}).\n\n"
            "{output_schema}\n\n"
            "Code to analyze:\n```\n{code}\n```"
        )

    def load_model(self) -> bool:
        """
        Initialize the API client.

        Returns:
            True if successful, False otherwise.
        """
        return self.model_loader.load()

    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze the given code using the LLM API.

        Args:
            code: Code to analyze
            language: Programming language of the code

        Returns:
            Dictionary containing analysis results
        """
        # Check for syntax errors first
        syntax_errors = check_syntax(code, language)

        # Initialize results structure
        analysis_results = {
            "language": language,
            "syntax_errors": syntax_errors,
            "bugs": [],
            "memory_issues": [],
            "security_vulnerabilities": [],
            "performance_issues": [],
            "style_issues": [],
            "summary": "",
            "overall_score": 0,
            "suggested_fixes": {},
        }

        # Check if API client is loaded
        if not self.model_loaded:
            print("[WARNING] API client not initialized. Initializing now...")
            if not self.load_model():
                print("[ERROR] Failed to initialize API. Returning only syntax analysis.")
                return analysis_results

        try:
            print("[INFO] Sending code for analysis...")

            # Build messages for the API
            system_message = self._system_prompt.replace("{code}", "").rstrip()
            user_message = f"Please analyze this code:\n\n```\n{code}\n```"

            # If system prompt contains {code} placeholder, use it directly
            if "{code}" in self._system_prompt:
                system_content = self._system_prompt.replace("{code}", code)
                messages = [
                    {"role": "system", "content": "You are an expert code reviewer. Respond only with valid JSON."},
                    {"role": "user", "content": system_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]

            # Call the API
            response = self.model_loader.analyze(
                messages=messages,
                json_mode=True,
            )

            # Parse JSON response
            parsed_results = parse_json_response(response)

            # Update results with parsed analysis
            for key in ["bugs", "memory_issues", "security_vulnerabilities",
                       "performance_issues", "style_issues"]:
                if key in parsed_results and parsed_results[key]:
                    analysis_results[key] = parsed_results[key]

            # Update summary, score, and fixes
            if parsed_results.get("summary"):
                analysis_results["summary"] = parsed_results["summary"]
            if parsed_results.get("overall_score"):
                analysis_results["overall_score"] = parsed_results["overall_score"]
            if parsed_results.get("suggested_fixes"):
                analysis_results["suggested_fixes"] = parsed_results["suggested_fixes"]
            if parsed_results.get("detected_language"):
                analysis_results["detected_language"] = parsed_results["detected_language"]

            total_issues = sum(
                len(analysis_results[k])
                for k in ['bugs', 'memory_issues', 'security_vulnerabilities',
                          'performance_issues', 'style_issues']
            )
            print(f"[INFO] Analysis completed. Found {total_issues} issues. Score: {analysis_results['overall_score']}/100")

        except TimeoutError as e:
            print(f"[ERROR] {str(e)}")
            analysis_results["error"] = str(e)

        except Exception as e:
            print(f"[ERROR] Failed to generate analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return analysis_results

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Backward compatibility wrapper for parse_analysis"""
        return parse_analysis(analysis_text)

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a code file.

        Args:
            file_path: Path to the code file

        Returns:
            Dictionary containing analysis results
        """
        # Path traversal protection
        is_valid, error_msg = validate_path(file_path)
        if not is_valid:
            return {"error": error_msg, "file_path": file_path}

        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "file_path": file_path}

        # Detect language based on file extension
        language = detect_language(file_path)
        if language == "unknown":
            print(f"[WARNING] Could not determine language for file {file_path}. Using generic analysis.")
            language = "default"

        # Read file content
        code, error_msg = read_file_content(file_path)
        if code is None:
            return {"error": error_msg, "file_path": file_path}

        if not code.strip():
            return {
                "error": "The file is empty. Please upload a file with code content to analyze.",
                "file_path": file_path
            }

        # Check cache
        cached_result = self._cache.get(file_path, code)
        if cached_result is not None:
            print(f"[INFO] Using cached analysis for {file_path}")
            return cached_result

        try:
            results = self.analyze_code(code, language)
            results["file_path"] = file_path
            results["language"] = language

            # Cache the results
            self._cache.set(file_path, code, results)
            return results
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[ERROR] {error_msg} - {file_path}")
            return {
                "error": f"{error_msg}. Please try again or contact support.",
                "file_path": file_path,
            }

    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        self._cache.clear()
