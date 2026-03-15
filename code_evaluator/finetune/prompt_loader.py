"""
Prompt loader for fine-tuning
Loads prompt templates from external files (shared with analyzer module)
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def get_prompts_dir() -> str:
    """
    Get the path to the prompts directory

    Returns:
        Path to prompts directory
    """
    # Look for prompts directory relative to the package
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompts_dir = os.path.join(base_dir, "prompts")
    
    if os.path.exists(prompts_dir):
        return prompts_dir
    
    # Fallback
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts")


def load_prompt_templates() -> Dict[str, str]:
    """
    Load prompt templates from the prompts/ directory.

    Returns:
        Dictionary mapping language names to prompt template strings.
    """
    prompts_dir = get_prompts_dir()

    # Load shared output format
    output_format_path = os.path.join(prompts_dir, "output_format.txt")
    try:
        with open(output_format_path, "r", encoding="utf-8") as f:
            output_format = f.read()
    except FileNotFoundError:
        logger.warning(f"Output format file not found: {output_format_path}. Using empty format.")
        output_format = ""

    # Load all language prompt files
    templates = {}
    try:
        for filename in os.listdir(prompts_dir):
            if filename == "output_format.txt" or not filename.endswith(".txt"):
                continue
            lang = filename.rsplit(".", 1)[0]
            filepath = os.path.join(prompts_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                template = f.read()
            templates[lang] = template.replace("{output_format}", output_format)
    except FileNotFoundError:
        logger.warning(f"Prompts directory not found: {prompts_dir}. Using fallback prompts.")

    # Ensure there's always a default template
    if "default" not in templates:
        templates["default"] = _get_fallback_prompt("default")

    return templates


def _get_fallback_prompt(language: str) -> str:
    """
    Get fallback prompt template if external files are not available

    Args:
        language: Programming language

    Returns:
        Prompt template string
    """
    if language == "cpp":
        return """You are an expert C++ code analyzer. Analyze the following C++ code for:
1. Potential bugs and logical errors
2. Memory management issues (leaks, dangling pointers, etc.)
3. Security vulnerabilities
4. Performance issues
5. Code style and readability issues

Provide a detailed analysis with specific line numbers and recommendations for improvement.

C++ CODE:
```cpp
{code}
```

ANALYSIS:
"""
    elif language == "python":
        return """You are an expert Python code analyzer. Analyze the following Python code for:
1. Potential bugs and logical errors
2. Memory and resource management issues
3. Security vulnerabilities
4. Performance issues
5. Code style and readability issues (PEP 8 compliance)

Provide a detailed analysis with specific line numbers and recommendations for improvement.

PYTHON CODE:
```python
{code}
```

ANALYSIS:
"""
    elif language == "javascript":
        return """You are an expert JavaScript code analyzer. Analyze the following JavaScript code for:
1. Potential bugs and logical errors
2. Memory leaks and resource management
3. Security vulnerabilities
4. Performance issues
5. Code style and readability issues

Provide a detailed analysis with specific line numbers and recommendations for improvement.

JAVASCRIPT CODE:
```javascript
{code}
```

ANALYSIS:
"""
    else:
        return """You are an expert code analyzer. Analyze the following code for:
1. Potential bugs and logical errors
2. Resource management issues
3. Security vulnerabilities
4. Performance issues
5. Code style and readability issues

Provide a detailed analysis with specific line numbers and recommendations for improvement.

CODE:
```
{code}
```

ANALYSIS:
"""


def create_training_prompt(code: str, language: str, analysis: str = "") -> str:
    """
    Create a training prompt with code and analysis

    Args:
        code: Source code
        language: Programming language
        analysis: Analysis text (for training data)

    Returns:
        Formatted prompt for training
    """
    templates = load_prompt_templates()
    
    # Get template for language or use default
    template = templates.get(language, templates.get("default", _get_fallback_prompt(language)))
    
    # Format the prompt with code
    prompt = template.format(code=code)
    
    # Add analysis if provided
    if analysis:
        prompt = f"{prompt}\n{analysis}"
    
    return prompt.strip()
