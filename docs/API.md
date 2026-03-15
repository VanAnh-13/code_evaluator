# Code Evaluator API Documentation

## Overview

Code Evaluator provides three main interfaces for code analysis:
1. **CLI** - Command-line interface for local analysis
2. **Web UI** - Browser-based interface for interactive analysis
3. **REST API** - HTTP endpoints for programmatic integration

## Table of Contents

- [REST API Endpoints](#rest-api-endpoints)
- [Python API](#python-api)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [Response Formats](#response-formats)

---

## REST API Endpoints

### POST /api/analyze

Analyze code and return structured results.

**Request:**
```http
POST /api/analyze HTTP/1.1
Content-Type: application/json

{
  "code": "def divide(a, b):\n    return a / b",
  "language": "python"
}
```

**Request Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | Source code to analyze |
| `language` | string | Yes | Programming language (`python`, `javascript`, `cpp`, `java`, etc.) |

**Response:**
```json
{
  "summary": "Code has potential runtime error",
  "overall_score": 65,
  "issues": [
    {
      "type": "bug",
      "severity": "high",
      "line": 2,
      "description": "Division by zero vulnerability",
      "suggestion": "Add input validation to check for zero divisor"
    }
  ],
  "suggestions": [
    "Implement error handling",
    "Add type hints"
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Brief summary of analysis |
| `overall_score` | integer | Quality score 0-100 |
| `issues` | array | List of identified issues |
| `suggestions` | array | General improvement suggestions |

**Issue Object:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Issue category (`bug`, `security`, `performance`, `style`) |
| `severity` | string | Severity level (`low`, `medium`, `high`, `critical`) |
| `line` | integer | Line number (optional) |
| `description` | string | Issue description |
| `suggestion` | string | Fix recommendation |

**Status Codes:**

- `200 OK` - Analysis completed successfully
- `400 Bad Request` - Invalid request (missing required fields)
- `413 Payload Too Large` - Code exceeds size limit (5MB)
- `500 Internal Server Error` - Server error during analysis

**Example with cURL:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"code":"console.log(userInput);","language":"javascript"}'
```

---

### GET /

Main web interface for interactive code analysis.

**Response:** HTML page with code editor and analysis dashboard

---

### GET /history

View history of previous analyses (session-based).

**Response:** HTML page with analysis history

---

## Python API

### CodeAnalyzer Class

Primary class for code analysis.

```python
from code_evaluator import CodeAnalyzer
from code_evaluator.model.config import APIConfig

# Initialize with configuration
config = APIConfig(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4o-mini",
    temperature=0.3
)

analyzer = CodeAnalyzer(config=config)

# Analyze code string
result = analyzer.analyze_code(
    code="your code here",
    language="python"
)

# Analyze file
result = analyzer.analyze_file("path/to/file.py")

print(f"Score: {result['overall_score']}")
print(f"Issues found: {len(result['issues'])}")
```

**Methods:**

#### `__init__(config: APIConfig = None, model_name: str = "")`

Initialize the analyzer.

**Parameters:**
- `config` (APIConfig, optional): API configuration. If None, loads from environment.
- `model_name` (str, optional): Deprecated, use config.model instead.

#### `analyze_code(code: str, language: str) -> dict`

Analyze a code string.

**Parameters:**
- `code` (str): Source code to analyze
- `language` (str): Programming language

**Returns:** Dictionary with analysis results

#### `analyze_file(file_path: str) -> dict`

Analyze a code file.

**Parameters:**
- `file_path` (str): Path to code file

**Returns:** Dictionary with analysis results

#### `load_model() -> bool`

Explicitly load the LLM client.

**Returns:** True if successful, False otherwise

---

### APIConfig Class

Configuration for LLM providers.

```python
from code_evaluator.model.config import APIConfig

# Load from environment variables
config = APIConfig.from_env()

# Or create manually
config = APIConfig(
    provider="anthropic",  # openai, anthropic, gemini, ollama
    api_key="your-key",
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096,
    timeout=120
)

# Validate configuration
if config.validate():
    print(f"Using {config.provider_display_name}")
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `"openai"` | LLM provider |
| `api_key` | str | `""` | API key (not required for Ollama) |
| `model` | str | Auto | Model name |
| `temperature` | float | `0.3` | Generation temperature |
| `max_tokens` | int | `4096` | Maximum tokens |
| `base_url` | str | None | Custom API base URL |
| `timeout` | int | `120` | Request timeout (seconds) |

**Default Models:**
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-sonnet-4-20250514`
- Gemini: `gemini-2.0-flash`
- Ollama: `codellama`

---

### Report Generation

```python
from code_evaluator.report import generate_report, save_results

# Generate markdown report
report = generate_report(result)
print(report)

# Save JSON results
save_results(result, "output/analysis.json")
```

---

## CLI Commands

### Analyze Command

Analyze one or more code files.

```bash
python -m code_evaluator.main analyze FILE [FILE...] [OPTIONS]
```

**Options:**
- `--provider PROVIDER` - API provider (openai, anthropic, gemini, ollama)
- `--api-key KEY` - API key
- `--api-model MODEL` - Model name
- `--output DIR` - Save JSON results to directory
- `--report DIR` - Save Markdown reports to directory
- `--verbose, -v` - Enable verbose output
- `--no-cache` - Disable caching

**Examples:**

```bash
# Analyze a single file
python -m code_evaluator.main analyze app.py

# Analyze multiple files with output
python -m code_evaluator.main analyze src/*.py --output results --report reports -v

# Use Ollama (local)
python -m code_evaluator.main analyze code.cpp --provider ollama --api-model codellama

# Use specific model
python -m code_evaluator.main analyze script.js --provider anthropic --api-model claude-3-haiku-20240307
```

---

### Serve Command

Start the web server.

```bash
python -m code_evaluator.main serve [OPTIONS]
```

**Options:**
- `--host HOST` - Host to bind (default: 0.0.0.0)
- `--port PORT` - Port to bind (default: 5000)
- `--debug` - Enable debug mode

**Example:**
```bash
python -m code_evaluator.main serve --port 8080
```

---

### Agent Commands

Multi-step AI agent workflows.

#### Agent Analyze

```bash
python -m code_evaluator.main agent analyze FILE [OPTIONS]
```

**Options:**
- `--max-steps N` - Maximum agent steps (default: 15)
- `--output DIR` - Save session results
- `--verbose, -v` - Verbose output

**Example:**
```bash
python -m code_evaluator.main agent analyze complex_code.py --max-steps 20 -v
```

#### Agent Project

```bash
python -m code_evaluator.main agent project DIRECTORY [OPTIONS]
```

**Options:**
- `--max-steps N` - Maximum steps (default: 25)
- `--output DIR` - Save results

**Example:**
```bash
python -m code_evaluator.main agent project ./myproject --output analysis
```

#### Agent Chat

```bash
python -m code_evaluator.main agent chat [OPTIONS]
```

**Options:**
- `--max-steps N` - Max steps per turn (default: 50)
- `--verbose, -v` - Verbose output

**Example:**
```bash
python -m code_evaluator.main agent chat --provider ollama
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Provider configuration
API_PROVIDER=openai
API_KEY=sk-your-api-key
API_MODEL=gpt-4o-mini

# Optional settings
API_TEMPERATURE=0.3
API_MAX_TOKENS=4096
API_TIMEOUT=120

# For Ollama local hosting
# API_PROVIDER=ollama
# API_MODEL=codellama
# API_BASE_URL=http://localhost:11434

# Flask settings
SECRET_KEY=your-secret-key
PORT=5000
```

### Supported Languages

- **Python** (`.py`)
- **JavaScript** (`.js`, `.mjs`)
- **TypeScript** (`.ts`, `.tsx`)
- **C/C++** (`.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`)
- **Java** (`.java`)
- **Go** (`.go`)
- **Rust** (`.rs`)
- **PHP** (`.php`)
- **Ruby** (`.rb`)
- **C#** (`.cs`)

---

## Response Formats

### Analysis Result

```json
{
  "summary": "Overall assessment of code quality",
  "overall_score": 85,
  "issues": [
    {
      "type": "security",
      "severity": "critical",
      "line": 42,
      "description": "SQL injection vulnerability",
      "suggestion": "Use parameterized queries"
    },
    {
      "type": "performance",
      "severity": "medium",
      "line": 15,
      "description": "Inefficient loop",
      "suggestion": "Use list comprehension"
    }
  ],
  "suggestions": [
    "Add input validation",
    "Implement error handling",
    "Add unit tests"
  ]
}
```

### Issue Categories

- **bug** - Logic errors, runtime errors
- **security** - Security vulnerabilities
- **performance** - Performance bottlenecks
- **style** - Code style, readability
- **maintainability** - Code maintainability issues

### Severity Levels

- **critical** - Must fix immediately
- **high** - Should fix soon
- **medium** - Should address
- **low** - Minor improvement

---

## Error Handling

### Common Errors

**API Key Missing:**
```
ValueError: API key is required for OpenAI.
Set the API_KEY environment variable or pass it in the configuration.
```

**Invalid Provider:**
```
ValueError: Unknown provider: 'invalid'.
Supported providers: openai, anthropic, gemini, ollama
```

**Ollama Connection Error:**
```
APIError: Cannot connect to Ollama server at http://localhost:11434.
Make sure Ollama is running (try: ollama serve)
```

**File Too Large:**
```
413 Payload Too Large: File exceeds 5MB limit
```

---

## Rate Limits

Rate limits depend on your LLM provider:

- **OpenAI**: [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- **Anthropic**: [Anthropic Rate Limits](https://docs.anthropic.com/claude/reference/rate-limits)
- **Gemini**: [Gemini Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- **Ollama**: No rate limits (local hosting)

---

## Examples

### Python Integration Example

```python
#!/usr/bin/env python
"""Example: Automated code review in CI/CD pipeline"""

import sys
from code_evaluator import CodeAnalyzer
from code_evaluator.model.config import APIConfig

def main():
    # Configure analyzer
    config = APIConfig.from_env()
    analyzer = CodeAnalyzer(config=config)

    # Analyze files
    files = sys.argv[1:]
    all_issues = []

    for file_path in files:
        result = analyzer.analyze_file(file_path)

        # Collect critical/high issues
        critical_issues = [
            issue for issue in result.get("issues", [])
            if issue["severity"] in ["critical", "high"]
        ]

        all_issues.extend(critical_issues)

        print(f"{file_path}: Score {result['overall_score']}, "
              f"{len(critical_issues)} critical issues")

    # Fail if critical issues found
    if all_issues:
        print(f"\nFound {len(all_issues)} critical issues!")
        sys.exit(1)

    print("\nAll files passed quality check!")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### JavaScript/Node.js Integration

```javascript
// Example: Call Code Evaluator API from Node.js

const axios = require('axios');
const fs = require('fs');

async function analyzeCode(filePath) {
  const code = fs.readFileSync(filePath, 'utf8');
  const language = filePath.split('.').pop();

  const response = await axios.post('http://localhost:5000/api/analyze', {
    code: code,
    language: language
  });

  const result = response.data;
  console.log(`Score: ${result.overall_score}/100`);
  console.log(`Issues: ${result.issues.length}`);

  return result;
}

analyzeCode('app.js')
  .then(result => console.log('Analysis complete:', result))
  .catch(err => console.error('Error:', err));
```

---

## Support

For issues, questions, or contributions:
- **GitHub Issues**: [github.com/VanAnh-13/code_evaluator/issues](https://github.com/VanAnh-13/code_evaluator/issues)
- **Documentation**: See README.md for additional information
