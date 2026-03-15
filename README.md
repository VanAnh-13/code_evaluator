# Code Analyzer

A multi-language code analysis tool — detects bugs, security vulnerabilities, performance issues, and suggests fixes.

> **This project has 2 versions:** v1 (self-hosted Qwen model) and v2 (API KEY — OpenAI / Anthropic / Gemini). See details below.

---

## Version Comparison

| | **v1 — Self-hosted Model** | **v2 — API KEY** |
|---|---|---|
| **LLM** | Qwen-7B-Chat (self-hosted locally) | OpenAI GPT-4o / Anthropic Claude / Google Gemini |
| **Hardware Requirements** | NVIDIA GPU (VRAM ≥ 8GB) or CPU (very slow) | No GPU needed — only internet + API key |
| **Web Interface** | ChatGPT-like UI, sidebar, file upload | Modern dashboard, split-panel, CodeMirror editor |
| **Code Input** | Upload file (.cpp, .py, .js...) | Paste directly / drag & drop / upload |
| **Output Format** | Markdown (parsed with regex) | JSON structured (with schema) |
| **Theme** | Dark only | Dark / Light toggle |
| **Fine-tuning** | LoRA fine-tuning supported | Not available (uses provider's pre-trained models) |
| **Dependencies** | ~8GB (torch, transformers, modelscope...) | ~50MB (openai, anthropic, google-generativeai) |
| **Prompts** | 4 separate language-specific files (cpp.txt, python.txt...) | 1 universal multi-language prompt |
| **API Endpoint** | None | `POST /api/analyze` (JSON) |

---

# V1 — Self-hosted Qwen Model (Original Version)

## Features (v1)

- Multi-language code analysis (C++, Python, JavaScript, Java...)
- Detects bugs, memory leaks, security vulnerabilities, performance issues
- Suggested fixes via `--fix` flag
- LoRA fine-tuning on custom datasets
- ChatGPT-like web UI (sidebar + chat messages)
- Analysis result caching
- JSON and Markdown report export

## Installation (v1)

```bash
git clone https://github.com/VanAnh-13/code_evaluator.git
cd code_evaluator

# Install dependencies (recommended)
python install_dependencies.py

# Or install manually
pip install -r requirements.txt
```

### Dependencies (v1)

```
transformers==4.38.2
torch==2.2.2
modelscope==1.9.5
sentencepiece==0.1.99
colorama==0.4.6
tqdm==4.66.1
numpy==1.24.4
flask==2.3.3
werkzeug==2.3.7
flask-wtf==1.2.1
wtforms==3.0.1
python-dotenv==1.0.0
# Fine-tuning
peft==0.7.1
datasets==2.16.1
accelerate==0.25.0
bitsandbytes==0.41.1
scipy==1.11.4
```

> ⚠️ Total dependency size ~8GB (including PyTorch + model weights).

## Usage (v1)

### CLI

```bash
# Analyze a file
python code_analyzer.py path/to/file.cpp

# Analyze multiple files
python code_analyzer.py file1.cpp file2.py file3.js

# Save JSON results
python code_analyzer.py path/to/file.cpp --output results_dir

# Save Markdown report
python code_analyzer.py path/to/file.py --report reports_dir

# Suggest fixes
python code_analyzer.py path/to/file.js --fix

# Specify a different model
python code_analyzer.py path/to/file.cpp --model Qwen/Qwen-14B-Chat

# Verbose output
python code_analyzer.py path/to/file.py --verbose
```

### Web UI (v1)

```bash
# Windows
run_web.bat
# or
python run_web.py

# Linux/Mac
./run_web.sh
# or
python3 run_web.py
```

Open **http://localhost:5000** — ChatGPT-like interface:
1. Select a code file from your computer (upload)
2. Click Analyze
3. View analysis results as chat messages
4. Browse history from the sidebar

### Fine-tuning (v1)

```bash
# Create sample dataset
python finetune.py --create_sample --sample_path my_dataset.json --num_samples 20

# Fine-tune with LoRA
python finetune.py --data_path my_dataset.json --output_dir fine-tuned-model --use_lora

# Fine-tune with 8-bit quantization
python finetune.py --data_path my_dataset.json --output_dir fine-tuned-model \
  --use_lora --load_in_8bit

# Fine-tune with custom hyperparameters
python finetune.py --data_path my_dataset.json --output_dir fine-tuned-model \
  --use_lora --lora_r 16 --lora_alpha 32 --learning_rate 1e-5 \
  --num_train_epochs 5 --per_device_train_batch_size 2 --fp16

# Use the fine-tuned model
python code_analyzer.py path/to/file.cpp --model fine-tuned-model
```

#### Fine-tuning Dataset Format

```json
[
  {
    "language": "cpp",
    "code": "// C++ code here",
    "analysis": "Detailed analysis..."
  },
  {
    "language": "python",
    "code": "# Python code here",
    "analysis": "Detailed analysis..."
  }
]
```

### Docker (v1)

```bash
docker build -t code-analyzer .
docker run -v $(pwd):/data code-analyzer /data/your_file.cpp
docker run -p 5000:5000 code-analyzer
```

### Architecture (v1)

```
code_evaluator/
├── model/
│   ├── loader.py         # Load Qwen model via transformers + torch
│   └── __init__.py
├── analyzer/
│   ├── code_analyzer.py  # Tokenize → generate → decode → regex parse
│   ├── parser.py         # Regex-based markdown parser
│   ├── syntax_checker.py
│   └── fix_suggester.py
├── finetune/             # LoRA fine-tuning pipeline
│   ├── trainer.py
│   ├── dataset.py
│   └── cli.py
├── web/
│   ├── app.py
│   ├── routes.py         # Upload file → analyze → render template
│   ├── templates/        # ChatGPT-like UI (sidebar + chat)
│   └── static/
├── report/
├── utils/
└── main.py

prompts/
├── cpp.txt               # C++-specific prompt
├── python.txt            # Python-specific prompt
├── javascript.txt        # JavaScript-specific prompt
├── default.txt           # Default prompt
└── output_format.txt     # Output format instructions
```

### Output Format (v1 — Markdown)

```
## Bugs and Logical Errors
- Line 15 (high): Potential null pointer dereference
  Recommendation: Add null check before dereferencing

## Memory Management Issues
- Line 23 (critical): Memory leak: allocated memory not freed
  Recommendation: Add delete[] or use smart pointers

## Security Vulnerabilities
- Line 42 (medium): Buffer overflow risk in strcpy
  Recommendation: Use strncpy or std::string instead
```

---

# V2 — API KEY (Current Version)

## Features (v2)

- **Multi-language** — C/C++, Python, JavaScript, Java, Go, Rust, Ruby, PHP, Swift, HTML, CSS
- **Multi-provider** — OpenAI (GPT-4o), Anthropic (Claude), Google Gemini — switch seamlessly
- **Modern dashboard** — Split-panel with CodeMirror editor, score ring, issue cards
- **Dark / Light theme** — Automatically saves preference, instant toggle
- **Realtime analysis** — Paste code / drag & drop files, get results via AJAX
- **JSON structured output** — Schema-validated results, easy to integrate
- **Detects 6 issue categories**: Syntax, Bugs, Memory, Security, Performance, Style
- **Score 0-100** with summary and specific fix suggestions
- **Result caching** to avoid duplicate analysis
- **CLI** for batch file analysis, JSON/Markdown export
- **API endpoint** `POST /api/analyze` for external integration

## Installation (v2)

```bash
git clone https://github.com/VanAnh-13/code_evaluator.git
cd code_evaluator
pip install -r requirements.txt
```

### Dependencies (v2)

```
flask==2.3.3
werkzeug==2.3.7
flask-wtf==1.2.1
python-dotenv==1.0.0
openai>=1.30.0
anthropic>=0.26.0
google-generativeai>=0.5.0
```

> ✅ Only ~50MB — no torch or GPU required.

### API Key Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Provider: openai | anthropic | gemini
API_PROVIDER=openai

# API key (required)
API_KEY=sk-your-api-key-here

# Model (optional, leave empty for default)
API_MODEL=
```

**Default models per provider:**

| Provider | Default Model | Other Models |
|----------|---------------|--------------|
| OpenAI | `gpt-4o-mini` | `gpt-4o`, `gpt-4-turbo` |
| Anthropic | `claude-sonnet-4-20250514` | `claude-3-haiku-20240307` |
| Gemini | `gemini-2.0-flash` | `gemini-1.5-pro` |

## Usage (v2)

### Web Interface (Recommended)

```bash
# Windows
python run_web.py
# or
run_web.bat

# Linux/Mac
python3 run_web.py
# or
./run_web.sh
```

Open **http://localhost:5000**

#### Dashboard Interface

- **Left panel**: CodeMirror editor — paste code, select language, or drag & drop files
- **Right panel**: Analysis results — score ring, summary, issues by category
- **Top bar**: Page navigation (New Analysis / History), dark/light theme toggle, active provider

#### Workflow

1. Paste code into the editor or drag & drop a file (auto-detects language)
2. Select language from dropdown (or leave as Auto Detect)
3. Click **Analyze** — results appear in realtime via AJAX
4. View score, summary, and issues by tab (Bugs, Security, Performance...)
5. View suggested fixes if available
6. Browse analysis history on the History page

### CLI (v2)

```bash
# Analyze a single file
python -m code_evaluator.main analyze path/to/file.cpp

# Analyze multiple files
python -m code_evaluator.main analyze file1.py file2.js file3.cpp

# Save JSON results
python -m code_evaluator.main analyze path/to/file.py --output results/

# Save Markdown report
python -m code_evaluator.main analyze path/to/file.py --report reports/

# Specify provider and model via CLI
python -m code_evaluator.main analyze path/to/file.cpp \
  --provider openai --api-key sk-xxx --api-model gpt-4o

# Verbose output
python -m code_evaluator.main analyze path/to/file.py -v
```

### Docker (v2)

```bash
# Build image (lightweight, no CUDA needed)
docker build -t code-analyzer .

# Run web server
docker run -p 5000:5000 \
  -e API_PROVIDER=openai \
  -e API_KEY=sk-your-key \
  code-analyzer
```

### Architecture (v2)

```
code_evaluator/
├── model/              # API provider layer
│   ├── config.py       # APIConfig — loads from env vars
│   ├── base_client.py  # BaseLLMClient ABC
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   ├── factory.py      # create_client() factory
│   └── loader.py       # ModelLoader wrapper
├── analyzer/
│   ├── code_analyzer.py  # Orchestrator — build messages → API call → parse
│   ├── parser.py       # JSON parser + regex fallback
│   ├── syntax_checker.py
│   └── fix_suggester.py
├── web/
│   ├── app.py          # App factory + dotenv
│   ├── routes.py       # Routes + POST /api/analyze
│   ├── validators.py
│   ├── templates/      # Dashboard UI (CodeMirror, score ring, tabs)
│   └── static/         # Dual-theme CSS + AJAX JS
├── report/
├── utils/
└── main.py

prompts/
├── universal.txt       # Single multi-language prompt (replaces 4 old files)
└── output_schema.json  # JSON Schema for output format
```

### Data Flow (v2)

```
User (CodeMirror Editor / CLI)
  → POST /api/analyze { code, language }
    → CodeAnalyzer.analyze_code()
      → Build chat messages (system prompt + user code)
      → ModelLoader.analyze() → BaseLLMClient.chat()
        → OpenAI / Anthropic / Gemini API
      → parse_json_response() → structured results
    → JSON response → Render dashboard (AJAX)
```

### Output Format (v2 — JSON)

```json
{
  "language": "python",
  "summary": "Code is simple and functional with minor style issues.",
  "overall_score": 85,
  "issues": [
    {
      "line": 2,
      "category": "style",
      "severity": "low",
      "description": "Missing docstring for function",
      "recommendation": "Add a docstring describing the function's purpose"
    }
  ],
  "suggested_fixes": [
    {
      "line": 1,
      "original": "def hello():",
      "fixed": "def hello():\n    \"\"\"Print greeting.\"\"\"",
      "explanation": "Added docstring"
    }
  ]
}
```

### API Endpoint

#### `POST /api/analyze`

Analyze code directly via JSON API (used by the frontend AJAX or external integrations):

**Request:**
```json
{
  "code": "def hello():\n    print('world')",
  "language": "python"
}
```

**Response:** (see Output Format v2 above)

### Advanced Configuration (v2)

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `API_PROVIDER` | Provider: `openai`, `anthropic`, `gemini` | `openai` |
| `API_KEY` | API key (required) | — |
| `API_MODEL` | Model name | Default per provider |
| `API_TEMPERATURE` | Text generation temperature (0.0 – 1.0) | `0.1` |
| `API_MAX_TOKENS` | Maximum tokens for response | `4096` |
| `API_TIMEOUT` | Timeout (seconds) | `120` |
| `API_BASE_URL` | Custom base URL (proxy, compatible API) | — |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `PORT` | Web server port | `5000` |

---

## Key Changes from v1 → v2

| Component | v1 | v2 |
|-----------|----|----|
| `model/loader.py` | Load Qwen via `transformers` + `torch` | Wraps `BaseLLMClient` (API call) |
| `model/` | Only `loader.py` | Added `config.py`, `base_client.py`, 3 provider clients, `factory.py` |
| `analyzer/parser.py` | Regex parse markdown output | `parse_json_response()` + regex fallback |
| `analyzer/code_analyzer.py` | Tokenize → generate → decode | Build chat messages → API call |
| `prompts/` | 4 separate files + `output_format.txt` | `universal.txt` + `output_schema.json` |
| Web templates | ChatGPT-like (sidebar + chat) | Dashboard (top bar + split panel + CodeMirror) |
| CSS | Dark only, sidebar layout | Dual theme (CSS variables), dashboard layout |
| JS | File upload + hljs highlight | CodeMirror editor + AJAX + drag-drop + theme toggle |
| `routes.py` | `POST /upload` (form) | Added `POST /api/analyze` (JSON API) |
| `requirements.txt` | torch, transformers, modelscope... (~8GB) | openai, anthropic, google-generativeai (~50MB) |
| `main.py` CLI | `--model Qwen/Qwen-7B-Chat`, `--fix` | `--provider`, `--api-key`, `--api-model` |
| Fine-tuning | Yes (`finetune/` module) | No (uses provider's pre-trained models) |
| Dockerfile | Multi-stage, requires g++, ~4GB image | Single stage, ~200MB image |

## System Requirements

### v1
- Python 3.8+
- NVIDIA GPU (VRAM ≥ 8GB) or CPU (slow)
- ~10GB disk (dependencies + model weights)

### v2
- Python 3.8+
- API key from at least 1 provider (OpenAI / Anthropic / Google)
- No GPU required

### Syntax Checker (Optional, both versions)

- **C/C++**: Install `g++` (MinGW on Windows, `build-essential` on Linux)
- **Python**: `pip install pylint`
- **JavaScript**: `npm install -g eslint`
- **Java**: Install JDK (includes `javac`)

## Troubleshooting

### V1

**`CUDA out of memory`**
- Use `--load_in_8bit` or `--load_in_4bit` when fine-tuning
- Reduce `--max_length` and `--per_device_train_batch_size`

**Model download slow / fails**
- Check network connectivity to HuggingFace and ModelScope
- Try `--model` pointing to a local model directory

**Wheel build errors (torch, sentencepiece...)**
- Windows: Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Linux: `sudo apt-get install build-essential`
- Or use `python install_dependencies.py` (handles fallback automatically)

### V2

**`API client failed to initialize`**
- Verify `API_KEY` in your `.env` file
- Verify `API_PROVIDER` is correct (`openai` / `anthropic` / `gemini`)

**`ModuleNotFoundError: No module named 'openai'`**
```bash
pip install -r requirements.txt
```

**Web server won't start**
```bash
python -c "from code_evaluator.web import create_app; print('OK')"
python run_web.py
```

**Results fail to parse as JSON**
- Switch to a larger model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
- The parser automatically falls back to regex if JSON is invalid

## License

MIT
