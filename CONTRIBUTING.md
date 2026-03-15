# Contributing to Code Evaluator

Thank you for your interest in contributing to Code Evaluator! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/code_evaluator.git
   cd code_evaluator
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/VanAnh-13/code_evaluator.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install in editable mode with dev dependencies
   ```

3. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys for testing
   ```

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, concise commit messages
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Keep your branch up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit
pytest tests/integration

# Run with coverage
pytest --cov=code_evaluator --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<functionality>_<scenario>`
- Use fixtures from `conftest.py` when possible
- Mock external API calls in unit tests

Example test:

```python
def test_analyze_code_with_valid_input(mock_api_config):
    """Test code analysis with valid Python code"""
    analyzer = CodeAnalyzer(config=mock_api_config)
    result = analyzer.analyze_code("def test(): pass", "python")

    assert result is not None
    assert "overall_score" in result
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_functionality():
    pass

@pytest.mark.integration
def test_integration_workflow():
    pass

@pytest.mark.requires_api
def test_with_real_api():
    pass
```

## Code Style

We follow these style guidelines:

### Python

- **Formatter**: Black (line length: 100)
- **Import sorting**: isort (black profile)
- **Linter**: flake8 (max line length: 120, max complexity: 10)
- **Type hints**: Encouraged but not required

### Running Code Quality Tools

```bash
# Format code
black code_evaluator tests

# Sort imports
isort code_evaluator tests

# Lint code
flake8 code_evaluator tests

# Type check (optional)
mypy code_evaluator
```

### Pre-commit

If you installed pre-commit hooks, these checks run automatically before each commit.

## Submitting Changes

1. **Run all tests and checks**:
   ```bash
   pytest
   black code_evaluator tests --check
   isort code_evaluator tests --check
   flake8 code_evaluator tests
   ```

2. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `chore:` Maintenance tasks

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**:
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Link any related issues

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation if needed
- Add a clear description of changes
- Reference related issues (e.g., "Fixes #123")
- Ensure CI checks pass

## Reporting Issues

### Bug Reports

Include:
- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, dependencies
- **Logs**: Relevant error messages or logs

### Feature Requests

Include:
- **Description**: Clear description of the feature
- **Use case**: Why this feature is needed
- **Proposed solution**: How you envision it working
- **Alternatives**: Other approaches you've considered

## Development Guidelines

### Adding a New LLM Provider

1. Create client in `code_evaluator/model/`:
   ```python
   from code_evaluator.model.base_client import BaseLLMClient

   class NewProviderClient(BaseLLMClient):
       def chat(self, messages, **kwargs):
           # Implementation
           pass
   ```

2. Update `config.py` with default model
3. Update `factory.py` to instantiate client
4. Add tests in `tests/unit/model/`
5. Update documentation

### Adding a New Feature

1. Discuss in an issue first (for major features)
2. Write tests before implementation (TDD)
3. Implement the feature
4. Update documentation
5. Add examples if applicable

### Project Structure

```
code_evaluator/
├── code_evaluator/       # Main package
│   ├── agent/           # Agent functionality
│   ├── analyzer/        # Code analysis
│   ├── model/           # LLM clients
│   ├── report/          # Report generation
│   ├── utils/           # Utilities
│   └── web/             # Web interface
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── examples/            # Example files
├── prompts/            # LLM prompts
└── docs/               # Documentation
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Review the documentation

Thank you for contributing to Code Evaluator! 🎉
