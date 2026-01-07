# Contributing to RAG-Lite

Thank you for your interest in contributing to RAG-Lite! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- pip package manager

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-lite-tfidf-eval.git
   cd rag-lite-tfidf-eval
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[all]"
   ```

4. **Install Pre-commit Hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, concise code
- Follow existing code style and conventions
- Add docstrings to functions and classes
- Keep functions focused and modular

### 3. Add Tests

```bash
# Create test file in tests/ directory
# tests/test_your_feature.py

def test_your_feature():
    # Test implementation
    assert True
```

Run tests:
```bash
pytest tests/test_your_feature.py -v
```

### 4. Run Code Quality Checks

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Run all tests
pytest --cov=src

# Type checking (if applicable)
mypy src
```

### 5. Commit Changes

Write clear commit messages following conventional commits:

```bash
git add .
git commit -m "feat: add new retrieval method"
# or
git commit -m "fix: resolve query caching issue"
# or
git commit -m "docs: update API documentation"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Summary of changes
- Test results

## 🧪 Testing Guidelines

### Writing Tests

- Use `pytest` for all tests
- Place tests in `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names

**Example:**
```python
def test_retrieve_returns_correct_number_of_results():
    """Test that retrieve returns k results."""
    from src.rag import build_index, retrieve
    
    passages = ["test1", "test2", "test3"]
    index = build_index(passages)
    results = retrieve(index, "test", k=2)
    
    assert len(results) == 2
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_retrieval.py

# Specific test
pytest tests/test_retrieval.py::test_specific_function

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only (skip slow tests)
pytest -m "not slow"

# Verbose mode
pytest -v
```

### Test Coverage

Aim for >80% code coverage. Check coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

## 📖 Documentation

### Docstrings

Use Google-style docstrings:

```python
def retrieve(index: Index, query: str, k: int = 10, method: str = "tfidf") -> list[tuple[str, float]]:
    """
    Retrieve top-k most relevant passages for a query.
    
    Args:
        index: Pre-built retrieval index
        query: Query string
        k: Number of results to return
        method: Retrieval method (tfidf, bm25, embeddings)
    
    Returns:
        List of (passage, score) tuples sorted by relevance
    
    Raises:
        ValueError: If method is not supported
    
    Example:
        >>> index = build_index(passages)
        >>> results = retrieve(index, "machine learning", k=5)
    """
```

### README Updates

Update README.md when:
- Adding new features
- Changing CLI/API interface
- Updating installation instructions
- Adding new configuration options

## 🎨 Code Style

### Python Style Guide

Follow PEP 8 and project conventions:

- **Line length**: 100 characters (enforced by Black)
- **Imports**: Use absolute imports, group by stdlib/third-party/local
- **Type hints**: Use for function signatures
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Black Formatting

All code is formatted with Black:
```bash
black src tests
```

### Ruff Linting

Lint code with Ruff:
```bash
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```

## 🏗️ Architecture Guidelines

### Adding New Retrieval Methods

1. Implement in `src/rag.py`
2. Add tests in `tests/test_retrieval.py`
3. Update CLI in `src/cli.py`
4. Update API in `src/api.py`
5. Add to documentation

### Adding New Features

- Keep modules focused and cohesive
- Use dependency injection where appropriate
- Add configuration options to `config.yaml`
- Update type hints and docstrings

### Performance Considerations

- Profile before optimizing
- Add benchmarks for critical paths
- Cache expensive computations
- Consider memory vs speed trade-offs

## 🐛 Bug Reports

### Creating Issues

Include:
1. **Clear title**: Describe the bug concisely
2. **Environment**: OS, Python version, package versions
3. **Steps to reproduce**: Minimal reproducible example
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Logs/errors**: Full error messages and stack traces

**Example:**
```markdown
## Bug: Query returns wrong number of results

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.11.5
- rag-lite: 0.1.0

**Steps to reproduce:**
1. Build index with 100 passages
2. Run query with k=5
3. Receive 10 results instead

**Expected:** 5 results
**Actual:** 10 results

**Error log:**
```
[error log here]
```
```

## 🎯 Feature Requests

When requesting features:
1. Describe the use case
2. Explain why it's needed
3. Suggest possible implementation
4. Consider alternatives

## 📦 Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create git tag
5. Build and publish to PyPI

## 💬 Communication

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Features**: Create an Issue or Discussion
- **Security**: Email maintainers directly

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## 🙏 Recognition

Contributors will be:
- Listed in release notes
- Mentioned in CHANGELOG.md
- Added to contributors list

## 📚 Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## ❓ Questions?

If you have questions not covered here:
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Open a new issue

Thank you for contributing! 🎉
