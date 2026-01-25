# Contributing to BioBERT Attention Visualization Tool

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bio-attn-viz-clean.git
   cd bio-attn-viz-clean
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies (including dev dependencies):
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Creating a Branch

Create a branch for your changes:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run these locally before committing:
```bash
# Format code
black .

# Run linter
ruff check .

# Type checking
mypy .
```

Or let pre-commit handle it automatically on each commit.

### Running Tests

Run the test suite with pytest:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_visualize.py
```

### Commit Messages

Use clear, descriptive commit messages:
- `feat: add support for PubMedBERT model`
- `fix: handle empty attention weights gracefully`
- `docs: update installation instructions`
- `test: add tests for entity detection`

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add tests for new functionality
4. Submit your PR with a clear description of changes
5. Link any related issues

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type hints added for new code
- [ ] Documentation updated if needed

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Any implementation ideas

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open an issue with the "question" label or start a discussion.

Thank you for contributing!
