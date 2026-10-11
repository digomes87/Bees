# Contributing to Bee Migration Analysis Project

Thank you for your interest in contributing to our bee migration analysis project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Bees.git`
3. Create a new branch from `develop`: `git switch -c feature/your-feature-name develop`
4. Make your changes
5. Test your changes
6. Submit a pull request against `develop`

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip or conda package manager
- Git
- Node.js 22 or higher, only to run the dashboard's tests

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/digomes87/Bees.git
cd Bees

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with its development tools
pip install -e ".[dev]"

# Check your changes before every commit
pre-commit install
```

With conda: `conda env create -f environment.yml`.

### Running Tests

```bash
# Run all tests
pytest

# Run one test file, without the coverage threshold
pytest tests/test_analysis.py --no-cov

# Run the dashboard's tests
node --test "tests/web/*.test.mjs"

# Try the dashboard locally
make site
```

## Contributing Guidelines

### Types of Contributions

- **Bug fixes**: Fix existing issues or bugs
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Performance improvements**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Before Contributing

1. Check existing issues and pull requests
2. Create an issue to discuss major changes
3. Ensure your contribution aligns with project goals

## Code Style

### Python Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (ruff formatter)

### Code Formatting

```bash
# Format and lint with ruff
ruff format src/ tests/
ruff check src/ tests/
```

### Documentation Style

- Use clear, concise language
- Include examples where appropriate
- Update README.md if adding new features
- Document any new dependencies

## Testing

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve test coverage
- Tests should be fast and reliable

### Test Structure

```
tests/
├── conftest.py        # shared fixtures
├── test_climate.py    # observations and simulated datasets
├── test_analysis.py   # trends, correlations, regions
├── test_prediction.py # features, models, scores
├── test_sources.py    # GBIF and Open-Meteo answers
├── test_web.py        # studies as JSON for the dashboard
├── test_outputs.py    # reports, charts, command line
├── test_notebook.py   # the tutorial runs end to end
└── web/               # dashboard utilities (node --test)
```

### Writing Tests

Name a test after the behaviour it checks and follow arrange, act, assert:

```python
from bee_migration.analysis.temporal import TemporalPatterns
from bee_migration.climate.observations import Observations
from tests.conftest import region_table


def test_trend_is_the_yearly_slope():
    table = region_table("North", temperature=[10.0, 10.5, 11.0, 11.5])

    trends = TemporalPatterns(Observations(table)).trends()

    assert trends.loc["North", "temperature"] == 0.5
```

## Writing Tests

```python
import pytest
from src.module import function_to_test


def test_function_behavior():
    """Test that function behaves correctly."""
    # Arrange
    input_data = "test_input"
    expected_output = "expected_result"

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output
```

## Submitting Changes

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(analysis): add temperature correlation analysis
fix(visualization): correct geographic projection
docs(readme): update installation instructions
```

### Pull Request Process

1. **Create a descriptive title**
2. **Fill out the PR template**
3. **Link related issues**
4. **Ensure all checks pass**
5. **Request review from maintainers**

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow conventional format

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Environment details** (OS, Python version, dependencies)
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Error messages or logs**
- **Minimal code example**

### Feature Requests

When requesting features, include:

- **Use case description**
- **Proposed solution**
- **Alternative solutions considered**
- **Additional context**

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority-high`: High priority issue

## Development Workflow

### Branches

The project follows Git Flow:

| Branch | Purpose | Receives changes from |
|--------|---------|-----------------------|
| `main` | Released code; every push deploys the site | `release/*` and `hotfix/*` pull requests |
| `develop` | Integration of finished work | `feature/*`, `fix/*`, `docs/*`, `refactor/*` pull requests |
| `feature/description` | New features | branched from `develop` |
| `fix/description` | Bug fixes | branched from `develop` |
| `docs/description` | Documentation updates | branched from `develop` |
| `refactor/description` | Code refactoring | branched from `develop` |
| `release/x.y.z` | Release preparation | branched from `develop`, merged into `main` and back into `develop` |
| `hotfix/description` | Urgent fix of released code | branched from `main`, merged into `main` and `develop` |

Nothing is pushed directly to `main` or `develop`. Every change arrives through
a pull request, and the `CI` workflow (lint and tests for Python, tests for the
site) must pass before it is merged.

```bash
# Work on something
git switch -c feature/my-change develop
# ...commit...
git push -u origin feature/my-change
gh pr create --base develop

# Release what is on develop
git switch -c release/1.1.0 develop
gh pr create --base main
```

### Code Review Process

1. **Automated checks** must pass
2. **At least one maintainer** review required
3. **Address feedback** promptly
4. **Squash commits** before merging (if requested)

## Resources

- [Project Documentation](./README.md)
- [Scientific Papers](./results/scientific_papers_synthesis.md)
- [Dashboard](https://digomes87.github.io/Bees/) (source in `./docs/`)
- [Issue Tracker](https://github.com/digomes87/Bees/issues)

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search existing issues
3. Create a new issue with the `question` label
4. Contact maintainers directly

---

Thank you for contributing to the bee migration analysis project! Your contributions help advance our understanding of climate change impacts on bee populations. 🐝
