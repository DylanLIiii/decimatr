# Contributing to Decimatr

Thank you for your interest in contributing to Decimatr! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/decimatr.git
   cd decimatr
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/DylanLIiii/decimatr.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- Git

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

   For GPU development:
   ```bash
   pip install -e ".[dev,gpu]"
   ```

3. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   python -c "import decimatr; print(decimatr.__version__)"
   pytest tests/ -v
   ```

### Using the Release Script

We provide a convenience script for common development tasks:

```bash
# Show available commands
./scripts/release.sh help

# Run tests
./scripts/release.sh test

# Format code
./scripts/release.sh format

# Run all checks
./scripts/release.sh check

# Build package
./scripts/release.sh build
```

## Making Changes

### Workflow

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes** following our coding standards

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "type: short description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples:**
```bash
feat(filters): add motion blur detection filter
fix(tagger): handle empty frame gracefully
docs: update API documentation for BlurTagger
test(filters): add tests for edge cases in DuplicateFilter
chore: update dependencies
```

## Coding Standards

### Code Style

We use automated tools to maintain consistent code style:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Run formatting before committing:
```bash
./scripts/release.sh format
```

Check code quality:
```bash
./scripts/release.sh lint
```

### Python Guidelines

1. **Type hints**: Use type hints for function signatures
   ```python
   def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def process_frame(self, frame: np.ndarray) -> bool:
       """Process a single frame and determine if it should pass.

       Args:
           frame: Input frame as numpy array (H, W, C).

       Returns:
           True if the frame passes the filter, False otherwise.

       Raises:
           ValueError: If frame dimensions are invalid.
       """
       ...
   ```

3. **Imports**: Group imports in this order:
   - Standard library
   - Third-party packages
   - Local imports

4. **Line length**: Maximum 100 characters

5. **Naming conventions**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private methods: `_leading_underscore`

### Architecture Guidelines

When adding new components, follow the existing architecture:

1. **Taggers** (in `decimatr/taggers/`):
   - Extend `Tagger` base class
   - Implement `compute_tags()` method
   - Define `tag_keys` property
   - Keep stateless - no side effects

2. **Filters** (in `decimatr/filters/`):
   - Extend `StatelessFilter` or `StatefulFilter`
   - Implement `should_pass()` method
   - Define `required_tags` property
   - Use `TemporalBuffer` for stateful operations

3. **Strategies** (in `decimatr/strategies/`):
   - Extend `FilterStrategy`
   - Implement `build_pipeline()` method
   - Document use cases

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=decimatr --cov-report=term-missing

# Run specific test file
pytest tests/filters/test_blur_filter.py -v

# Run specific test
pytest tests/filters/test_blur_filter.py::test_blur_filter_basic -v

# Run tests matching pattern
pytest tests/ -k "blur" -v
```

### Writing Tests

1. **Location**: Place tests in `tests/` directory mirroring source structure
   - `decimatr/filters/blur.py` → `tests/filters/test_blur_filter.py`

2. **Naming**: Use descriptive test names
   ```python
   def test_blur_filter_passes_sharp_frames():
       ...

   def test_blur_filter_rejects_blurry_frames():
       ...
   ```

3. **Structure**: Follow Arrange-Act-Assert pattern
   ```python
   def test_duplicate_filter_detects_identical_frames():
       # Arrange
       filter = DuplicateFilter(threshold=0.05)
       frame1 = create_test_frame(0)
       frame2 = create_test_frame(0)  # Identical

       # Act
       result1 = filter.should_pass(frame1)
       result2 = filter.should_pass(frame2)

       # Assert
       assert result1 is True
       assert result2 is False
   ```

4. **Fixtures**: Use pytest fixtures for common setup
   ```python
   @pytest.fixture
   def sample_frames():
       return [create_test_frame(i) for i in range(10)]
   ```

5. **Coverage**: Aim for >70% coverage on new code

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Test performance characteristics

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   ./scripts/release.sh test
   ```

2. **Ensure code quality checks pass**:
   ```bash
   ./scripts/release.sh lint
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

5. **Update CHANGELOG.md** for significant changes

### PR Guidelines

1. **Title**: Use conventional commit format
   ```
   feat(filters): add Gaussian blur detection
   ```

2. **Description**: Include:
   - Summary of changes
   - Motivation and context
   - How to test
   - Screenshots (if UI changes)

3. **Size**: Keep PRs focused and reasonably sized
   - Large changes should be split into smaller PRs

4. **Draft PRs**: Use draft PRs for work-in-progress

### Review Process

1. All PRs require at least one review
2. CI checks must pass
3. Address review feedback promptly
4. Squash commits if requested

### After Merging

- Delete your feature branch
- Sync your fork with upstream

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Title**: Clear, concise description
2. **Environment**:
   - Python version
   - Operating system
   - Decimatr version
   - GPU (if applicable)
3. **Steps to reproduce**
4. **Expected behavior**
5. **Actual behavior**
6. **Error messages/stack traces**
7. **Minimal reproducible example**

Example:
```markdown
## Bug: BlurFilter crashes on grayscale images

### Environment
- Python 3.10.12
- Ubuntu 22.04
- Decimatr 0.1.0

### Steps to Reproduce
1. Load a grayscale video
2. Create BlurFilter with threshold=100
3. Process frames

### Expected Behavior
Filter should process grayscale frames

### Actual Behavior
Raises ValueError: "expected 3 channels"

### Error Message
```
ValueError: expected 3 channels, got 1
  at decimatr/taggers/blur.py:45
```

### Minimal Example
```python
from decimatr import FrameProcessor
processor = FrameProcessor.with_blur_removal(threshold=100)
# Process grayscale video...
```
```

## Feature Requests

When requesting features:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case**: Why is this feature needed?
3. **Propose a solution**: How might it work?
4. **Consider alternatives**: What other approaches exist?

Example:
```markdown
## Feature Request: Add SSIM-based duplicate detection

### Use Case
Current hash-based duplicate detection misses perceptually similar frames
with slight variations (noise, compression artifacts).

### Proposed Solution
Add SSIMDuplicateFilter that uses structural similarity index.

### API
```python
from decimatr.filters.ssim import SSIMDuplicateFilter
filter = SSIMDuplicateFilter(threshold=0.95, buffer_size=30)
```

### Alternatives Considered
- Enhance current hash algorithm (less accurate)
- Use learned embeddings (heavier dependencies)
```

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas

Thank you for contributing to Decimatr!
