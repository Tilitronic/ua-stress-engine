# Publishing Luscinia to PyPI

## Quick Publish

```powershell
# Navigate to package directory
cd packages\luscinia

# Build distribution
python -m build

# Test locally
pip install dist/luscinia-1.0.0-py3-none-any.whl

# Test basic functionality
python -c "from luscinia import LusciniaPredictor; p = LusciniaPredictor(); print(p.predict('мама'))"

# Publish to TestPyPI (for testing)
python -m twine upload --repository testpypi dist/*

# Publish to PyPI (production)
python -m twine upload dist/*
```

## Prerequisites

```powershell
pip install build twine
```

## PyPI Credentials

Set environment variable:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-your-token-here"
```

Or configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
username = __token__
password = pypi-your-testpypi-token
```

## Build Package

```powershell
# Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build wheel and source distribution
python -m build
```

This creates:

- `dist/luscinia-1.0.0-py3-none-any.whl` (~30 MB)
- `dist/luscinia-1.0.0.tar.gz`

## Test Installation

```powershell
# Create test environment
python -m venv test_env
.\test_env\Scripts\activate

# Install from wheel
pip install dist/luscinia-1.0.0-py3-none-any.whl

# Test
python -c "from luscinia import LusciniaPredictor; p = LusciniaPredictor(); print(p.predict('університет'))"
# Should print: 4

# Run tests
pip install pytest
pytest tests/
```

## Publish

### TestPyPI (Recommended First)

```powershell
python -m twine upload --repository testpypi dist/*
```

Test installation:

```powershell
pip install -i https://test.pypi.org/simple/ luscinia
```

### Production PyPI

```powershell
python -m twine upload dist/*
```

Verify:

```powershell
pip install luscinia
python -c "from luscinia import LusciniaPredictor; print(LusciniaPredictor())"
```

## Versioning

Update version in `pyproject.toml`:

```toml
version = "1.0.1"
```

And in `luscinia/__init__.py`:

```python
__version__ = "1.0.1"
```

## GitHub Actions (Automated)

The package includes GitHub Actions workflow for automatic publishing on release.
See `.github/workflows/publish-luscinia.yml`

## Troubleshooting

### Error: "File too large"

The ONNX model is ~30 MB compressed, which is acceptable for PyPI. If you encounter size issues, ensure:

- Model file is `.onnx.gz` (compressed)
- No unnecessary files in dist/

### Error: "Invalid distribution"

Ensure pyproject.toml is valid:

```powershell
python -m build --check
```

### Test failures

Run tests before publishing:

```powershell
pytest tests/ -v
```
