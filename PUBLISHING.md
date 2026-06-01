# Publishing ua-stress-engine to PyPI

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org/
2. **API Token**: Generate at https://pypi.org/manage/account/token/
3. **Rust Toolchain**: Install from https://rustup.rs/
4. **Maturin**: `pip install maturin`

## Configure PyPI Credentials

Create or update `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...your-token-here
```

Or use environment variable:
```powershell
$env:MATURIN_PYPI_TOKEN = "pypi-AgEIcHlwaS5vcmcC..."
```

## Publishing Steps

### Option 1: Local Build + Publish (Windows only)

```powershell
# Build and publish (creates wheel for current platform only)
maturin publish --username __token__
```

### Option 2: Cross-Platform Build (Recommended)

For multi-platform wheels (Linux, macOS, Windows), use GitHub Actions or `maturin build --release` with docker:

```powershell
# Build locally for testing
maturin build --release

# The wheel will be in target/wheels/
# Test installation:
pip install target/wheels/ua_stress_engine-1.0.1-*.whl
```

### Option 3: GitHub Actions (Best Practice)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
      
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
          path: dist

  publish:
    needs: [build-wheels]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          path: dist
          merge-multiple: true
      
      - name: Publish to PyPI
        uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --skip-existing dist/*
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
```

## Pre-publish Checklist

- [ ] Version updated in `pyproject.toml` and `crates/python/Cargo.toml`
- [ ] README.md is up to date
- [ ] LICENSE file exists
- [ ] Data file exists: `data/processed/ua_stress.bin.bz2`
- [ ] All tests pass: `pytest tests/`
- [ ] Built and tested wheel locally
- [ ] Committed all changes to git
- [ ] Created git tag: `git tag v1.0.1 && git push --tags`

## Testing Installation

After publishing, test installation:

```bash
# Create fresh venv
python -m venv test_env
.\test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Linux/Mac

# Install from PyPI
pip install ua-stress-engine

# Test basic functionality
python -c "import ukrainian_stress; print(ukrainian_stress.mark('мама'))"
# Should print: ма́ма
```

## Versioning

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Update version in both files:
- `pyproject.toml`
- `crates/python/Cargo.toml`

## Troubleshooting

### Error: "no matching distribution found"
- Check if wheel was built for your platform
- Try installing with `--no-binary`: `pip install --no-binary ua-stress-engine ua-stress-engine`

### Build fails on Windows
- Ensure Rust toolchain is installed: `rustc --version`
- Install Visual Studio Build Tools

### Data file missing
- Run data preparation scripts before building
- Check that `data/processed/ua_stress.bin.bz2` exists and is included in build
