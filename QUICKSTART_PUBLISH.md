# ✅ ua-stress-engine — Готово до публікації на PyPI

## Статус

- ✅ Пакет зібрано успішно
- ✅ Тести пройдено (ukrainian_stress.mark('мама') → 'ма́ма')
- ✅ Метадані додано (classifiers, keywords, urls, license)
- ✅ Версії синхронізовано (1.0.1)
- ✅ Wheel створено: `target/wheels/ua_stress_engine-1.0.1-cp313-cp313-win_amd64.whl`

## Швидка публікація (Рекомендовано: спочатку TestPyPI)

### 1. Отримати API токен PyPI

1. Зареєструватися на https://pypi.org/ (або https://test.pypi.org/ для тестування)
2. Перейти в Account Settings → API tokens
3. Створити токен для "ua-stress-engine"
4. Зберегти токен (показується один раз!)

### 2. Налаштувати credentials

**Варіант A: Через змінну середовища (рекомендовано)**
```powershell
$env:MATURIN_PYPI_TOKEN = "pypi-AgEIcHlwaS5vcmcC...ваш-токен"
```

**Варіант B: Через ~/.pypirc**
```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...ваш-токен

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...ваш-токен-testpypi
```

### 3. Публікація (TestPyPI для тестування)

```powershell
# Тестова публікація
python -m maturin publish --repository testpypi --username __token__

# Тестування установки з TestPyPI
pip install -i https://test.pypi.org/simple/ ua-stress-engine

# Перевірка
python -c "import ukrainian_stress; print(ukrainian_stress.mark('мама'))"
```

### 4. Публікація (Production PyPI)

```powershell
# Остаточна публікація на PyPI
python -m maturin publish --username __token__

# Перевірка
pip install ua-stress-engine
python -c "import ukrainian_stress; print(ukrainian_stress.word_count())"
```

## Альтернатива: Використати готовий скрипт

```powershell
# Для TestPyPI
.\publish.ps1 test

# Для Production PyPI
.\publish.ps1 prod
```

## Налаштування GitHub Actions (Для кросплатформенних wheels)

Створіть `.github/workflows/publish.yml` для автоматичної публікації при створенні release:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
      
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist

  publish:
    needs: [build-wheels]
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/ua-stress-engine
    permissions:
      id-token: write
    
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

Додайте секрет `PYPI_TOKEN` в Settings → Secrets and variables → Actions.

## Після публікації

1. Створіть git tag:
   ```bash
   git tag v1.0.1
   git push --tags
   ```

2. Створіть GitHub release

3. Оновіть README з інструкціями установки:
   ```bash
   pip install ua-stress-engine
   ```

4. Перевірте сторінку на PyPI: https://pypi.org/project/ua-stress-engine/

## Оновлення версії (для наступних релізів)

1. Оновіть версію в обох файлах:
   - `pyproject.toml` → `version = "1.0.2"`
   - `crates/python/Cargo.toml` → `version = "1.0.2"`

2. Перебудуйте та опублікуйте:
   ```powershell
   python -m maturin build --release
   python -m maturin publish --username __token__
   ```

## Troubleshooting

### Помилка: "File already exists"
Версія вже опублікована. Оновіть версію в pyproject.toml та Cargo.toml.

### Помилка: "Invalid authentication"
Перевірте API токен. Він повинен починатися з `pypi-`.

### Помилка: "403 Forbidden"
Перевірте що у вас є права на публікацію пакету з такою назвою.

### Попередження про external libraries (zlib.dll)
Для Linux/macOS це не проблема. Для Windows можна використати `--auditwheel=repair`, але це потребує додаткових інструментів.

## Документація

- Повна інструкція: `PUBLISHING.md`
- Maturin docs: https://www.maturin.rs/
- PyPI docs: https://packaging.python.org/
