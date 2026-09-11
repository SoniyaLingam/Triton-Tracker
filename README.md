# Tracker

AI/ML Internship Project - Day 1: Environment Architecture & Repository Foundation

## Project Overview

This is a foundational AI/ML project repository focused on proper environment setup, dependency management, and project structure. Day 1 establishes reproducible development practices without implementing application logic or ML models.

## Project Structure

```
tracker/
├── data/                  # Datasets (raw, processed, external)
│   └── .gitkeep
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   └── .gitkeep
├── configs/               # Configuration files (hyperparameters, settings)
│   └── .gitkeep
├── scripts/               # Utility scripts (preprocessing, analysis, etc.)
│   └── .gitkeep
├── src/                   # Reusable source code modules
│   └── .gitkeep
├── tests/                 # Unit and integration tests
│   └── .gitkeep
├── .gitignore             # Git ignore rules (Python, ML artifacts, IDE files)
├── README.md              # This file
└── pyproject.toml         # Project metadata and dependency management
```

**Directory Purposes:**
- **`data/`** — Store all data files (CSV, Parquet, raw/processed splits). Excluded from Git.
- **`notebooks/`** — Jupyter notebooks for EDA, prototyping, and exploration. Tracked by Git (notebook outputs should be cleared before commits).
- **`configs/`** — YAML/JSON configuration files for model hyperparameters and settings.
- **`scripts/`** — Standalone Python scripts for data preprocessing, training, evaluation.
- **`src/`** — Reusable Python modules and packages (importable as `from src.module import ...`).
- **`tests/`** — Unit tests for `src/` modules using pytest.

## Python Version Requirement

**Minimum Python 3.9** — Recommended Python 3.10 or 3.11 for best compatibility

Check your Python version:
```bash
python --version
```

## Dependencies

### No External Dependencies for Day 1
This foundation intentionally includes **zero** ML/data science packages. On Day 2+, we'll add:
- `numpy` — Numerical computing
- `pandas` — Data manipulation
- `scikit-learn` — ML algorithms
- `torch` or `tensorflow` — Deep learning (if needed)

**Development Dependencies** (in `pyproject.toml`):
- `pytest` — Unit testing framework
- `pytest-cov` — Code coverage
- `black` — Code formatting
- `flake8` — Linting
- `isort` — Import sorting

## Environment Setup

### Step 1: Create Virtual Environment

**Using venv (built-in, recommended):**
```bash
# Navigate to project folder
cd tracker

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Using conda (if preferred):**
```bash
conda create --name tracker python=3.11
conda activate tracker
```

### Step 2: Upgrade pip and Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install project in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Step 3: Verify Installation

```bash
# Check Python version
python --version

# Check pip list
pip list

# Verify pytest is installed
pytest --version
```

Expected output: All commands succeed, no errors. Pytest version should be 7.4.3 or similar.

**Note:** The `tests/` directory is currently empty (Day 1 foundation only). Tests will be added in Day 2+.

## Project Structure Explanation

### Why This Layout?

**`src/` Layout vs Flat Layout:**
- ✅ `src/` layout isolates source code, preventing import confusion
- ✅ Ensures tests import from installed package (catches dependency issues)
- ✅ Professional standard in ML/production projects
- ✅ Easier to package and distribute

**Dependency Management:**
- `pyproject.toml` replaces `setup.py` and `requirements.txt` (modern standard)
- Pinned versions (`==`) ensure reproducibility across environments
- Optional groups (`[dev]`, `[ml]`) allow flexible installations

**Virtual Environment:**
- Isolates project dependencies from system Python
- Prevents version conflicts between projects
- Essential for reproducibility

### Why No Data/Models in Git?

- `data/` and `models/` excluded via `.gitignore`
- Large files slow down Git; use DVC (Data Version Control) for production
- Prevents accidental commits of sensitive datasets
- Keeps repository lightweight

## Common Workflows

### Create and Activate Environment

```bash
# First time setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### Run Tests

Tests will be added starting in Day 2. Once tests are created in `tests/`, use:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_module.py -v
```

### Format and Lint Code

```bash
# Auto-format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Check for linting issues
flake8 src/ tests/
```

### Add New Dependencies

**For production use:**
```bash
# Install package
pip install package-name

# Pin version in pyproject.toml [project] dependencies
# Edit pyproject.toml, then reinstall
pip install -e .
```

**For development only:**
```bash
# Add to [project.optional-dependencies] dev section
# Then reinstall
pip install -e ".[dev]"
```

## Troubleshooting

### "venv not found" or "python command not found"
- Ensure Python is installed and in PATH
- On Windows, use full path: `C:\Python311\python.exe -m venv venv`

### "No module named 'pytest'" after activation
- Make sure venv is activated
- Reinstall: `pip install -e ".[dev]"`

### Import errors in IDE
- Ensure IDE uses the venv interpreter
  - VS Code: Select Python interpreter → choose `.venv/bin/python`
  - PyCharm: Settings → Project → Python Interpreter → Add existing venv

## Next Steps (Day 2+)

- [ ] Add data loading utilities in `src/`
- [ ] Create configuration system for hyperparameters
- [ ] Add ML dependencies (`numpy`, `pandas`, `scikit-learn`)
- [ ] Implement data preprocessing scripts
- [ ] Write unit tests for utilities
- [ ] Create Jupyter notebooks for exploration

## References

- **pyproject.toml Standard:** [PEP 517](https://peps.python.org/pep-0517/), [PEP 518](https://peps.python.org/pep-0518/), [PEP 621](https://peps.python.org/pep-0621/)
- **Virtual Environments:** [Python venv Documentation](https://docs.python.org/3/library/venv.html)
- **Pytest:** [Pytest Documentation](https://docs.pytest.org/)
- **Python Packaging:** [Packaging Python Projects](https://packaging.python.org/)

## License

MIT License — See LICENSE file for details

---

**Last Updated:** 2026-09-11 | **Day:** 1 | **Status:** Foundation Complete ✓
