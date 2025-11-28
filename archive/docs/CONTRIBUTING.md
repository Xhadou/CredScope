# Contributing to CredScope

## Getting Started for New Contributors

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credscope.git
cd credscope
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Get the Data

Since data files are not tracked in Git:
1. Download from [Kaggle Home Credit Competition](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Place CSV files in `data/raw/`
3. Verify with: `dir data\raw\` (should see application_train.csv, etc.)

### 4. Verify Setup

```bash
# Test data loading
python -c "from src.credscope.data.loader import DataLoader; print('✅ Setup OK')"
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Commit Changes

```bash
git add .
git commit -m "Add: brief description of changes"
```

Use commit prefixes:
- `Add:` - New feature
- `Fix:` - Bug fix
- `Update:` - Modify existing feature
- `Docs:` - Documentation only
- `Test:` - Add/update tests
- `Refactor:` - Code restructuring

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions/classes
- Write unit tests for new features
- Keep commits focused and atomic

## Project Areas

- `src/credscope/data/` - Data loading and preprocessing
- `src/credscope/features/` - Feature engineering
- `src/credscope/models/` - Model implementations
- `src/credscope/utils/` - Utility functions
- `scripts/` - Training and deployment scripts
- `tests/` - Unit tests

## Questions?

Open an issue on GitHub or reach out to the maintainers!
