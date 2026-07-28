from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Log directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Documentation directory
DOCS_DIR = PROJECT_ROOT / "docs"

# Notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# Test directory
TESTS_DIR = PROJECT_ROOT / "tests"

# Deployment directory
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

# Scripts directory
SCRIPTS_DIR = PROJECT_ROOT / "scripts"