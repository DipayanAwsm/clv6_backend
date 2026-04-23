from __future__ import annotations

import os
from pathlib import Path

from app.config import DATA_ROOT, MODELS_ROOT, PROJECT_ROOT, REPORTS_ROOT
from app.utils import get_logger

# Ensure matplotlib can write cache in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

LOGGER = get_logger("clv-training")

RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
DEMO_DATA_DIR = DATA_ROOT / "demo"
SAMPLE_INPUT_DIR = DATA_ROOT / "sample_input"
FIGURES_DIR = REPORTS_ROOT / "figures"
METRICS_DIR = REPORTS_ROOT / "metrics"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEMO_DATA_DIR,
    SAMPLE_INPUT_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)
