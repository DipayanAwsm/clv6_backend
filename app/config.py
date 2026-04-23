from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
project_root_env = os.getenv("CLV_PROJECT_ROOT")
if project_root_env:
    PROJECT_ROOT = Path(project_root_env)
elif (BACKEND_ROOT.parent / "data").exists():
    PROJECT_ROOT = BACKEND_ROOT.parent
else:
    PROJECT_ROOT = BACKEND_ROOT

DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
DOCS_ROOT = PROJECT_ROOT / "docs"
MODELS_ROOT = BACKEND_ROOT / "models"
TRAINING_ROOT = BACKEND_ROOT / "training"

DEFAULT_INPUT_CSV = os.getenv(
    "CLV_INPUT_CSV",
    str(DATA_ROOT / "raw" / "customer_clv_row_level.csv"),
)
HIGH_VALUE_QUANTILE = float(os.getenv("HIGH_VALUE_QUANTILE", "0.8"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "true").lower() in {"1", "true", "yes"}
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"file://{(PROJECT_ROOT / 'mlruns').resolve()}"
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "clv_showcase_experiment")
USE_MLFLOW_MODELS = os.getenv("USE_MLFLOW_MODELS", "false").lower() in {"1", "true", "yes"}

for directory in [DATA_ROOT, REPORTS_ROOT, DOCS_ROOT, MODELS_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)
