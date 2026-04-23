from __future__ import annotations

import pandas as pd

from app.config import REPORTS_ROOT
from app.utils import read_json, write_text
from training.common import METRICS_DIR


def build_model_comparison_report() -> str:
    reg_path = METRICS_DIR / "regression_metrics.csv"
    cls_path = METRICS_DIR / "classification_metrics.csv"
    metadata = read_json(METRICS_DIR / "model_metrics.json", default={})

    if not reg_path.exists() or not cls_path.exists():
        raise FileNotFoundError("Model metrics CSV files are missing.")

    reg_df = pd.read_csv(reg_path)
    cls_df = pd.read_csv(cls_path)

    best_reg = metadata.get("selected_regression_model", reg_df.iloc[0]["model"])
    best_cls = metadata.get("selected_classification_model", cls_df.iloc[0]["model"])

    lines = [
        "# Model Comparison",
        "",
        "## Regression Models (CLV Prediction)",
        reg_df.to_markdown(index=False),
        "",
        "## Classification Models (High-Value Identification)",
        cls_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_markdown(index=False),
        "",
        "## Final Model Selection Rationale",
        f"- Selected regression model: **{best_reg}** based on strongest R2 with competitive MAE/RMSE stability.",
        f"- Selected classification model: **{best_cls}** based on best F1 and recall balance for premium-customer capture.",
        "- Selection prioritized business utility: high-value customer miss rate was treated as costly, so recall and F1 were emphasized.",
    ]

    markdown = "\n".join(lines)
    write_text(REPORTS_ROOT / "model_comparison.md", markdown)
    return markdown


if __name__ == "__main__":
    build_model_comparison_report()
