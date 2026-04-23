from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import DOCS_ROOT, HIGH_VALUE_QUANTILE, REPORTS_ROOT
from app.utils import write_text
from training.build_demo_dataset import build_calibrated_demo_dataset, save_demo_dataset
from training.common import LOGGER, PROCESSED_DATA_DIR, RAW_DATA_DIR
from training.evaluate_models import build_model_comparison_report
from training.explain_model import run_explainability
from training.feature_engineering import engineer_features
from training.feature_selection import run_feature_selection
from training.preprocess import (
    choose_input_csv,
    detect_dataset_type,
    infer_target_column,
    inspect_dataset,
    load_input_csv,
    perform_eda,
)
from training.train_models import train_and_select_models


def _create_default_row_level_data() -> Path:
    df = build_calibrated_demo_dataset(pd.DataFrame({"metric": [], "value": []}), n_customers=2200)
    path = RAW_DATA_DIR / "customer_clv_row_level.csv"
    df.to_csv(path, index=False)
    LOGGER.info("Generated default row-level demo data at %s", path)
    return path


def _write_assumptions(notes: list[str], dataset_type: str, source_path: str) -> None:
    lines = [
        "# Assumptions",
        "",
        f"- Data source used: `{source_path}`",
        f"- Dataset interpretation: **{dataset_type}**",
        "- Pipeline uses adaptive feature engineering and gracefully skips unavailable derived features.",
        "- XGBoost and SHAP are optional dependencies; fallback models and feature importance are used when unavailable.",
    ]

    if notes:
        lines.append("")
        lines.append("## Run Notes")
        lines.extend([f"- {note}" for note in notes])

    write_text(DOCS_ROOT / "assumptions.md", "\n".join(lines))


def _write_executive_summary(
    dataset_meta: Dict[str, Any],
    target_col: str,
    best_reg: str,
    best_cls: str,
    threshold_value: float,
) -> None:
    lines = [
        "# Executive Summary",
        "",
        "## Business Objective",
        "- Predict customer lifetime value (CLV) for revenue forecasting and budget optimization.",
        "- Identify high-value customers for retention and premium prioritization.",
        "",
        "## What Was Done",
        "- Ingested and profiled the input data with automatic dataset-grain detection.",
        "- Executed EDA to surface value concentration, quality issues, and early business signals.",
        "- Built adaptive RFM-centric feature engineering with graceful fallbacks.",
        "- Applied multi-method feature selection and multi-model benchmarking.",
        "- Selected final models objectively using held-out performance metrics.",
        "",
        "## Selected Models",
        f"- Regression winner: **{best_reg}**",
        f"- Classification winner: **{best_cls}**",
        f"- High-value threshold based on CLV quantile: **{dataset_meta.get('high_value_quantile', 0.8)}** (CLV cutoff: **{threshold_value:.2f}**)",
        "",
        "## Business Value",
        "- Focus retention budget on high CLV customers with elevated churn risk.",
        "- Prioritize premium servicing and upsell for high CLV customers with stable engagement.",
        "- Scale low-cost automated journeys for low CLV segments.",
        "",
        "## Artifacts",
        f"- Target modeled: `{target_col}`",
        f"- CLV target definition: `{dataset_meta.get('target_definition', {}).get('formula', 'n/a')}`",
        "- Model metrics: `reports/metrics/model_metrics.json`",
        "- Business action playbook: `reports/business_recommendations.md`",
    ]
    write_text(REPORTS_ROOT / "executive_summary.md", "\n".join(lines))


def run_pipeline(input_csv: str | None, high_value_quantile: float = HIGH_VALUE_QUANTILE) -> Dict[str, Any]:
    notes: list[str] = []

    try:
        source_path = choose_input_csv(input_csv)
    except FileNotFoundError:
        source_path = _create_default_row_level_data()
        notes.append("No raw input file was provided; generated baseline row-level demo dataset.")

    input_df = load_input_csv(source_path)
    profile = inspect_dataset(input_df, str(source_path))
    detected_type = detect_dataset_type(input_df)

    if detected_type == "summary_level":
        calibrated = build_calibrated_demo_dataset(input_df, n_customers=2800)
        calibrated_path = save_demo_dataset(calibrated)
        working_df = calibrated
        dataset_type = "summary_level_calibrated_demo"
        notes.append(
            "Input appeared to be summary-level; generated calibrated synthetic row-level data for full showcase modeling."
        )
        data_source = str(calibrated_path)
    else:
        working_df = input_df
        dataset_type = "row_level"
        data_source = str(source_path)

    target_col = infer_target_column(working_df)
    perform_eda(working_df, target_col)

    fe_result = engineer_features(
        working_df,
        target_col,
        high_value_quantile=high_value_quantile,
    )
    notes.extend(fe_result.messages)

    fs_result = run_feature_selection(fe_result.dataframe, fe_result.target_column)
    notes.extend(fs_result.notes)

    classification_target_col: str | None = None
    for candidate in ["high_value_flag", "premium_flag", "is_high_value"]:
        if candidate in fe_result.dataframe.columns and fe_result.dataframe[candidate].nunique(dropna=True) >= 2:
            classification_target_col = candidate
            notes.append(
                f"Using `{candidate}` as classification target for high-value customer modeling."
            )
            break

    split_df = fe_result.dataframe.copy()
    stratify_col = None
    if "high_value_flag" in split_df.columns and split_df["high_value_flag"].nunique(dropna=True) >= 2:
        stratify_col = split_df["high_value_flag"]
    train_df, test_df = train_test_split(
        split_df, test_size=0.2, random_state=42, stratify=stratify_col
    )
    train_path = PROCESSED_DATA_DIR / "training_dataset.csv"
    test_path = PROCESSED_DATA_DIR / "testing_dataset.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    notes.append(
        f"Created train/test datasets: {train_path} (rows={len(train_df)}), {test_path} (rows={len(test_df)})."
    )

    dataset_meta = {
        "dataset_type": dataset_type,
        "data_source": data_source,
        "notes": notes,
        "high_value_quantile": high_value_quantile,
        "profile": profile,
        "classification_target_column": classification_target_col,
        "target_definition": fe_result.target_definition,
    }

    train_result = train_and_select_models(
        fe_result.dataframe,
        fe_result.target_column,
        fs_result.shortlisted_features,
        high_value_quantile=high_value_quantile,
        dataset_meta=dataset_meta,
        train_df=train_df,
        test_df=test_df,
    )

    build_model_comparison_report()
    explain_summary = run_explainability(
        fe_result.dataframe, fe_result.target_column, fs_result.shortlisted_features
    )

    _write_assumptions(notes, dataset_type=dataset_type, source_path=data_source)
    _write_executive_summary(
        {**dataset_meta, "high_value_quantile": high_value_quantile},
        target_col=fe_result.target_column,
        best_reg=train_result.best_regression_model,
        best_cls=train_result.best_classification_model,
        threshold_value=train_result.high_value_threshold_value,
    )

    LOGGER.info("Pipeline completed successfully")
    return {
        "dataset_type": dataset_type,
        "data_source": data_source,
        "target_column": fe_result.target_column,
        "target_definition": fe_result.target_definition,
        "train_dataset_path": str(train_path),
        "test_dataset_path": str(test_path),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_regression_model": train_result.best_regression_model,
        "best_classification_model": train_result.best_classification_model,
        "mlflow_enabled": train_result.mlflow_enabled,
        "mlflow_run_id": train_result.mlflow_run_id,
        "mlflow_regressor_uri": train_result.mlflow_regressor_uri,
        "mlflow_classifier_uri": train_result.mlflow_classifier_uri,
        "explainability": explain_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLV showcase end-to-end pipeline")
    parser.add_argument("--input-csv", type=str, default=None, help="Optional input CSV path")
    parser.add_argument(
        "--high-value-quantile",
        type=float,
        default=HIGH_VALUE_QUANTILE,
        help="Quantile threshold for classifying high-value customers",
    )

    args = parser.parse_args()
    run_pipeline(args.input_csv, args.high_value_quantile)


if __name__ == "__main__":
    main()
