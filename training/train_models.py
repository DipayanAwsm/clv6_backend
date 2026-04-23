from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    ENABLE_MLFLOW,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_ROOT,
)
from app.utils import write_json
from training.common import LOGGER, METRICS_DIR, PROCESSED_DATA_DIR, SAMPLE_INPUT_DIR

try:
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]


ENABLE_XGBOOST = os.getenv("ENABLE_XGBOOST", "false").lower() in {"1", "true", "yes"}


@dataclass
class TrainingResult:
    best_regression_model: str
    best_classification_model: str
    regression_metrics: pd.DataFrame
    classification_metrics: pd.DataFrame
    high_value_threshold_value: float
    train_rows: int
    test_rows: int
    mlflow_enabled: bool
    mlflow_run_id: str | None
    mlflow_regressor_uri: str | None
    mlflow_classifier_uri: str | None


def _safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float | None:
    y_true_np = np.array(y_true, dtype=float)
    y_pred_np = np.array(y_pred, dtype=float)
    non_zero = np.abs(y_true_np) > 1e-8
    if non_zero.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true_np[non_zero] - y_pred_np[non_zero]) / y_true_np[non_zero])) * 100)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def _mlflow_log_metrics(metrics: Dict[str, Any]) -> None:
    if mlflow is None:
        return
    for key, value in metrics.items():
        safe_value = _to_float(value)
        if safe_value is not None:
            mlflow.log_metric(key, safe_value)


def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
    else:
        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def _regression_candidates() -> List[Tuple[str, Any, bool]]:
    candidates: List[Tuple[str, Any, bool]] = [
        ("LinearRegression", LinearRegression(), True),
        ("Ridge", Ridge(alpha=1.0, random_state=42), True),
        ("Lasso", Lasso(alpha=0.001, random_state=42, max_iter=12000), True),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=1), False),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42), False),
    ]

    if not ENABLE_XGBOOST:
        LOGGER.info("ENABLE_XGBOOST is false; skipping XGBoost regressor")
        return candidates

    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "XGBoostRegressor",
                XGBRegressor(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.2,
                    random_state=42,
                    n_jobs=1,
                ),
                False,
            )
        )
    except Exception:
        LOGGER.info("xgboost not available, skipping XGBoost regressor")

    return candidates


def _classification_candidates() -> List[Tuple[str, Any, bool]]:
    candidates: List[Tuple[str, Any, bool]] = [
        ("LogisticRegression", LogisticRegression(max_iter=1500), True),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=1), False),
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42), False),
    ]

    if not ENABLE_XGBOOST:
        LOGGER.info("ENABLE_XGBOOST is false; skipping XGBoost classifier")
        return candidates

    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "XGBoostClassifier",
                XGBClassifier(
                    n_estimators=320,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                ),
                False,
            )
        )
    except Exception:
        LOGGER.info("xgboost not available, skipping XGBoost classifier")

    return candidates


def train_and_select_models(
    df: pd.DataFrame,
    target_col: str,
    selected_features: List[str],
    high_value_quantile: float,
    dataset_meta: Dict[str, Any],
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> TrainingResult:
    usable_features = [feature for feature in selected_features if feature in df.columns and feature != target_col]
    if not usable_features:
        raise ValueError("No usable features available after selection.")

    classification_target_col = dataset_meta.get("classification_target_column")

    if train_df is not None and test_df is not None:
        X_train = train_df[usable_features].copy()
        X_test = test_df[usable_features].copy()
        y_train_reg = pd.to_numeric(train_df[target_col], errors="coerce").fillna(
            train_df[target_col].median()
        )
        y_test_reg = pd.to_numeric(test_df[target_col], errors="coerce").fillna(
            test_df[target_col].median()
        )
    else:
        X = df[usable_features].copy()
        y_reg = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

    high_value_threshold = float(y_train_reg.quantile(high_value_quantile))
    if (
        classification_target_col
        and classification_target_col in (train_df.columns if train_df is not None else df.columns)
        and classification_target_col in (test_df.columns if test_df is not None else df.columns)
    ):
        source_train = train_df if train_df is not None else df.loc[X_train.index]
        source_test = test_df if test_df is not None else df.loc[X_test.index]
        y_train_cls = pd.to_numeric(source_train[classification_target_col], errors="coerce").fillna(0).astype(int)
        y_test_cls = pd.to_numeric(source_test[classification_target_col], errors="coerce").fillna(0).astype(int)
        if y_train_cls.nunique() < 2:
            y_train_cls = (y_train_reg >= high_value_threshold).astype(int)
            y_test_cls = (y_test_reg >= high_value_threshold).astype(int)
    else:
        y_train_cls = (y_train_reg >= high_value_threshold).astype(int)
        y_test_cls = (y_test_reg >= high_value_threshold).astype(int)

    if y_train_cls.nunique() < 2:
        median_threshold = float(y_train_reg.median())
        y_train_cls = (y_train_reg > median_threshold).astype(int)
        y_test_cls = (y_test_reg > median_threshold).astype(int)

    if y_train_cls.nunique() < 2:
        surrogate_candidates = [
            col
            for col in [
                "monetary",
                "total_spend",
                "directwrittenpremium_am",
                "earnedpremium_am",
                "coverageamount",
                "householdincome",
                "creditscore",
                "frequency",
                "customertenure",
            ]
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col])
        ]

        if surrogate_candidates:
            train_signal = X_train[surrogate_candidates].rank(pct=True).mean(axis=1)
            test_signal = X_test[surrogate_candidates].rank(pct=True).mean(axis=1)
            cutoff = float(train_signal.quantile(high_value_quantile))
            y_train_cls = (train_signal >= cutoff).astype(int)
            y_test_cls = (test_signal >= cutoff).astype(int)
            classification_target_col = "surrogate_high_value_signal"
            dataset_meta.setdefault("notes", []).append(
                "Classification target lacked variance; created surrogate high-value classes "
                f"from: {', '.join(surrogate_candidates)}."
            )

    if y_train_cls.nunique() < 2:
        raise ValueError(
            "Unable to build classification target with at least two classes even after fallback."
        )

    mlflow_enabled = ENABLE_MLFLOW and mlflow is not None
    if ENABLE_MLFLOW and mlflow is None:
        LOGGER.warning("MLflow integration requested but mlflow package is unavailable.")

    mlflow_run_id: str | None = None
    mlflow_regressor_uri: str | None = None
    mlflow_classifier_uri: str | None = None
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        except Exception as exc:
            LOGGER.warning("Failed to initialize MLflow tracking; proceeding without MLflow. %s", exc)
            mlflow_enabled = False

    reg_metrics: List[Dict[str, Any]] = []
    cls_metrics: List[Dict[str, Any]] = []
    trained_reg_models: Dict[str, Pipeline] = {}
    trained_cls_models: Dict[str, Pipeline] = {}
    parent_run_ctx = (
        mlflow.start_run(run_name=f"clv_training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        if mlflow_enabled
        else nullcontext()
    )

    with parent_run_ctx as active_run:
        if mlflow_enabled and active_run is not None:
            mlflow_run_id = active_run.info.run_id
            mlflow.log_params(
                {
                    "dataset_type": str(dataset_meta.get("dataset_type")),
                    "target_column": target_col,
                    "classification_target_column": classification_target_col
                    or "derived_from_clv_quantile",
                    "high_value_quantile": high_value_quantile,
                    "selected_feature_count": len(usable_features),
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                }
            )
            if usable_features:
                mlflow.log_text(
                    "\n".join(usable_features), "feature_artifacts/selected_features.txt"
                )
            notes = dataset_meta.get("notes", [])
            if notes:
                mlflow.log_text("\n".join(str(note) for note in notes), "feature_artifacts/run_notes.txt")

        for name, estimator, scale_numeric in _regression_candidates():
            candidate_ctx = (
                mlflow.start_run(run_name=f"regression_{name}", nested=True)
                if mlflow_enabled
                else nullcontext()
            )
            with candidate_ctx:
                preprocessor = _build_preprocessor(X_train, scale_numeric=scale_numeric)
                pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                pipeline.fit(X_train, y_train_reg)
                pred = pipeline.predict(X_test)

                rmse = float(np.sqrt(mean_squared_error(y_test_reg, pred)))
                mape = _safe_mape(y_test_reg, pred)
                metric_row = {
                    "model": name,
                    "r2": float(r2_score(y_test_reg, pred)),
                    "mae": float(mean_absolute_error(y_test_reg, pred)),
                    "rmse": rmse,
                    "mape": mape,
                }
                reg_metrics.append(metric_row)
                trained_reg_models[name] = pipeline

                if mlflow_enabled:
                    try:
                        mlflow.log_params({"track": "regression", "algorithm": name})
                        _mlflow_log_metrics(
                            {
                                "r2": metric_row["r2"],
                                "mae": metric_row["mae"],
                                "rmse": metric_row["rmse"],
                                "mape": metric_row["mape"],
                            }
                        )
                        mlflow.sklearn.log_model(
                            pipeline, artifact_path=f"candidates/regression/{name}"
                        )
                    except Exception as exc:
                        LOGGER.warning("MLflow logging failed for regression model %s: %s", name, exc)

        for name, estimator, scale_numeric in _classification_candidates():
            candidate_ctx = (
                mlflow.start_run(run_name=f"classification_{name}", nested=True)
                if mlflow_enabled
                else nullcontext()
            )
            with candidate_ctx:
                preprocessor = _build_preprocessor(X_train, scale_numeric=scale_numeric)
                pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                pipeline.fit(X_train, y_train_cls)

                pred = pipeline.predict(X_test)
                if hasattr(pipeline, "predict_proba"):
                    prob = pipeline.predict_proba(X_test)[:, 1]
                else:
                    prob = pred.astype(float)

                try:
                    roc = float(roc_auc_score(y_test_cls, prob))
                except Exception:
                    roc = float("nan")

                metric_row = {
                    "model": name,
                    "accuracy": float(accuracy_score(y_test_cls, pred)),
                    "precision": float(precision_score(y_test_cls, pred, zero_division=0)),
                    "recall": float(recall_score(y_test_cls, pred, zero_division=0)),
                    "f1": float(f1_score(y_test_cls, pred, zero_division=0)),
                    "roc_auc": roc,
                    "confusion_matrix": confusion_matrix(y_test_cls, pred).tolist(),
                }
                cls_metrics.append(metric_row)
                trained_cls_models[name] = pipeline

                if mlflow_enabled:
                    try:
                        mlflow.log_params({"track": "classification", "algorithm": name})
                        _mlflow_log_metrics(
                            {
                                "accuracy": metric_row["accuracy"],
                                "precision": metric_row["precision"],
                                "recall": metric_row["recall"],
                                "f1": metric_row["f1"],
                                "roc_auc": metric_row["roc_auc"],
                            }
                        )
                        mlflow.sklearn.log_model(
                            pipeline, artifact_path=f"candidates/classification/{name}"
                        )
                    except Exception as exc:
                        LOGGER.warning("MLflow logging failed for classification model %s: %s", name, exc)

    reg_df = pd.DataFrame(reg_metrics)
    reg_df["r2"] = reg_df["r2"].replace([np.inf, -np.inf], np.nan)
    reg_df["r2_sort"] = reg_df["r2"].fillna(-9999)
    reg_df.sort_values(["r2_sort", "rmse"], ascending=[False, True], inplace=True)
    reg_df.drop(columns=["r2_sort"], inplace=True)
    cls_df = pd.DataFrame(cls_metrics).sort_values(["f1", "recall", "roc_auc"], ascending=[False, False, False])

    best_reg_name = str(reg_df.iloc[0]["model"])
    best_cls_name = str(cls_df.iloc[0]["model"])

    best_reg_model = trained_reg_models[best_reg_name]
    best_cls_model = trained_cls_models[best_cls_name]

    joblib.dump(best_reg_model, MODELS_ROOT / "clv_regressor.pkl")
    joblib.dump(best_cls_model, MODELS_ROOT / "high_value_classifier.pkl")
    joblib.dump(best_reg_model.named_steps["preprocessor"], MODELS_ROOT / "preprocessing.pkl")

    reg_df.to_csv(METRICS_DIR / "regression_metrics.csv", index=False)
    cls_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)

    metric_payload = {
        "regression": reg_df.to_dict(orient="records"),
        "classification": cls_df.to_dict(orient="records"),
        "selected_regression_model": best_reg_name,
        "selected_classification_model": best_cls_name,
        "target_definition": dataset_meta.get("target_definition", {}),
        "mlflow": {
            "enabled": bool(mlflow_enabled),
            "tracking_uri": MLFLOW_TRACKING_URI if mlflow_enabled else None,
            "experiment_name": MLFLOW_EXPERIMENT_NAME if mlflow_enabled else None,
            "run_id": mlflow_run_id,
            "regressor_model_uri": mlflow_regressor_uri,
            "classifier_model_uri": mlflow_classifier_uri,
        },
    }
    write_json(METRICS_DIR / "model_metrics.json", metric_payload)

    if mlflow_enabled and mlflow_run_id:
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_params(
                    {
                        "best_regression_model": best_reg_name,
                        "best_classification_model": best_cls_name,
                    }
                )
                best_reg_row = reg_df[reg_df["model"] == best_reg_name].iloc[0].to_dict()
                best_cls_row = cls_df[cls_df["model"] == best_cls_name].iloc[0].to_dict()
                _mlflow_log_metrics(
                    {
                        "best_regression_r2": best_reg_row.get("r2"),
                        "best_regression_rmse": best_reg_row.get("rmse"),
                        "best_regression_mae": best_reg_row.get("mae"),
                        "best_classification_f1": best_cls_row.get("f1"),
                        "best_classification_recall": best_cls_row.get("recall"),
                        "best_classification_precision": best_cls_row.get("precision"),
                        "best_classification_accuracy": best_cls_row.get("accuracy"),
                        "best_classification_roc_auc": best_cls_row.get("roc_auc"),
                    }
                )
                mlflow.sklearn.log_model(best_reg_model, artifact_path="models/clv_regressor")
                mlflow.sklearn.log_model(
                    best_cls_model, artifact_path="models/high_value_classifier"
                )
                mlflow.log_artifact(
                    str(METRICS_DIR / "regression_metrics.csv"), artifact_path="reports"
                )
                mlflow.log_artifact(
                    str(METRICS_DIR / "classification_metrics.csv"), artifact_path="reports"
                )
                mlflow.log_artifact(str(METRICS_DIR / "model_metrics.json"), artifact_path="reports")
                mlflow_regressor_uri = f"runs:/{mlflow_run_id}/models/clv_regressor"
                mlflow_classifier_uri = f"runs:/{mlflow_run_id}/models/high_value_classifier"
        except Exception as exc:
            LOGGER.warning("Failed to log final models to MLflow: %s", exc)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_type": dataset_meta.get("dataset_type"),
        "data_source": dataset_meta.get("data_source"),
        "target_column": target_col,
        "target_definition": dataset_meta.get("target_definition", {}),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "high_value_quantile": high_value_quantile,
        "high_value_threshold_value": high_value_threshold,
        "selected_features": usable_features,
        "training_features": usable_features,
        "regression_model_selected": best_reg_name,
        "classification_model_selected": best_cls_name,
        "notes": dataset_meta.get("notes", []),
        "classification_target_column": classification_target_col or "derived_from_clv_quantile",
        "mlflow": {
            "enabled": bool(mlflow_enabled),
            "tracking_uri": MLFLOW_TRACKING_URI if mlflow_enabled else None,
            "experiment_name": MLFLOW_EXPERIMENT_NAME if mlflow_enabled else None,
            "run_id": mlflow_run_id,
            "regressor_model_uri": mlflow_regressor_uri,
            "classifier_model_uri": mlflow_classifier_uri,
        },
    }
    write_json(MODELS_ROOT / "metadata.json", metadata)

    sample_batch = X_test.head(25).copy()
    sample_batch.to_csv(SAMPLE_INPUT_DIR / "prediction_sample.csv", index=False)

    # Score full engineered dataset for business summaries and dashboard-friendly artifacts.
    X_full = df[usable_features].copy()
    predicted_clv_full = np.array(best_reg_model.predict(X_full), dtype=float)
    if hasattr(best_cls_model, "predict_proba"):
        high_value_prob_full = np.array(best_cls_model.predict_proba(X_full)[:, 1], dtype=float)
    else:
        high_value_prob_full = np.array(best_cls_model.predict(X_full), dtype=float)
    high_value_flag_full = (high_value_prob_full >= 0.5).astype(int)

    scored_df = df.copy()
    scored_df["predicted_clv"] = predicted_clv_full
    scored_df["high_value_probability"] = high_value_prob_full
    scored_df["high_value_flag"] = high_value_flag_full

    high_value_cut = max(float(high_value_threshold), 1.0)
    scored_df["customer_segment"] = np.select(
        [
            scored_df["predicted_clv"] >= high_value_cut * 1.2,
            scored_df["predicted_clv"] >= high_value_cut,
            scored_df["predicted_clv"] >= high_value_cut * 0.7,
        ],
        ["Strategic Premium", "High Value", "Growth Potential"],
        default="Base Portfolio",
    )
    scored_df["action_priority"] = np.select(
        [
            (scored_df["high_value_flag"] == 1) & (scored_df["high_value_probability"] >= 0.75),
            scored_df["high_value_flag"] == 1,
            scored_df["high_value_probability"] >= 0.4,
        ],
        ["critical", "high", "medium"],
        default="baseline",
    )
    scored_df["recommended_action"] = np.select(
        [
            scored_df["action_priority"] == "critical",
            scored_df["action_priority"] == "high",
            scored_df["action_priority"] == "medium",
        ],
        [
            "Activate urgent retention playbook with executive outreach and proactive servicing.",
            "Prioritize loyalty and upsell engagement with premium service treatment.",
            "Run nurture and cross-sell campaigns with monitored budget allocation.",
        ],
        default="Use cost-efficient automation and periodic monitoring.",
    )
    scored_df.to_csv(PROCESSED_DATA_DIR / "scored_customers.csv", index=False)

    policy_state_col = (
        "policyratedstate_tp"
        if "policyratedstate_tp" in scored_df.columns
        else ("region" if "region" in scored_df.columns else None)
    )
    top_state = None
    if policy_state_col:
        try:
            state_scores = (
                scored_df.groupby(policy_state_col)["predicted_clv"].mean().sort_values(ascending=False)
            )
            top_state = None if state_scores.empty else str(state_scores.index[0])
        except Exception:
            top_state = None

    profitable_pct = 0.0
    if "profit" in scored_df.columns:
        profitable_pct = float((pd.to_numeric(scored_df["profit"], errors="coerce").fillna(0) > 0).mean() * 100)
    elif "clv_formula_value" in scored_df.columns:
        profitable_pct = float(
            (pd.to_numeric(scored_df["clv_formula_value"], errors="coerce").fillna(0) > 0).mean() * 100
        )

    if "clv" in scored_df.columns:
        base_clv_series = pd.to_numeric(scored_df["clv"], errors="coerce")
    elif "clv_formula_value" in scored_df.columns:
        base_clv_series = pd.to_numeric(scored_df["clv_formula_value"], errors="coerce")
    else:
        base_clv_series = pd.to_numeric(scored_df["predicted_clv"], errors="coerce")
    average_clv_before_prediction = float(base_clv_series.fillna(0).mean())

    business_summary = {
        "total_customers": int(len(scored_df)),
        "total_predicted_clv": round(float(scored_df["predicted_clv"].sum()), 2),
        "average_predicted_clv": round(float(scored_df["predicted_clv"].mean()), 2),
        "average_clv_before_prediction": round(average_clv_before_prediction, 2),
        "high_value_customers": int(scored_df["high_value_flag"].sum()),
        "high_value_percentage": round(float(scored_df["high_value_flag"].mean() * 100), 2),
        "profitable_percentage": round(profitable_pct, 2),
        "top_state_by_clv": top_state,
    }
    write_json(METRICS_DIR / "business_summary.json", business_summary)

    LOGGER.info(
        "Training complete. Best regression model=%s, best classifier=%s",
        best_reg_name,
        best_cls_name,
    )

    return TrainingResult(
        best_regression_model=best_reg_name,
        best_classification_model=best_cls_name,
        regression_metrics=reg_df,
        classification_metrics=cls_df,
        high_value_threshold_value=high_value_threshold,
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        mlflow_enabled=bool(mlflow_enabled),
        mlflow_run_id=mlflow_run_id,
        mlflow_regressor_uri=mlflow_regressor_uri,
        mlflow_classifier_uri=mlflow_classifier_uri,
    )
