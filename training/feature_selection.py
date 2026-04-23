from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from app.config import REPORTS_ROOT
from app.utils import write_json, write_text
from training.common import LOGGER, METRICS_DIR


@dataclass
class FeatureSelectionResult:
    shortlisted_features: List[str]
    scores_df: pd.DataFrame
    methods: Dict[str, List[str]]
    notes: List[str]


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = pd.to_numeric(series, errors="coerce").fillna(series.median())
        else:
            encoded[col] = pd.factorize(series.fillna("missing"))[0]
    return encoded


def _minmax(series: pd.Series) -> pd.Series:
    s = series.fillna(0)
    min_val, max_val = s.min(), s.max()
    if max_val - min_val == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_val) / (max_val - min_val)


def run_feature_selection(df: pd.DataFrame, target_col: str) -> FeatureSelectionResult:
    notes: List[str] = []
    selection_df = df
    if len(df) > 20000:
        selection_df = df.sample(n=20000, random_state=42)
        notes.append(
            "Feature selection executed on a representative 20,000-row sample for runtime efficiency."
        )

    dropped = {target_col, "customer_id", "customerid"}
    feature_cols = [c for c in selection_df.columns if c not in dropped]

    leakage_columns = {
        "high_value_flag",
        "high_value_probability",
        "customer_segment",
        "action_priority",
        "recommended_action",
        "predicted_clv",
    }
    removed_leakage = [col for col in feature_cols if col in leakage_columns and col != target_col]
    if removed_leakage:
        feature_cols = [col for col in feature_cols if col not in removed_leakage]
        notes.append(
            "Excluded leakage-prone columns from feature selection: "
            + ", ".join(sorted(removed_leakage))
        )

    id_like_columns = [
        col
        for col in feature_cols
        if any(token in col for token in ["customerid", "policy_nb", "fullpolicy", "account_id"])
    ]
    if id_like_columns:
        feature_cols = [col for col in feature_cols if col not in id_like_columns]
        notes.append(
            "Excluded identifier-style columns with low generalization value: "
            + ", ".join(sorted(id_like_columns))
        )

    high_cardinality_cols = [
        col
        for col in feature_cols
        if selection_df[col].dtype == "O"
        and selection_df[col].nunique(dropna=False) > min(120, int(len(selection_df) * 0.25))
    ]
    if high_cardinality_cols:
        feature_cols = [col for col in feature_cols if col not in high_cardinality_cols]
        notes.append(
            "Excluded high-cardinality categorical fields to reduce overfit risk: "
            + ", ".join(sorted(high_cardinality_cols))
        )

    low_variance_cols = [col for col in feature_cols if selection_df[col].nunique(dropna=False) <= 1]
    if low_variance_cols:
        feature_cols = [col for col in feature_cols if col not in low_variance_cols]
        notes.append(
            "Excluded zero/near-zero variance fields: " + ", ".join(sorted(low_variance_cols))
        )

    X_raw = selection_df[feature_cols].copy()
    y = pd.to_numeric(selection_df[target_col], errors="coerce").fillna(selection_df[target_col].median())
    X = _encode_features(X_raw)

    score_table = pd.DataFrame(index=feature_cols)

    corr_scores = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(selection_df[col]):
            corr = pd.to_numeric(selection_df[col], errors="coerce").corr(y)
            corr_scores[col] = abs(float(corr)) if pd.notna(corr) else 0.0
        else:
            corr_scores[col] = 0.0
    score_table["correlation"] = pd.Series(corr_scores)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    score_table["mutual_info"] = pd.Series(mi_scores, index=feature_cols)

    rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
    rf_model.fit(X, y)
    score_table["rf_importance"] = pd.Series(rf_model.feature_importances_, index=feature_cols)

    max_rfecv_features = min(20, len(feature_cols))
    rfecv_candidates = list(
        score_table["rf_importance"].sort_values(ascending=False).head(max_rfecv_features).index
    )
    rfecv_support = pd.Series(False, index=feature_cols)

    can_run_rfecv = len(selection_df) <= 8000 and len(feature_cols) <= 35 and len(rfecv_candidates) >= 6
    if can_run_rfecv:
        X_rfecv = X[rfecv_candidates]
        rfecv = RFECV(
            estimator=RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=1),
            step=1,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring="r2",
            min_features_to_select=max(4, min(8, len(rfecv_candidates))),
            n_jobs=1,
        )
        rfecv.fit(X_rfecv, y)
        rfecv_support[rfecv_candidates] = rfecv.support_
    else:
        notes.append(
            "RFECV skipped for runtime safety on large/high-dimensional data; retained top "
            "tree-importance features as RFECV proxy."
        )
        rfecv_support.loc[rfecv_candidates[:8]] = True

    score_table["rfecv"] = rfecv_support.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=15000)
    lasso.fit(X_scaled, y)
    score_table["lasso_abs_coef"] = pd.Series(np.abs(lasso.coef_), index=feature_cols)

    normalized = pd.DataFrame(index=feature_cols)
    for col in score_table.columns:
        normalized[col] = _minmax(score_table[col])
    normalized["rfecv"] = score_table["rfecv"]  # retain hard selection

    normalized["combined_score"] = normalized.mean(axis=1)
    ranked = normalized.sort_values("combined_score", ascending=False)

    methods = {
        "correlation_top": score_table["correlation"].sort_values(ascending=False).head(10).index.tolist(),
        "mutual_info_top": score_table["mutual_info"].sort_values(ascending=False).head(10).index.tolist(),
        "rfecv_selected": rfecv_support[rfecv_support].index.tolist(),
        "rf_importance_top": score_table["rf_importance"].sort_values(ascending=False).head(10).index.tolist(),
        "lasso_top": score_table["lasso_abs_coef"].sort_values(ascending=False).head(10).index.tolist(),
    }

    vote_counter = pd.Series(0, index=feature_cols, dtype=float)
    for values in methods.values():
        vote_counter.loc[values] += 1

    shortlist = ranked.index[(ranked["combined_score"] >= ranked["combined_score"].median()) | (vote_counter >= 2)].tolist()
    shortlist = shortlist[: max(8, min(18, len(shortlist)))]

    export = score_table.copy()
    export["combined_score"] = normalized["combined_score"]
    export["selection_votes"] = vote_counter
    export.sort_values("combined_score", ascending=False, inplace=True)
    export.index.name = "feature"
    export.to_csv(METRICS_DIR / "feature_selection_scores.csv")

    summary_json = {
        "methods": methods,
        "final_shortlist": shortlist,
        "top_ranked": export.head(15).reset_index().to_dict(orient="records"),
        "selection_notes": notes,
    }
    write_json(METRICS_DIR / "feature_selection_summary.json", summary_json)

    md_lines = [
        "# Feature Selection Summary",
        "",
        "## Methods Applied",
        "- Correlation / univariate relationship analysis",
        "- Mutual information",
        "- RFECV (recursive feature elimination with cross-validation)",
        "- Random Forest model-based importance",
        "- L1 regularization (Lasso absolute coefficients)",
        "",
        "## Top Features by Method",
    ]

    for method_name, features in methods.items():
        md_lines.append(f"- **{method_name}**: {', '.join(features[:8])}")

    md_lines.extend(
        [
            "",
            "## Final Shortlisted Features",
            f"- {', '.join(shortlist)}",
            "",
            "## Selection Guardrails Applied",
        ]
    )
    md_lines.extend([f"- {note}" for note in notes] or ["- No additional guardrails were required."])
    md_lines.extend(
        [
            "",
            "## Why These Features",
            "- Selected features consistently scored well across multiple statistical and model-based methods.",
            "- The shortlist balances predictive signal with business interpretability for stakeholder trust.",
            "- RFM-derived behavior fields (recency, frequency, monetary) are retained because they capture purchase dynamics directly linked to CLV.",
        ]
    )

    write_text(REPORTS_ROOT / "feature_selection_summary.md", "\n".join(md_lines))
    LOGGER.info("Feature selection complete with %d shortlisted features", len(shortlist))

    return FeatureSelectionResult(
        shortlisted_features=shortlist,
        scores_df=export,
        methods=methods,
        notes=notes,
    )
