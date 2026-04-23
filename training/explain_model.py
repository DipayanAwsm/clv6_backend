from __future__ import annotations

from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.config import MODELS_ROOT, REPORTS_ROOT
from app.utils import write_text
from training.common import FIGURES_DIR, METRICS_DIR


def _save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _top_feature_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 12,
) -> pd.DataFrame:
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=8,
        random_state=42,
        scoring="r2",
        n_jobs=1,
    )

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance", ascending=False)

    return table.head(top_n)


def _build_business_actions(
    df: pd.DataFrame,
    predicted_clv: np.ndarray,
    hv_probability: np.ndarray,
) -> str:
    frame = df.copy()
    frame["predicted_clv"] = predicted_clv
    frame["high_value_probability"] = hv_probability

    recency = pd.to_numeric(frame.get("recency", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0)
    complaint = pd.to_numeric(frame.get("complaint_rate", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0)
    renewal = pd.to_numeric(frame.get("renewal_ratio", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0)

    recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-9)
    complaint_norm = (complaint - complaint.min()) / (complaint.max() - complaint.min() + 1e-9)
    renewal_norm = (renewal - renewal.min()) / (renewal.max() - renewal.min() + 1e-9)

    frame["churn_risk"] = 0.5 * recency_norm + 0.3 * complaint_norm + 0.2 * (1 - renewal_norm)

    q75 = frame["predicted_clv"].quantile(0.75)
    q40 = frame["predicted_clv"].quantile(0.40)

    conditions = [
        (frame["predicted_clv"] >= q75) & (frame["churn_risk"] >= 0.60),
        (frame["predicted_clv"] >= q75) & (frame["churn_risk"] < 0.60),
        (frame["predicted_clv"].between(q40, q75)) & (frame["churn_risk"] < 0.60),
    ]
    choices = [
        "High CLV + high churn risk",
        "High CLV + low churn risk",
        "Medium CLV + healthy activity",
    ]
    frame["action_segment"] = np.select(conditions, choices, default="Lower CLV / monitor")

    segment_counts = frame["action_segment"].value_counts(normalize=True).mul(100).round(1)

    lines = [
        "# Business Recommendations",
        "",
        "## Action Framework",
        "- **High CLV + high churn risk**: trigger urgent save campaigns (white-glove outreach, service recovery, retention incentive).",
        "- **High CLV + low churn risk**: prioritize loyalty programs, premium service, and upsell/cross-sell offers.",
        "- **Medium CLV + good recent activity**: run nurture journeys to increase share-of-wallet.",
        "- **Low CLV**: move to lower-cost automated engagement tracks with periodic evaluation.",
        "",
        "## Portfolio Mix",
    ]

    for segment, pct in segment_counts.items():
        lines.append(f"- {segment}: **{pct}%** of customers")

    lines.extend(
        [
            "",
            "## Budget Allocation Guidance",
            "- Allocate retention budget first to high-value customers with elevated churn risk.",
            "- Reserve upsell budget for high-value customers with stable engagement and low friction.",
            "- Use automated marketing for lower-value segments to preserve CAC efficiency.",
            "- Review segment migration monthly to update campaign and service priorities.",
        ]
    )

    return "\n".join(lines)


def run_explainability(df: pd.DataFrame, target_col: str, selected_features: List[str]) -> Dict[str, str]:
    model = joblib.load(MODELS_ROOT / "clv_regressor.pkl")
    classifier = joblib.load(MODELS_ROOT / "high_value_classifier.pkl")

    usable_features = [f for f in selected_features if f in df.columns]
    X = df[usable_features].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())
    sample_n = min(350, len(X))
    X_sample = X.sample(sample_n, random_state=42)
    y_sample = y.loc[X_sample.index]

    top_imp = _top_feature_table(model, X_sample, y_sample)
    top_imp.to_csv(METRICS_DIR / "top_feature_impacts.csv", index=False)

    plt.figure(figsize=(9, 6))
    ordered = top_imp.sort_values("importance")
    plt.barh(ordered["feature"], ordered["importance"], color="#0f766e")
    plt.xlabel("Permutation Importance")
    plt.title("Top Feature Impact on CLV Prediction")
    _save_fig(FIGURES_DIR / "top_feature_impact.png")

    shap_status = "fallback"
    try:
        import shap

        shap_sample = X_sample.head(min(220, len(X_sample)))
        explainer = shap.Explainer(model.predict, shap_sample)
        shap_values = explainer(shap_sample)

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        _save_fig(FIGURES_DIR / "shap_summary.png")
        shap_status = "generated"
    except Exception:
        plt.figure(figsize=(9, 6))
        ordered = top_imp.sort_values("importance")
        plt.barh(ordered["feature"], ordered["importance"], color="#1d4ed8")
        plt.xlabel("Importance (Fallback)")
        plt.title("SHAP Unavailable: Fallback Feature Importance")
        _save_fig(FIGURES_DIR / "shap_summary.png")

    corr_sign = (
        df[usable_features + [target_col]]
        .corr(numeric_only=True)
        .get(target_col, pd.Series(dtype=float))
        .drop(labels=[target_col], errors="ignore")
    )

    explanations = []
    for _, row in top_imp.head(6).iterrows():
        feature = row["feature"]
        direction = "increases" if corr_sign.get(feature, 0) >= 0 else "decreases"
        explanations.append(
            f"- `{feature}` is a top driver; higher values generally **{direction}** predicted CLV in this portfolio context."
        )

    explain_md = [
        "# Explainability Summary",
        "",
        f"- SHAP status: **{shap_status}** (fallback chart is produced when SHAP is unavailable).",
        "- Positive SHAP values indicate a feature contribution that pushes predicted CLV upward.",
        "- Negative SHAP values indicate a feature contribution that pushes predicted CLV downward.",
        "",
        "## Key Feature Interpretations",
    ]
    explain_md.extend(explanations)

    write_text(REPORTS_ROOT / "explainability_summary.md", "\n".join(explain_md))

    pred_clv = model.predict(X)
    if hasattr(classifier, "predict_proba"):
        pred_hv = classifier.predict_proba(X)[:, 1]
    else:
        pred_hv = classifier.predict(X)

    recommendations = _build_business_actions(df, pred_clv, pred_hv)
    write_text(REPORTS_ROOT / "business_recommendations.md", recommendations)

    return {
        "shap_status": shap_status,
        "top_feature_path": str(METRICS_DIR / "top_feature_impacts.csv"),
        "recommendations_path": str(REPORTS_ROOT / "business_recommendations.md"),
    }


if __name__ == "__main__":
    engineered = pd.read_csv("data/processed/engineered_dataset.csv")
    target = "clv" if "clv" in engineered.columns else engineered.columns[-1]
    features = [col for col in engineered.columns if col not in {target, "customer_id"}]
    run_explainability(engineered, target, features)
