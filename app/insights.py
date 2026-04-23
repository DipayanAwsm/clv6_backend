from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def confidence_band(probability: float) -> str:
    distance = abs(float(probability) - 0.5)
    if distance >= 0.35:
        return "high"
    if distance >= 0.2:
        return "medium"
    return "low"


def infer_churn_risk_score(row: pd.Series) -> float:
    recency = float(row.get("recency", row.get("days_since_last_purchase", 0)) or 0)
    complaint_rate = float(row.get("complaint_rate", 0) or 0)
    renewal_ratio = float(row.get("renewal_ratio", 0) or 0)

    recency_norm = clamp(recency / 365.0, 0.0, 1.0)
    complaint_norm = clamp(complaint_rate / 0.6, 0.0, 1.0)
    renewal_norm = clamp(renewal_ratio, 0.0, 1.0)

    score = 0.45 * recency_norm + 0.35 * complaint_norm + 0.20 * (1.0 - renewal_norm)
    return round(clamp(score, 0.0, 1.0), 3)


def customer_segment_from_clv(predicted_clv: float, high_value_threshold: float) -> str:
    threshold = max(float(high_value_threshold), 1.0)
    clv = float(predicted_clv)
    if clv >= threshold * 1.2:
        return "Strategic Premium"
    if clv >= threshold:
        return "High Value"
    if clv >= threshold * 0.7:
        return "Growth Potential"
    return "Base Portfolio"


def reason_codes(
    row: pd.Series,
    high_value_flag: int,
    predicted_clv: float,
    churn_risk_score: float,
) -> List[str]:
    reasons: List[str] = []
    frequency = float(row.get("frequency", row.get("transactions", 0)) or 0)
    monetary = float(row.get("monetary", row.get("total_spend", 0)) or 0)
    recency = float(row.get("recency", row.get("days_since_last_purchase", 0)) or 0)
    renewal_ratio = float(row.get("renewal_ratio", 0) or 0)
    complaint_rate = float(row.get("complaint_rate", 0) or 0)

    if frequency >= 8 and monetary >= 1200:
        reasons.append("Purchase frequency and spend indicate strong value momentum")
    if renewal_ratio >= 0.6:
        reasons.append("Renewal behavior supports higher projected lifetime value")
    if recency >= 120:
        reasons.append("Recent inactivity increases retention risk")
    if complaint_rate >= 0.2:
        reasons.append("Service friction may suppress realized customer value")
    if churn_risk_score >= 0.65:
        reasons.append("Composite risk score is elevated and requires proactive handling")
    if high_value_flag and predicted_clv > 0 and not reasons:
        reasons.append("Combined profile places customer in high-value band")

    if not reasons:
        reasons.append("Profile suggests stable value with moderate growth opportunity")

    return reasons[:3]


def recommended_strategy(
    high_value_flag: int,
    high_value_probability: float,
    churn_risk_score: float,
) -> Dict[str, str]:
    if high_value_flag and churn_risk_score >= 0.6:
        return {
            "action_priority": "critical",
            "recommended_action": "Activate urgent retention playbook with executive outreach and service recovery.",
            "budget_treatment": "Prioritize retention budget; assign premium service owner immediately.",
        }
    if high_value_flag:
        return {
            "action_priority": "high",
            "recommended_action": "Prioritize loyalty, proactive account management, and upsell offers.",
            "budget_treatment": "Allocate premium relationship budget and upsell resources.",
        }
    if high_value_probability >= 0.4:
        return {
            "action_priority": "medium",
            "recommended_action": "Run nurture journeys with targeted cross-sell and engagement nudges.",
            "budget_treatment": "Allocate monitored growth budget with monthly performance reviews.",
        }
    return {
        "action_priority": "baseline",
        "recommended_action": "Manage through cost-efficient automated engagement and periodic monitoring.",
        "budget_treatment": "Use low-cost automation budget and monitor for segment migration.",
    }


def explanation_message(
    predicted_clv: float,
    high_value_flag: int,
    high_value_probability: float,
    churn_risk_score: float,
) -> str:
    value_band = "high-value" if high_value_flag else "non-premium"
    return (
        f"Predicted CLV is {predicted_clv:.2f}; customer is currently classified as {value_band} "
        f"with high-value probability {high_value_probability:.1%} and churn-risk score {churn_risk_score:.2f}."
    )


def summarize_batch_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not predictions:
        return {
            "average_predicted_clv": 0.0,
            "high_value_customers": 0,
            "high_value_rate": 0.0,
            "average_high_value_probability": 0.0,
            "segment_mix": {},
            "top_recommended_action": "n/a",
        }

    df = pd.DataFrame(predictions)
    segment_series = df.apply(
        lambda row: (row.get("prediction_context") or {}).get("customer_segment", "Unknown"),
        axis=1,
    )
    segment_mix = Counter(segment_series.fillna("Unknown"))
    actions = Counter(df.get("recommended_action", pd.Series(dtype=object)).fillna("n/a"))

    high_value_count = int(df.get("high_value_flag", pd.Series(dtype=float)).fillna(0).sum())

    return {
        "average_predicted_clv": round(float(df["predicted_clv"].mean()), 2),
        "high_value_customers": high_value_count,
        "high_value_rate": round(high_value_count / len(df), 4),
        "average_high_value_probability": round(float(df["high_value_probability"].mean()), 4),
        "segment_mix": dict(segment_mix),
        "top_recommended_action": actions.most_common(1)[0][0] if actions else "n/a",
    }


def safe_top_rows(df: pd.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    preview = df.head(n_rows).replace({np.nan: None})
    return preview.to_dict(orient="records")
