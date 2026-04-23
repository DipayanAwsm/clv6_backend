from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from training.common import DEMO_DATA_DIR, LOGGER

RNG = np.random.default_rng(42)


def _metric_dict_from_summary(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    cols = set(df.columns)

    if {"metric", "value"}.issubset(cols):
        for _, row in df.iterrows():
            key = str(row["metric"]).strip().lower().replace(" ", "_")
            val = row["value"]
            if pd.notna(val):
                try:
                    metrics[key] = float(val)
                except Exception:
                    continue
        return metrics

    if {"feature", "mean"}.issubset(cols):
        for _, row in df.iterrows():
            feature = str(row["feature"]).strip().lower().replace(" ", "_")
            for stat_col in ["mean", "std", "min", "max"]:
                if stat_col in row and pd.notna(row[stat_col]):
                    try:
                        metrics[f"{feature}_{stat_col}"] = float(row[stat_col])
                    except Exception:
                        continue
        return metrics

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and len(df) <= 10:
            metrics[f"{col}_mean"] = float(df[col].mean())
            metrics[f"{col}_std"] = float(df[col].std(ddof=0) if len(df[col].dropna()) > 1 else 0)

    return metrics


def _pick(metrics: Dict[str, float], keys: list[str], default: float) -> float:
    for key in keys:
        if key in metrics and np.isfinite(metrics[key]):
            return float(metrics[key])
    return float(default)


def build_calibrated_demo_dataset(summary_df: pd.DataFrame, n_customers: int = 2500) -> pd.DataFrame:
    metrics = _metric_dict_from_summary(summary_df)

    mean_tenure = _pick(metrics, ["tenure_months_mean", "avg_tenure", "tenure_mean"], 34)
    std_tenure = max(_pick(metrics, ["tenure_months_std", "tenure_std"], 12), 3)

    mean_freq = _pick(metrics, ["frequency_mean", "transactions_mean", "orders_mean"], 8)
    mean_monetary = _pick(metrics, ["monetary_mean", "total_spend_mean", "premium_mean"], 1400)
    mean_income = _pick(metrics, ["annual_income_mean", "income_mean"], 72000)

    customer_id = [f"CUST-{idx:06d}" for idx in range(1, n_customers + 1)]
    tenure_months = np.clip(RNG.normal(mean_tenure, std_tenure, n_customers), 1, 180)
    recency = np.clip(RNG.gamma(shape=2.2, scale=25, size=n_customers), 1, 365)
    frequency = np.clip(RNG.poisson(lam=max(mean_freq, 1), size=n_customers), 1, 48)
    average_order_value = np.clip(
        RNG.normal(mean_monetary / max(mean_freq, 1), 45, size=n_customers), 30, 1200
    )
    monetary = np.clip(frequency * average_order_value + RNG.normal(0, 120, n_customers), 60, None)

    claims_count = np.clip(RNG.poisson(lam=0.6, size=n_customers), 0, 12)
    complaints_count = np.clip(RNG.poisson(lam=0.4, size=n_customers), 0, 10)
    renewals_count = np.clip((tenure_months / 12 + RNG.normal(0, 0.7, n_customers)).round(), 0, None)
    policies_count = np.clip((renewals_count + RNG.integers(1, 4, n_customers)), 1, None)

    annual_income = np.clip(RNG.normal(mean_income, 18000, n_customers), 18000, 280000)
    age = np.clip(RNG.normal(41, 11, n_customers), 18, 83)

    premium_amount = np.clip(monetary * RNG.uniform(0.45, 0.92, n_customers), 50, None)

    channel = RNG.choice(["online", "agent", "branch", "partner"], size=n_customers, p=[0.35, 0.3, 0.2, 0.15])
    region = RNG.choice(["north", "south", "east", "west"], size=n_customers)
    product_type = RNG.choice(["basic", "plus", "premium", "enterprise"], size=n_customers, p=[0.32, 0.33, 0.24, 0.11])

    renewal_ratio = np.divide(
        renewals_count,
        np.where(policies_count == 0, 1, policies_count),
    )
    complaint_rate = np.divide(
        complaints_count,
        np.where(frequency == 0, 1, frequency),
    )

    clv = (
        1100
        + 7.5 * tenure_months
        + 42 * frequency
        + 0.72 * monetary
        + 960 * renewal_ratio
        - 4.4 * recency
        - 430 * complaint_rate
        + 0.02 * annual_income
        + RNG.normal(0, 420, n_customers)
    )
    clv = np.clip(clv, 120, None)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age.round(0),
            "annual_income": annual_income.round(2),
            "tenure_months": tenure_months.round(1),
            "days_since_last_purchase": recency.round(1),
            "transactions": frequency,
            "total_spend": monetary.round(2),
            "premium_amount": premium_amount.round(2),
            "claims_count": claims_count,
            "complaints_count": complaints_count,
            "renewals_count": renewals_count.astype(int),
            "policies_count": policies_count.astype(int),
            "channel": channel,
            "region": region,
            "product_type": product_type,
            "clv": clv.round(2),
        }
    )

    return df


def save_demo_dataset(df: pd.DataFrame, filename: str = "calibrated_demo_dataset.csv") -> Path:
    output_path = DEMO_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved calibrated demo dataset to %s", output_path)
    return output_path


if __name__ == "__main__":
    # Fallback standalone behavior: generate a synthetic dataset without summary input.
    synthetic_df = build_calibrated_demo_dataset(pd.DataFrame({"metric": [], "value": []}), n_customers=3000)
    save_demo_dataset(synthetic_df)
