from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.utils import write_json, write_text
from training.common import FIGURES_DIR, LOGGER, METRICS_DIR, RAW_DATA_DIR
from app.config import PROJECT_ROOT, REPORTS_ROOT


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        str(col)
        .strip()
        .lower()
        .replace("%", "percent")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        for col in cleaned.columns
    ]
    return cleaned


def choose_input_csv(preferred_path: str | None = None) -> Path:
    if preferred_path:
        path = Path(preferred_path).expanduser()
        if path.exists():
            return path

    preferred_names = [
        "predictions_clv_realistic_50000_5yr.csv",
        "clv_realistic_50000_5yr.csv",
        "customer_clv_row_level.csv",
    ]

    search_roots = [RAW_DATA_DIR, PROJECT_ROOT, PROJECT_ROOT.parent]
    for root in search_roots:
        for file_name in preferred_names:
            candidate = root / file_name
            if candidate.exists():
                return candidate

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        prioritized = sorted(
            csv_files,
            key=lambda p: (
                "predictions_clv_realistic_50000_5yr" not in p.name.lower(),
                "clv_realistic_50000_5yr" not in p.name.lower(),
                p.name,
            ),
        )
        return prioritized[0]

    raise FileNotFoundError(
        f"No CSV found in {RAW_DATA_DIR}. Provide an input path or add a CSV to raw data."
    )


def load_input_csv(path: Path | str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    return normalize_column_names(df)


def inspect_dataset(df: pd.DataFrame, source_path: str) -> Dict[str, Any]:
    missing = df.isna().sum().sort_values(ascending=False)
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}

    profile = {
        "source_path": source_path,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": {k: int(v) for k, v in missing.to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    write_json(METRICS_DIR / "dataset_profile.json", profile)
    return profile


def detect_dataset_type(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    colset = set(df.columns)

    summary_keywords = {
        "metric",
        "value",
        "mean",
        "std",
        "min",
        "max",
        "summary",
        "statistic",
    }

    metric_value_shape = {"metric", "value"}.issubset(colset)
    keyword_hits = sum(1 for c in df.columns if any(k in c for k in summary_keywords))
    low_grain = rows <= 30 and cols <= 25

    id_like = any(
        token in col
        for col in df.columns
        for token in ["customer_id", "client_id", "account_id", "policy_id"]
    )
    behavior_like = sum(
        1
        for col in df.columns
        if any(
            token in col
            for token in [
                "transaction",
                "purchase",
                "spend",
                "premium",
                "frequency",
                "recency",
                "tenure",
                "complaint",
                "claim",
                "renewal",
                "order",
            ]
        )
    )

    if metric_value_shape:
        return "summary_level"
    if low_grain and keyword_hits >= 2 and behavior_like <= 3:
        return "summary_level"
    if rows <= 10 and not id_like:
        return "summary_level"
    return "row_level"


def infer_target_column(df: pd.DataFrame) -> str | None:
    preferred_candidates = [
        "clv",
        "customer_lifetime_value",
        "lifetime_value",
        "clv_target",
        "target_clv",
        "total_customer_value",
    ]

    for col in preferred_candidates:
        if col in df.columns:
            return col

    # `predicted_clv` is frequently a model output column; use only if it has real variance.
    if "predicted_clv" in df.columns:
        pred = pd.to_numeric(df["predicted_clv"], errors="coerce")
        if pred.nunique(dropna=True) > 10 and float(pred.std(ddof=0) or 0) > 1e-6:
            return "predicted_clv"

    value_like = [
        col
        for col in df.columns
        if any(token in col for token in ["clv", "lifetime"]) and df[col].dtype != "O"
    ]
    return value_like[0] if value_like else None


def _first_existing(columns: List[str], candidates: List[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def perform_eda(df: pd.DataFrame, target_col: str | None) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    missing_df = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
        .sort_values("missing_count", ascending=False)
    )
    missing_df.to_csv(METRICS_DIR / "missing_values_report.csv", index=False)

    numeric_summary = df[numeric_cols].describe().transpose() if numeric_cols else pd.DataFrame()
    numeric_summary.to_csv(METRICS_DIR / "numeric_summary.csv")

    if categorical_cols:
        cat_summary_rows: List[Dict[str, Any]] = []
        for col in categorical_cols:
            top = df[col].value_counts(dropna=False).head(5)
            cat_summary_rows.append(
                {
                    "column": col,
                    "unique_values": int(df[col].nunique(dropna=False)),
                    "top_values": "; ".join([f"{k}:{v}" for k, v in top.to_dict().items()]),
                }
            )
        pd.DataFrame(cat_summary_rows).to_csv(METRICS_DIR / "categorical_summary.csv", index=False)

    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[target_col].dropna(), bins=30, color="#2b6cb0", alpha=0.85)
        ax.set_title("CLV Distribution")
        ax.set_xlabel("CLV")
        ax.set_ylabel("Customers")
        _save_plot(fig, FIGURES_DIR / "clv_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(df[target_col].dropna())
        ax.set_title("CLV Boxplot")
        ax.set_ylabel("CLV")
        _save_plot(fig, FIGURES_DIR / "clv_boxplot.png")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="Blues", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title("Correlation Heatmap")
        _save_plot(fig, FIGURES_DIR / "correlation_heatmap.png")

    relationship_notes: List[str] = []
    top_drivers: List[Tuple[str, float]] = []
    if target_col and target_col in df.columns and target_col in numeric_cols:
        target_corr = (
            df[numeric_cols]
            .corr(numeric_only=True)[target_col]
            .drop(labels=[target_col], errors="ignore")
            .dropna()
            .abs()
            .sort_values(ascending=False)
        )
        top_drivers = [(k, float(v)) for k, v in target_corr.head(8).to_dict().items()]

        for feature, _ in top_drivers[:3]:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(df[feature], df[target_col], alpha=0.4, color="#0f766e")
            ax.set_title(f"{feature} vs {target_col}")
            ax.set_xlabel(feature)
            ax.set_ylabel(target_col)
            _save_plot(fig, FIGURES_DIR / f"feature_target_{feature}.png")

        relationship_notes = [
            f"`{name}` shows strong exploratory correlation with CLV (|corr|={score:.2f})."
            for name, score in top_drivers[:5]
        ]

    state_col = _first_existing(
        list(df.columns),
        [
            "policyratedstate_tp",
            "policy_state",
            "state",
            "state_code",
            "region",
        ],
    )
    premium_col = _first_existing(
        list(df.columns),
        [
            "earnedpremium_am",
            "directwrittenpremium_am",
            "premium_amount",
            "annual_premium",
        ],
    )
    loss_col = _first_existing(
        list(df.columns),
        [
            "netloss_paid_am",
            "grosslosspaio_am",
            "grosslosspaid_am",
            "loss_paid_amount",
        ],
    )
    claim_col = _first_existing(
        list(df.columns),
        [
            "claimcount_ct",
            "claims_count",
            "claims",
            "num_claims",
        ],
    )

    state_wise_rows: List[Dict[str, Any]] = []
    state_summary_available = False
    if state_col and any([premium_col, loss_col, claim_col]):
        state_df = df[[state_col] + [c for c in [premium_col, loss_col, claim_col] if c]].copy()
        state_df[state_col] = state_df[state_col].fillna("Unknown")
        rename_map = {}
        if premium_col:
            rename_map[premium_col] = "total_premium"
        if loss_col:
            rename_map[loss_col] = "total_losses"
        if claim_col:
            rename_map[claim_col] = "total_claim_count"

        state_agg = (
            state_df.groupby(state_col, observed=True)[list(rename_map.keys())]
            .sum(numeric_only=True)
            .rename(columns=rename_map)
            .reset_index()
            .rename(columns={state_col: "state"})
        )

        for metric_col in ["total_premium", "total_losses", "total_claim_count"]:
            if metric_col not in state_agg.columns:
                state_agg[metric_col] = 0.0

        state_agg = state_agg.sort_values("total_premium", ascending=False)
        state_agg.to_csv(METRICS_DIR / "state_wise_eda.csv", index=False)
        state_wise_rows = state_agg.head(15).to_dict(orient="records")
        state_summary_available = True

    segment_summary_path = METRICS_DIR / "segment_summary.csv"
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        segment_df = df.copy()
        bucket_codes = pd.qcut(
            segment_df[target_col],
            q=4,
            labels=False,
            duplicates="drop",
        )
        n_bins = int(pd.Series(bucket_codes).dropna().nunique())
        n_bins = max(n_bins, 1)
        label_pool = ["Low", "Medium", "High", "Premium"]
        labels = (
            label_pool[:n_bins]
            if n_bins <= len(label_pool)
            else [f"Segment_{i+1}" for i in range(n_bins)]
        )
        label_map = {idx: label for idx, label in enumerate(labels)}
        segment_df["clv_segment"] = pd.Series(bucket_codes, index=segment_df.index).map(label_map)
        agg_candidates = [
            c
            for c in ["frequency", "monetary", "recency", "tenure_months", target_col]
            if c in segment_df.columns and pd.api.types.is_numeric_dtype(segment_df[c])
        ]
        if agg_candidates:
            seg_summary = segment_df.groupby("clv_segment", observed=True)[agg_candidates].mean().round(2)
            seg_summary.to_csv(segment_summary_path)

    skew_note = ""
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        skew_value = float(df[target_col].skew())
        skew_note = (
            "CLV is strongly right-skewed, indicating a small segment likely drives a disproportionate share of long-term revenue."
            if skew_value > 1
            else "CLV skew is moderate, suggesting value concentration exists but is not extreme."
        )

    complaint_note = (
        "Complaint indicators should be tracked as service-experience risk signals that can suppress long-term value."
        if any("complaint" in c for c in df.columns)
        else "Complaint fields are absent, so service-friction effects could not be quantified directly."
    )
    delay_note = (
        "Payment-delay style features can be used as early warning signals for collections and retention risk."
        if any(token in c for c in df.columns for token in ["delay", "overdue", "dpd"])
        else "No explicit payment delay feature was available; risk proxies should be added in future ingestion."
    )
    renewal_note = (
        "Renewal behavior appears in the dataset and is expected to be a major CLV driver."
        if any("renew" in c for c in df.columns)
        else "Renewal behavior fields were limited; the model relies more on recency/frequency/spend proxies."
    )

    eda_md = [
        "# EDA Summary",
        "",
        "## Base Data Overview",
        f"- Rows: **{df.shape[0]}**",
        f"- Columns: **{df.shape[1]}**",
        f"- Duplicate rows: **{int(df.duplicated().sum())}**",
        f"- Numeric columns: **{len(numeric_cols)}**",
        f"- Categorical columns: **{len(categorical_cols)}**",
        "",
        "## Data Quality",
        "- Missing values report saved to `reports/metrics/missing_values_report.csv`.",
        "- Numeric profile saved to `reports/metrics/numeric_summary.csv`.",
        "- Categorical profile saved to `reports/metrics/categorical_summary.csv` when applicable.",
        "",
        "## Business Interpretation",
        f"- {skew_note}" if skew_note else "- CLV target unavailable at this stage; distribution insights deferred.",
        f"- {complaint_note}",
        f"- {delay_note}",
        f"- {renewal_note}",
        "- High spend, long tenure, and healthy renewal behavior generally indicate premium-value customer profiles.",
        "",
        "## Exploratory Drivers",
    ]
    eda_md.extend([f"- {note}" for note in relationship_notes] or ["- Target-driver correlation analysis was limited by available target fields."])
    if state_summary_available:
        eda_md.extend(
            [
                "",
                "## State-Wise Portfolio View",
                f"- Premium by state was aggregated using `{premium_col}`.",
                f"- Losses by state were aggregated using `{loss_col}`." if loss_col else "- Loss column unavailable for state-wise losses.",
                f"- Claim count by state was aggregated using `{claim_col}`." if claim_col else "- Claim count column unavailable for state-wise claims.",
                "- State summary table saved to `reports/metrics/state_wise_eda.csv`.",
            ]
        )
    eda_md.extend(
        [
            "",
            "## Artifacts",
            "- `reports/figures/clv_histogram.png`",
            "- `reports/figures/clv_boxplot.png`",
            "- `reports/figures/correlation_heatmap.png`",
            "- `reports/metrics/segment_summary.csv`",
            "- `reports/metrics/state_wise_eda.csv`",
        ]
    )

    write_text(REPORTS_ROOT / "eda_summary.md", "\n".join(eda_md))

    target_series = (
        pd.to_numeric(df[target_col], errors="coerce")
        if target_col and target_col in df.columns
        else pd.Series(dtype=float)
    )
    summary_payload = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target_column": target_col,
        "target_readiness": {
            "target_available": bool(target_col and target_col in df.columns),
            "target_numeric": bool(target_col and target_col in numeric_cols),
            "target_unique_values": int(target_series.nunique(dropna=True)) if not target_series.empty else 0,
            "target_std": float(target_series.std(ddof=0)) if not target_series.empty else 0.0,
        },
        "top_drivers": [{"feature": k, "abs_corr": v} for k, v in top_drivers],
        "state_wise_summary": {
            "available": state_summary_available,
            "state_column": state_col,
            "premium_column": premium_col,
            "loss_column": loss_col,
            "claim_column": claim_col,
            "rows": state_wise_rows,
        },
    }
    write_json(METRICS_DIR / "eda_summary.json", summary_payload)
    LOGGER.info("EDA completed with %d numeric columns and %d categorical columns", len(numeric_cols), len(categorical_cols))
    return summary_payload
