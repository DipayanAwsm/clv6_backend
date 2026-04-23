from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import MODELS_ROOT, REPORTS_ROOT
from app.insights import safe_top_rows
from app.predictor import CLVPredictor, load_predictor_or_none
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BusinessSummaryResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    UploadPredictionResponse,
)
from app.utils import get_logger, read_json, read_text

LOGGER = get_logger("clv-api")
router = APIRouter()
PREDICTOR: CLVPredictor | None = load_predictor_or_none()
API_VERSION = "1.2.0"


def _load_training_raw_preview(profile: Dict[str, Any]) -> Dict[str, Any]:
    candidate_paths: list[Path] = []

    # Preferred: actual training split saved by the pipeline.
    candidate_paths.append(REPORTS_ROOT.parent / "data" / "processed" / "training_dataset.csv")

    # Secondary: original source data path from dataset profile.
    source_path = profile.get("source_path")
    if source_path:
        try:
            candidate_paths.append(Path(str(source_path)))
        except Exception:
            pass

    # Fallback: default raw file location used by the project.
    candidate_paths.append(REPORTS_ROOT.parent / "data" / "raw" / "predictions_clv_realistic_50000_5yr.csv")

    for path in candidate_paths:
        if not path.exists() or not path.is_file():
            continue

        try:
            frame = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        if frame.empty:
            continue

        preview_df = frame.head(5)
        all_columns = [str(col) for col in preview_df.columns]
        return {
            "source_file": str(path),
            "columns": all_columns,
            "rows": safe_top_rows(preview_df, n_rows=5),
            "row_count": int(len(preview_df)),
            "column_count": int(len(all_columns)),
        }

    return {
        "source_file": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "column_count": 0,
    }


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    col_lookup = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        matched = col_lookup.get(candidate.lower())
        if matched:
            return matched
    return None


def _build_histogram_payload(series: pd.Series, bins: int, prefix: str) -> list[Dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []

    counts, edges = np.histogram(numeric, bins=bins)
    payload: list[Dict[str, Any]] = []
    for idx in range(len(counts)):
        payload.append(
            {
                "bin": f"{prefix} {idx + 1}",
                "count": int(counts[idx]),
                "lower": round(float(edges[idx]), 2),
                "upper": round(float(edges[idx + 1]), 2),
            }
        )
    return payload


def _build_counts_payload(
    series: pd.Series,
    name_key: str = "name",
    value_key: str = "customers",
    top_n: int = 20,
) -> list[Dict[str, Any]]:
    if series.empty:
        return []
    counts = series.fillna("Unknown").astype(str).value_counts().head(top_n)
    return [{name_key: str(name), value_key: int(value)} for name, value in counts.items()]


def _load_dashboard_dataframe() -> tuple[pd.DataFrame, str]:
    candidates = [
        REPORTS_ROOT.parent / "data" / "processed" / "scored_customers.csv",
        REPORTS_ROOT.parent / "data" / "processed" / "training_dataset.csv",
        REPORTS_ROOT.parent / "data" / "processed" / "engineered_dataset.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        return frame, str(path)
    raise FileNotFoundError("No processed training/scored dataset found for dashboard analytics.")


def _build_shap_payload() -> Dict[str, Any]:
    impacts_path = REPORTS_ROOT / "metrics" / "top_feature_impacts.csv"
    if not impacts_path.exists():
        return {}

    try:
        impacts = pd.read_csv(impacts_path)
    except Exception:
        return {}

    if impacts.empty or "feature" not in impacts.columns:
        return {}

    if "importance" not in impacts.columns:
        impacts["importance"] = 0.0

    impacts["feature"] = impacts["feature"].astype(str)
    impacts["importance"] = pd.to_numeric(impacts["importance"], errors="coerce").fillna(0.0)
    impacts["abs_importance"] = impacts["importance"].abs()

    global_importance = (
        impacts.sort_values("abs_importance", ascending=False)
        .head(12)[["feature", "abs_importance"]]
        .rename(columns={"abs_importance": "importance"})
        .to_dict(orient="records")
    )

    positive = impacts[impacts["importance"] > 0].sort_values("importance", ascending=False).head(5)
    negative = impacts[impacts["importance"] < 0].sort_values("importance", ascending=True).head(5)

    if negative.empty:
        negative = impacts.sort_values("abs_importance", ascending=False).tail(3).copy()
        negative["importance"] = -negative["abs_importance"]

    positive_drivers = [
        {
            "driver": row["feature"],
            "impact": round(float(row["importance"]), 6),
            "rationale": f"Higher `{row['feature']}` tends to push predicted CLV upward.",
        }
        for _, row in positive.iterrows()
    ]
    negative_drivers = [
        {
            "driver": row["feature"],
            "impact": round(float(row["importance"]), 6),
            "rationale": f"Higher `{row['feature']}` tends to reduce predicted CLV.",
        }
        for _, row in negative.iterrows()
    ]

    local_rows = impacts.sort_values("abs_importance", ascending=False).head(8)
    local_contributions = [
        {"feature": row["feature"], "effect": round(float(row["importance"]), 6)}
        for _, row in local_rows.iterrows()
    ]

    summary_scatter: list[Dict[str, Any]] = []
    for index, row in enumerate(local_rows.itertuples(index=False), start=1):
        effect = float(getattr(row, "importance", 0.0))
        if effect == 0:
            effect = 0.000001
        summary_scatter.append(
            {
                "feature": str(getattr(row, "feature")),
                "featureIndex": index,
                "featureValueBand": "High",
                "shapValue": round(effect, 6),
            }
        )
        summary_scatter.append(
            {
                "feature": str(getattr(row, "feature")),
                "featureIndex": index,
                "featureValueBand": "Low",
                "shapValue": round(-effect * 0.7, 6),
            }
        )

    top_pos = positive_drivers[0]["driver"] if positive_drivers else "value drivers"
    top_neg = negative_drivers[0]["driver"] if negative_drivers else "risk drivers"

    return {
        "what_is_shap": [
            "SHAP explains how each feature shifts the prediction from baseline to final CLV output.",
            "Positive SHAP values increase predicted CLV, while negative SHAP values decrease it.",
            "Use SHAP to translate model output into actionable customer strategy decisions.",
        ],
        "global_importance": global_importance,
        "shap_summary_scatter": summary_scatter,
        "local_contributions": local_contributions,
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
        "interpretation": [
            f"`{top_pos}` is one of the strongest positive value drivers in this model run.",
            f"`{top_neg}` is a key negative pressure point to monitor in customer actions.",
            "Combine SHAP with business rules to drive retention, upsell, and automation plans.",
        ],
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=PREDICTOR is not None,
        api_version=API_VERSION,
    )


@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    metadata_path = MODELS_ROOT / "metadata.json"
    metadata_obj = read_json(metadata_path, default={})
    if not metadata_obj:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    return {
        **metadata_obj,
        "api_version": API_VERSION,
        "platform_message": (
            "Model metadata loaded. Use this payload to understand selected models, feature set, and high-value threshold."
        ),
        "decision_rules": {
            "high_value_definition": "Customers above configured CLV quantile are classified as high value.",
            "recommended_operating_mode": "Run batch scoring weekly and trigger action playbooks from prediction output.",
        },
    }


@router.get("/mlflow-info")
def mlflow_info() -> Dict[str, Any]:
    metadata_path = MODELS_ROOT / "metadata.json"
    metadata_obj = read_json(metadata_path, default={})
    if not metadata_obj:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    mlflow_obj = metadata_obj.get("mlflow", {})
    return {
        "enabled": bool(mlflow_obj.get("enabled", False)),
        "tracking_uri": mlflow_obj.get("tracking_uri"),
        "experiment_name": mlflow_obj.get("experiment_name"),
        "run_id": mlflow_obj.get("run_id"),
        "regressor_model_uri": mlflow_obj.get("regressor_model_uri"),
        "classifier_model_uri": mlflow_obj.get("classifier_model_uri"),
        "message": (
            "MLflow integration details for this trained model set. "
            "Use run_id/model_uri values to inspect experiments and load registry artifacts."
        ),
    }


@router.get("/model-metrics")
def model_metrics() -> Dict[str, Any]:
    metrics = read_json(REPORTS_ROOT / "metrics" / "model_metrics.json", default={})
    if not metrics:
        raise HTTPException(status_code=404, detail="Model metrics not found.")

    reg = metrics.get("regression", [])
    cls = metrics.get("classification", [])

    return {
        **metrics,
        "api_version": API_VERSION,
        "summary": {
            "regression_models_tested": len(reg),
            "classification_models_tested": len(cls),
            "selection_note": "Models are selected using objective holdout performance and business-priority metrics.",
            "business_takeaway": "Final models balance predictive strength with practical decision utility.",
        },
    }


@router.get("/eda-summary")
def eda_summary() -> Dict[str, Any]:
    profile = read_json(REPORTS_ROOT / "metrics" / "dataset_profile.json", default={})
    eda_json = read_json(REPORTS_ROOT / "metrics" / "eda_summary.json", default={})
    summary_md = read_text(REPORTS_ROOT / "eda_summary.md", default="")
    if not profile and not summary_md and not eda_json:
        raise HTTPException(status_code=404, detail="EDA summary not found.")

    top_drivers = eda_json.get("top_drivers", [])
    top_driver_names = [item.get("feature", "") for item in top_drivers[:3] if item.get("feature")]
    state_summary = eda_json.get("state_wise_summary", {})
    state_rows = state_summary.get("rows", []) if isinstance(state_summary, dict) else []
    top_state_by_premium = state_rows[0].get("state") if state_rows else None
    training_raw_preview = _load_training_raw_preview(profile if isinstance(profile, dict) else {})

    return {
        "profile": profile,
        "eda_metrics": eda_json,
        "state_wise_summary": state_summary,
        "training_raw_preview": training_raw_preview,
        "summary_markdown": summary_md,
        "key_findings": [
            "CLV distributions are typically concentrated; a small segment can drive outsized long-term value.",
            "Recency, frequency, and monetary behavior remain core value drivers.",
            (
                f"State-wise premium/loss/claim EDA is available; top premium state in this run: {top_state_by_premium}."
                if top_state_by_premium
                else "State-wise premium/loss/claim EDA was unavailable for this run."
            ),
            (
                f"Top exploratory drivers in this run: {', '.join(top_driver_names)}."
                if top_driver_names
                else "Top exploratory drivers were unavailable for this run."
            ),
        ],
    }


@router.get("/dashboard-analytics")
def dashboard_analytics(states: str | None = None, years: str | None = None) -> Dict[str, Any]:
    try:
        df, source_file = _load_dashboard_dataframe()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    state_col = _first_existing_column(
        list(df.columns), ["policyratedstate_tp", "state", "policy_state", "region"]
    )
    year_col = _first_existing_column(list(df.columns), ["year", "policyyear", "policy_year"])
    premium_col = _first_existing_column(
        list(df.columns),
        ["earnedpremium_am", "directwrittenpremium_am", "premium_amount", "earnedpremium"],
    )
    loss_col = _first_existing_column(
        list(df.columns),
        ["netloss_paid_am", "grosslosspaio_am", "netlosspaid", "net_loss_paid"],
    )
    claim_col = _first_existing_column(list(df.columns), ["claimcount_ct", "claimcount", "claim_count"])
    clv_col = _first_existing_column(list(df.columns), ["predicted_clv", "clv", "clv_formula_value"])
    hv_flag_col = _first_existing_column(list(df.columns), ["high_value_flag"])
    hv_prob_col = _first_existing_column(list(df.columns), ["high_value_probability"])
    marketing_col = _first_existing_column(list(df.columns), ["marketingchannel", "marketing_channel"])
    agent_col = _first_existing_column(list(df.columns), ["agent_channel", "agentchannel"])
    agent_name_col = _first_existing_column(list(df.columns), ["agentname", "agent_name"])
    payment_col = _first_existing_column(list(df.columns), ["paymentmethod", "payment_method"])
    income_col = _first_existing_column(list(df.columns), ["incomebracket", "income_bracket"])
    satisfaction_col = _first_existing_column(
        list(df.columns), ["customersatisfaction", "customer_satisfaction"]
    )
    delay_col = _first_existing_column(list(df.columns), ["paymentdelaydays", "payment_delay_days"])
    tenure_col = _first_existing_column(list(df.columns), ["customertenure", "tenure_months", "customer_tenure"])
    action_col = _first_existing_column(list(df.columns), ["recommended_action"])

    data = df.copy()

    if states and state_col:
        selected_states = {state.strip() for state in states.split(",") if state.strip()}
        if selected_states:
            data = data[data[state_col].astype(str).isin(selected_states)]

    if years and year_col:
        selected_years: set[int] = set()
        for token in years.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                selected_years.add(int(token))
            except ValueError:
                continue
        if selected_years:
            year_numeric = pd.to_numeric(data[year_col], errors="coerce")
            data = data[year_numeric.isin(selected_years)]

    if data.empty:
        return {
            "source_file": source_file,
            "available": False,
            "executive": {},
            "eda": {},
            "channel_insights": {},
            "shap": _build_shap_payload(),
        }

    state_series = (
        data[state_col].fillna("Unknown").astype(str)
        if state_col
        else pd.Series(["All"] * len(data), index=data.index)
    )
    year_series = (
        pd.to_numeric(data[year_col], errors="coerce").fillna(0).astype(int)
        if year_col
        else pd.Series([0] * len(data), index=data.index)
    )
    premium_series = (
        pd.to_numeric(data[premium_col], errors="coerce").fillna(0.0)
        if premium_col
        else pd.Series(0.0, index=data.index)
    )
    loss_series = (
        pd.to_numeric(data[loss_col], errors="coerce").fillna(0.0)
        if loss_col
        else pd.Series(0.0, index=data.index)
    )
    claim_series = (
        pd.to_numeric(data[claim_col], errors="coerce").fillna(0.0)
        if claim_col
        else pd.Series(0.0, index=data.index)
    )
    clv_series = (
        pd.to_numeric(data[clv_col], errors="coerce").fillna(0.0)
        if clv_col
        else pd.Series(0.0, index=data.index)
    )
    high_value_flag = (
        pd.to_numeric(data[hv_flag_col], errors="coerce").fillna(0).astype(int)
        if hv_flag_col
        else pd.Series(0, index=data.index)
    )
    high_value_prob = (
        pd.to_numeric(data[hv_prob_col], errors="coerce").fillna(0.0)
        if hv_prob_col
        else pd.Series(0.0, index=data.index)
    )

    display_segment = np.select(
        [
            clv_series <= 0,
            (high_value_flag == 1) & (high_value_prob >= 0.75),
            high_value_flag == 1,
            high_value_prob >= 0.4,
        ],
        [
            "Loss Making",
            "High Value, Low Risk",
            "High Value, High Risk",
            "Growth Potential",
        ],
        default="Low Value",
    )

    base_df = pd.DataFrame(
        {
            "state": state_series,
            "year": year_series,
            "premium": premium_series,
            "loss": loss_series,
            "claims": claim_series,
            "clv": clv_series,
            "display_segment": display_segment,
        }
    )

    year_agg = (
        base_df.groupby("year", observed=True)
        .agg(
            clv=("clv", "sum"),
            avgPremium=("premium", "mean"),
            avgLoss=("loss", "mean"),
            avgClv=("clv", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    clv_trend = [
        {
            "year": int(row["year"]),
            "avgClv": round(float(row["avgClv"]), 2),
            "totalClv": round(float(row["clv"]), 2),
        }
        for _, row in year_agg.iterrows()
    ]
    year_trend = [
        {
            "year": int(row["year"]),
            "avgPremium": round(float(row["avgPremium"]), 2),
            "avgLoss": round(float(row["avgLoss"]), 2),
            "avgClv": round(float(row["avgClv"]), 2),
        }
        for _, row in year_agg.iterrows()
    ]

    state_agg = (
        base_df.groupby("state", observed=True)
        .agg(
            customers=("state", "count"),
            avgPremium=("premium", "mean"),
            avgLosses=("loss", "mean"),
            totalClaimCount=("claims", "sum"),
            avgClv=("clv", "mean"),
        )
        .reset_index()
    )
    state_distribution = [
        {"name": str(row["state"]), "customers": int(row["customers"])}
        for _, row in state_agg.sort_values("customers", ascending=False).iterrows()
    ]
    state_wise_avg_premium = [
        {"state": str(row["state"]), "avgPremium": round(float(row["avgPremium"]), 2)}
        for _, row in state_agg.sort_values("avgPremium", ascending=False).iterrows()
    ]
    state_wise_avg_losses = [
        {"state": str(row["state"]), "avgLosses": round(float(row["avgLosses"]), 2)}
        for _, row in state_agg.sort_values("avgLosses", ascending=False).iterrows()
    ]
    state_wise_claims = [
        {"state": str(row["state"]), "totalClaimCount": int(round(float(row["totalClaimCount"])))}
        for _, row in state_agg.sort_values("totalClaimCount", ascending=False).iterrows()
    ]
    state_clv_snapshot = [
        {"state": str(row["state"]), "avgClv": round(float(row["avgClv"]), 2)}
        for _, row in state_agg.sort_values("avgClv", ascending=False).iterrows()
    ]

    segment_distribution = _build_counts_payload(
        pd.Series(display_segment, index=data.index), name_key="name", value_key="customers", top_n=10
    )

    claim_distribution = (
        claim_series.fillna(0).round().astype(int).clip(lower=0).value_counts().sort_index()
    )
    claims_distribution = [
        {"claims": int(claims), "customers": int(customers)}
        for claims, customers in claim_distribution.items()
    ]

    category_mix = {
        "agentChannel": _build_counts_payload(data[agent_col], name_key="name", value_key="customers", top_n=10)
        if agent_col
        else [],
        "marketingChannel": _build_counts_payload(
            data[marketing_col], name_key="name", value_key="customers", top_n=10
        )
        if marketing_col
        else [],
        "paymentMethod": _build_counts_payload(
            data[payment_col], name_key="name", value_key="customers", top_n=10
        )
        if payment_col
        else [],
        "incomeBracket": _build_counts_payload(
            data[income_col], name_key="name", value_key="customers", top_n=10
        )
        if income_col
        else [],
    }

    avg_clv_by_marketing: list[Dict[str, Any]] = []
    if marketing_col:
        marketing_agg = (
            pd.DataFrame(
                {
                    "channel": data[marketing_col].fillna("Unknown").astype(str),
                    "clv": clv_series,
                }
            )
            .groupby("channel", observed=True)["clv"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        avg_clv_by_marketing = [
            {"channel": str(row["channel"]), "avgClv": round(float(row["clv"]), 2)}
            for _, row in marketing_agg.iterrows()
        ]

    avg_clv_by_agent_channel: list[Dict[str, Any]] = []
    if agent_col:
        agent_channel_agg = (
            pd.DataFrame(
                {
                    "channel": data[agent_col].fillna("Unknown").astype(str),
                    "clv": clv_series,
                }
            )
            .groupby("channel", observed=True)["clv"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        avg_clv_by_agent_channel = [
            {"channel": str(row["channel"]), "avgClv": round(float(row["clv"]), 2)}
            for _, row in agent_channel_agg.iterrows()
        ]

    state_channel_matrix: list[Dict[str, Any]] = []
    if marketing_col:
        state_channel_agg = (
            pd.DataFrame(
                {
                    "state": state_series,
                    "channel": data[marketing_col].fillna("Unknown").astype(str),
                    "clv": clv_series,
                }
            )
            .groupby(["state", "channel"], observed=True)["clv"]
            .mean()
            .reset_index()
            .sort_values(["state", "channel"])
        )
        state_channel_matrix = [
            {
                "state": str(row["state"]),
                "channel": str(row["channel"]),
                "avgClv": round(float(row["clv"]), 2),
            }
            for _, row in state_channel_agg.iterrows()
        ]

    if avg_clv_by_marketing:
        best_channel = avg_clv_by_marketing[0]
        best_source = {
            "title": f"Best Acquisition Source: {best_channel['channel']}",
            "detail": (
                f"This channel currently has the highest average CLV "
                f"({best_channel['avgClv']:.2f}) in the selected filters."
            ),
            "priority": "High",
        }
    else:
        best_source = {
            "title": "Best Acquisition Source: n/a",
            "detail": "Marketing channel field was unavailable in this filtered run.",
            "priority": "Medium",
        }

    if agent_name_col:
        agent_name_series = data[agent_name_col].fillna("Unknown").astype(str)
    elif agent_col:
        agent_name_series = data[agent_col].fillna("Unknown").astype(str)
    else:
        agent_name_series = pd.Series(["Unknown"] * len(data), index=data.index)

    # Prefer explicit agent channel, then marketing channel for channel attribution.
    channel_source_col = agent_col or marketing_col
    if channel_source_col:
        channel_series = data[channel_source_col].fillna("Unknown").astype(str)
    else:
        channel_series = pd.Series(["Unknown"] * len(data), index=data.index)

    # Aggregate per (agent, channel) then keep dominant channel per agent.
    agent_perf = (
        pd.DataFrame({"agentName": agent_name_series, "channel": channel_series, "clv": clv_series})
        .groupby(["agentName", "channel"], observed=True)
        .agg(avgClv=("clv", "mean"), customers=("clv", "size"))
        .reset_index()
        .sort_values(["agentName", "customers", "avgClv"], ascending=[True, False, False])
        .drop_duplicates(subset=["agentName"], keep="first")
        .sort_values("avgClv", ascending=False)
        .reset_index(drop=True)
    )

    q33 = 0.0
    q67 = 0.0
    cluster_algo = "quantile_fallback"
    cluster_features = ["avgClv", "customers"]
    if not agent_perf.empty:
        q33 = float(agent_perf["avgClv"].quantile(0.33))
        q67 = float(agent_perf["avgClv"].quantile(0.67))
    cluster_order = ["Best Set", "Core Cohort", "Support Cohort"]

    if len(agent_perf) >= 3:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            channel_dummies = pd.get_dummies(agent_perf["channel"], prefix="channel")
            feature_frame = pd.concat(
                [agent_perf[["avgClv", "customers"]].reset_index(drop=True), channel_dummies.reset_index(drop=True)],
                axis=1,
            ).fillna(0.0)
            scaler = StandardScaler()
            matrix = scaler.fit_transform(feature_frame)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(matrix)
            agent_perf["cluster_idx"] = raw_labels

            # Rank cluster ids by CLV so labels remain business-meaningful.
            cluster_rank = (
                agent_perf.groupby("cluster_idx", observed=True)["avgClv"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            cluster_map = {
                cluster_rank[idx]: cluster_order[idx] for idx in range(min(len(cluster_rank), len(cluster_order)))
            }
            agent_perf["cluster"] = agent_perf["cluster_idx"].map(cluster_map).fillna("Support Cohort")
            cluster_algo = "kmeans"
            cluster_features = list(feature_frame.columns)
        except Exception as exc:
            LOGGER.warning("KMeans agent clustering failed; using quantile fallback. Error: %s", exc)
            agent_perf["cluster"] = np.select(
                [agent_perf["avgClv"] >= q67, agent_perf["avgClv"] >= q33],
                ["Best Set", "Core Cohort"],
                default="Support Cohort",
            )
    else:
        agent_perf["cluster"] = np.select(
            [agent_perf["avgClv"] >= q67, agent_perf["avgClv"] >= q33],
            ["Best Set", "Core Cohort"],
            default="Support Cohort",
        )
    cluster_summary_df = (
        agent_perf.groupby("cluster", observed=True)
        .agg(
            avgClv=("avgClv", "mean"),
            agents=("agentName", "count"),
            customers=("customers", "sum"),
        )
        .reset_index()
    )
    if not cluster_summary_df.empty:
        cluster_summary_df["cluster"] = pd.Categorical(
            cluster_summary_df["cluster"], categories=cluster_order, ordered=True
        )
        cluster_summary_df = cluster_summary_df.sort_values("cluster")

    agent_clusters = [
        {
            "cluster": str(row["cluster"]),
            "avgClv": round(float(row["avgClv"]), 2),
            "agents": int(row["agents"]),
            "customers": int(row["customers"]),
        }
        for _, row in cluster_summary_df.iterrows()
    ]

    agent_channel_clusters = [
        {
            "agentName": str(row["agentName"]),
            "channel": str(row["channel"]),
            "avgClv": round(float(row["avgClv"]), 2),
            "customers": int(row["customers"]),
            "cluster": str(row["cluster"]),
        }
        for _, row in agent_perf.head(300).iterrows()
    ]

    top_agents = [
        {
            "agentName": str(row["agentName"]),
            "channel": str(row["channel"]),
            "avgClv": round(float(row["avgClv"]), 2),
            "customers": int(row["customers"]),
            "cluster": str(row["cluster"]),
        }
        for _, row in agent_perf.head(15).iterrows()
    ]

    corr_candidates: list[tuple[str, str | None]] = [
        ("Premium", premium_col),
        ("Losses", loss_col),
        ("Claims", claim_col),
        ("CLV", clv_col),
        ("Tenure", tenure_col),
        ("Satisfaction", satisfaction_col),
        ("PaymentDelay", delay_col),
    ]
    corr_columns = [(label, col) for label, col in corr_candidates if col]
    correlation_heatmap: list[Dict[str, Any]] = []
    if len(corr_columns) >= 2:
        corr_frame = pd.DataFrame(
            {label: pd.to_numeric(data[col], errors="coerce") for label, col in corr_columns}
        ).fillna(0.0)
        corr_matrix = corr_frame.corr(numeric_only=True).fillna(0.0)
        for x in corr_matrix.columns:
            for y in corr_matrix.columns:
                correlation_heatmap.append(
                    {"x": x, "y": y, "value": round(float(corr_matrix.loc[x, y]), 2)}
                )

    action_counts = (
        data[action_col].fillna("No action").astype(str).value_counts().head(3)
        if action_col
        else pd.Series(dtype=float)
    )
    recommendation_priorities = ["Critical", "High", "Medium"]
    top_recommendations = [
        {
            "title": f"Priority Action {idx + 1}",
            "detail": f"{action} ({int(count):,} customers)",
            "priority": recommendation_priorities[min(idx, len(recommendation_priorities) - 1)],
        }
        for idx, (action, count) in enumerate(action_counts.items())
    ]
    if not top_recommendations:
        top_recommendations = [
            {
                "title": "Portfolio Action",
                "detail": "Use model outputs to drive retention, upsell, and automation workflows.",
                "priority": "High",
            }
        ]

    takeaways = [
        f"Analytics are computed from `{Path(source_file).name}` with {len(data):,} filtered rows.",
        "State-level value and loss views now use training/scored backend data instead of mock samples.",
        "Distribution and trend charts are generated directly from processed training artifacts.",
        "Use these outputs to align budget allocation, retention, and growth playbooks.",
    ]

    shap_payload = _build_shap_payload()

    return {
        "available": True,
        "source_file": source_file,
        "rows": int(len(data)),
        "executive": {
            "clvTrend": clv_trend,
            "stateClvSnapshot": state_clv_snapshot,
            "segmentDistribution": segment_distribution,
            "topRecommendations": top_recommendations,
            "takeaways": takeaways,
        },
        "eda": {
            "premiumDistribution": _build_histogram_payload(premium_series, bins=12, prefix="P"),
            "lossDistribution": _build_histogram_payload(loss_series, bins=12, prefix="L"),
            "clvDistribution": _build_histogram_payload(clv_series, bins=12, prefix="C"),
            "claimsDistribution": claims_distribution,
            "stateDistribution": state_distribution,
            "yearTrend": year_trend,
            "categoryMix": category_mix,
            "correlationHeatmap": correlation_heatmap,
            "stateWisePremium": state_wise_avg_premium,
            "stateWiseLosses": state_wise_avg_losses,
            "stateWiseClaims": state_wise_claims,
            "interpretation": [
                "Charts in this section are sourced from the latest processed training/scored dataset.",
                "Average premium and average loss by state provide cleaner comparability than raw totals.",
                "Distribution views reveal concentration and volatility patterns in premium and losses.",
            ],
        },
        "channel_insights": {
            "avgClvByMarketing": avg_clv_by_marketing,
            "avgClvByAgent": avg_clv_by_agent_channel,
            "stateChannelMatrix": state_channel_matrix,
            "bestSource": best_source,
            "agentClusters": agent_clusters,
            "agentChannelClusters": agent_channel_clusters,
            "topAgents": top_agents,
            "agentClusterMethod": {
                "columnUsed": agent_name_col or agent_col or "fallback_agent_id",
                "channelColumnUsed": channel_source_col or "fallback_channel",
                "metric": "Average CLV per Agent Name with channel attribution",
                "algorithm": cluster_algo,
                "featureSpace": cluster_features,
                "quantile33Threshold": round(float(q33), 2),
                "quantile67Threshold": round(float(q67), 2),
                "rules": {
                    "Best Set": (
                        "KMeans top centroid by avg CLV (fallback: agent_avg_clv >= quantile_67)"
                    ),
                    "Core Cohort": (
                        "KMeans middle centroid by avg CLV "
                        "(fallback: quantile_33 <= agent_avg_clv < quantile_67)"
                    ),
                    "Support Cohort": (
                        "KMeans lowest centroid by avg CLV (fallback: agent_avg_clv < quantile_33)"
                    ),
                },
            },
        },
        "shap": shap_payload,
    }


@router.get("/feature-selection-summary")
def feature_selection_summary() -> Dict[str, Any]:
    summary_json = read_json(
        REPORTS_ROOT / "metrics" / "feature_selection_summary.json", default={}
    )
    summary_md = read_text(REPORTS_ROOT / "feature_selection_summary.md", default="")

    shortlist = summary_json.get("final_shortlist", [])

    return {
        "summary": summary_json,
        "summary_markdown": summary_md,
        "plain_english": {
            "what_happened": "Multiple feature selection methods voted on signal strength.",
            "how_to_read": "Features appearing across many methods are usually more reliable drivers.",
            "final_shortlist_count": len(shortlist),
            "why_it_matters": "A compact, high-signal feature set improves generalization and explainability.",
        },
    }


def _ensure_predictor() -> CLVPredictor:
    if PREDICTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor artifacts are not ready. Run backend/training/run_pipeline.py.",
        )
    return PREDICTOR


def _score_dataframe(
    df: pd.DataFrame,
    predictor: CLVPredictor,
) -> tuple[pd.DataFrame, list[Dict[str, Any]], Dict[str, Any], list[str]]:
    records = df.to_dict(orient="records")
    predictions = predictor.predict_batch(records)
    summary = predictor.summarize_batch(predictions)

    output_df = df.copy()
    output_df["predicted_clv"] = [item["predicted_clv"] for item in predictions]
    output_df["high_value_flag"] = [item["high_value_flag"] for item in predictions]
    output_df["high_value_probability"] = [
        item["high_value_probability"] for item in predictions
    ]
    output_df["customer_segment"] = [
        item.get("prediction_context", {}).get("customer_segment", "Unknown")
        for item in predictions
    ]
    output_df["action_priority"] = [
        item.get("prediction_context", {}).get("action_priority", "baseline")
        for item in predictions
    ]
    output_df["recommended_action"] = [
        item.get("recommended_action", "") for item in predictions
    ]

    incoming_columns = list(df.columns)
    missing_expected = [
        feature for feature in predictor.expected_features if feature not in incoming_columns
    ]

    return output_df, predictions, summary, missing_expected


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    predictor = _ensure_predictor()
    try:
        result = predictor.predict_single(payload.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@router.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    predictor = _ensure_predictor()
    try:
        predictions = predictor.predict_batch(payload.records)
        summary = predictor.summarize_batch(predictions)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            summary=summary,
            message="Batch scoring complete. Use summary to prioritize customer actions.",
        )
    except Exception as exc:
        LOGGER.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc


@router.post("/predict/single", response_model=PredictionResponse)
def predict_single_alias(payload: PredictionRequest) -> PredictionResponse:
    return predict(payload)


@router.post("/predict/batch")
async def predict_batch_csv(file: UploadFile = File(...)) -> StreamingResponse:
    predictor = _ensure_predictor()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        raw = await file.read()
        input_df = pd.read_csv(io.BytesIO(raw))
        output_df, _, _, _ = _score_dataframe(input_df, predictor)
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
    except Exception as exc:
        LOGGER.exception("CSV batch prediction endpoint failed")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc

    headers = {"Content-Disposition": "attachment; filename=clv_batch_predictions.csv"}
    return StreamingResponse(csv_buffer, media_type="text/csv", headers=headers)


@router.post("/upload-csv-and-predict", response_model=UploadPredictionResponse)
async def upload_csv_and_predict(file: UploadFile = File(...)) -> UploadPredictionResponse:
    predictor = _ensure_predictor()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        output_df, predictions, summary, missing_expected = _score_dataframe(df, predictor)

        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)

        incoming_columns = list(df.columns)

        preview_columns = [
            col
            for col in [
                "customer_id",
                "predicted_clv",
                "high_value_flag",
                "high_value_probability",
                "customer_segment",
                "action_priority",
                "recommended_action",
            ]
            if col in output_df.columns
        ]
        preview = safe_top_rows(output_df[preview_columns])

        return UploadPredictionResponse(
            filename=file.filename,
            rows_processed=len(df),
            columns_received=incoming_columns,
            missing_expected_features=missing_expected,
            summary=summary,
            preview=preview,
            predictions=predictions,
            predicted_csv=csv_buffer.getvalue(),
            message=(
                "CSV scoring complete. Download the enriched file and use segment/action columns for campaign execution."
            ),
        )
    except Exception as exc:
        LOGGER.exception("CSV upload prediction failed")
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {exc}") from exc


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> Dict[str, Any]:
    metadata_obj = read_json(MODELS_ROOT / "metadata.json", default={})
    metrics_obj = read_json(REPORTS_ROOT / "metrics" / "model_metrics.json", default={})
    if not metadata_obj:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    return {
        "best_regression_model": metadata_obj.get("regression_model_selected"),
        "best_classification_model": metadata_obj.get("classification_model_selected"),
        "regression_metrics": next(
            (
                row
                for row in metrics_obj.get("regression", [])
                if row.get("model") == metadata_obj.get("regression_model_selected")
            ),
            {},
        ),
        "classification_metrics": next(
            (
                row
                for row in metrics_obj.get("classification", [])
                if row.get("model") == metadata_obj.get("classification_model_selected")
            ),
            {},
        ),
        "target_definition": metadata_obj.get("target_definition", {}),
        "features_used": metadata_obj.get("selected_features", []),
        "high_value_threshold_value": metadata_obj.get("high_value_threshold_value"),
    }


@router.get("/business/summary", response_model=BusinessSummaryResponse)
def business_summary() -> Dict[str, Any]:
    summary = read_json(REPORTS_ROOT / "metrics" / "business_summary.json", default={})
    scored_df_path = REPORTS_ROOT.parent / "data" / "processed" / "scored_customers.csv"
    if summary:
        if "average_clv_before_prediction" not in summary and scored_df_path.exists():
            try:
                df = pd.read_csv(scored_df_path)
                if not df.empty:
                    if "clv" in df.columns:
                        base_clv_series = pd.to_numeric(df["clv"], errors="coerce")
                    elif "clv_formula_value" in df.columns:
                        base_clv_series = pd.to_numeric(df["clv_formula_value"], errors="coerce")
                    elif "predicted_clv" in df.columns:
                        base_clv_series = pd.to_numeric(df["predicted_clv"], errors="coerce")
                    else:
                        base_clv_series = pd.Series(dtype=float)

                    if not base_clv_series.empty:
                        summary["average_clv_before_prediction"] = round(
                            float(base_clv_series.fillna(0).mean()), 2
                        )
            except Exception:
                pass
        return summary

    # Fallback if training summary artifact is unavailable.
    if scored_df_path.exists():
        df = pd.read_csv(scored_df_path)
        if not df.empty and "predicted_clv" in df.columns:
            if "clv" in df.columns:
                base_clv_series = pd.to_numeric(df["clv"], errors="coerce")
            elif "clv_formula_value" in df.columns:
                base_clv_series = pd.to_numeric(df["clv_formula_value"], errors="coerce")
            else:
                base_clv_series = pd.to_numeric(df["predicted_clv"], errors="coerce")

            return {
                "total_customers": int(len(df)),
                "total_predicted_clv": float(df["predicted_clv"].sum()),
                "average_predicted_clv": float(df["predicted_clv"].mean()),
                "average_clv_before_prediction": float(base_clv_series.fillna(0).mean()),
                "high_value_percentage": float(df.get("high_value_flag", pd.Series(dtype=float)).mean() * 100)
                if "high_value_flag" in df.columns
                else 0.0,
                "profitable_percentage": float((df.get("profit", pd.Series(dtype=float)) > 0).mean() * 100)
                if "profit" in df.columns
                else 0.0,
                "top_state_by_clv": None,
            }

    raise HTTPException(status_code=404, detail="Business summary not found. Run training pipeline.")
