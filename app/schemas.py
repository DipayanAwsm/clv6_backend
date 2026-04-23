from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictionRequest(ApiBaseModel):
    customer_id: Optional[str] = None
    age: Optional[float] = None
    annual_income: Optional[float] = None
    tenure_months: Optional[float] = None
    recency: Optional[float] = None
    frequency: Optional[float] = None
    monetary: Optional[float] = None
    average_order_value: Optional[float] = None
    claim_rate: Optional[float] = None
    complaint_rate: Optional[float] = None
    renewal_ratio: Optional[float] = None
    premium_amount: Optional[float] = None
    channel: Optional[str] = None
    region: Optional[str] = None
    product_type: Optional[str] = None

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class BatchPredictionRequest(ApiBaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictionResponse(ApiBaseModel):
    predicted_clv: float
    high_value_flag: int
    high_value_probability: float
    selected_input_fields: List[str]
    missing_expected_fields: List[str]
    top_reason_codes: List[str]
    explanation_message: str
    recommended_action: str
    budget_treatment: str
    prediction_context: Dict[str, Any]
    model_context: Dict[str, Any]


class BatchPredictionResponse(ApiBaseModel):
    predictions: List[Dict[str, Any]]
    count: int
    summary: Dict[str, Any]
    message: str


class UploadPredictionResponse(ApiBaseModel):
    filename: str
    rows_processed: int
    columns_received: List[str]
    missing_expected_features: List[str]
    summary: Dict[str, Any]
    preview: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    predicted_csv: str
    message: str


class HealthResponse(ApiBaseModel):
    status: str
    model_ready: bool
    api_version: str


class ModelInfoResponse(ApiBaseModel):
    best_regression_model: Optional[str] = None
    best_classification_model: Optional[str] = None
    regression_metrics: Dict[str, Any] = Field(default_factory=dict)
    classification_metrics: Dict[str, Any] = Field(default_factory=dict)
    target_definition: Dict[str, Any] = Field(default_factory=dict)
    features_used: List[str] = Field(default_factory=list)
    high_value_threshold_value: Optional[float] = None


class BusinessSummaryResponse(ApiBaseModel):
    total_customers: int
    total_predicted_clv: float
    average_predicted_clv: float
    average_clv_before_prediction: Optional[float] = None
    high_value_percentage: float
    profitable_percentage: float
    high_value_customers: Optional[int] = None
    top_state_by_clv: Optional[str] = None
