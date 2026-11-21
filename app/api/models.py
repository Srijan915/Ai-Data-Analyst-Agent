"""Pydantic request and response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.models.schemas import CatalogProfile, DatasetMetadata, RelationshipSuggestion


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    datasets: List[DatasetMetadata] = Field(default_factory=list)
    relationships: List[RelationshipSuggestion] = Field(default_factory=list)
    has_agent: bool = False
    conversation_history_length: int = 0


class RelationshipUpdateRequest(BaseModel):
    session_id: str
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    status: Literal["confirmed", "rejected"]


class AnalysisRequest(BaseModel):
    session_id: str
    question: str
    chart_type: Optional[str] = None


class CustomChartRequest(BaseModel):
    session_id: str
    dataset_id: Optional[str] = None
    chart_type: str
    x_col: str
    y_col: Optional[str] = None
    color_col: Optional[str] = None
    agg_func: Optional[str] = "Sum"


class SampleLoadRequest(BaseModel):
    session_id: Optional[str] = None
    sample_key: Literal["ecommerce_single", "multi_retail", "item_catalog"]


class AnalysisResponse(BaseModel):
    session_id: str
    question: str
    sql: str
    validation: Dict[str, Any]
    query_result: Dict[str, Any]
    explanation: Optional[str] = None
    chart: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    grounded_verified: bool = True
