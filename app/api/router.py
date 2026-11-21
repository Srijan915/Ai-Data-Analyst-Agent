"""FastAPI REST routes for QueryPilotAI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
import numpy as np
import pandas as pd

from app.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    CustomChartRequest,
    RelationshipUpdateRequest,
    SampleLoadRequest,
    SessionInfo,
)
from app.api.session import session_manager
from app.config import config
from app.core.agent import (
    AgentCapacityError,
    AgentDataError,
    AgentExecutionError,
    AgentPlanningError,
    AgentValidationError,
    DataAnalystAgent,
)
from app.core.chart_generator import build_custom_dataset_figure
from app.core.data_loader import (
    CSVEncodingError,
    CSVParsingError,
    DataLoaderError,
    EmptyFileError,
    FileTooLargeError,
)
from app.core.dataset_catalog import DatasetCatalog
from app.core.llm_client import (
    LLMAccessDeniedError,
    LLMAPIError,
    LLMConfigurationError,
    LLMCredentialsError,
    LLMNetworkError,
    LLMTLSVerificationError,
    LLMThrottlingError,
    LLMTimeoutError,
)
from app.core.query_planner import MAX_CONVERSATION_TURNS, QueryInputError
from app.utils.dataset_resources import DatasetResourceError
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _planning_error_message(error: Any) -> str:
    messages = {
        LLMCredentialsError: "The AI service credentials are missing or invalid. Check server environment settings.",
        LLMAccessDeniedError: "The configured AI model is unavailable for this account. Check model access permissions.",
        LLMThrottlingError: "The AI request quota or hourly service budget has been reached. Please try again later.",
        LLMConfigurationError: "The configured AI model or request is unavailable. Check model configuration.",
        LLMTLSVerificationError: "The secure connection to the AI service could not be verified. Check TLS certificate settings.",
        LLMTimeoutError: "The AI service request timed out. Check network connection and retry.",
        LLMNetworkError: "The AI service could not be reached. Check network connection and retry.",
        LLMAPIError: "The AI service could not be reached or rejected the request. Please try again shortly.",
    }
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for kind, message in messages.items():
            if isinstance(error, kind):
                return message
        error = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    return "The AI model could not generate a valid analysis plan."


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe rows to JSON-safe dictionaries."""
    sanitized = df.copy()
    for col in sanitized.columns:
        if pd.api.types.is_datetime64_any_dtype(sanitized[col]):
            sanitized[col] = sanitized[col].astype(str)
        elif pd.api.types.is_numeric_dtype(sanitized[col]):
            # Replace NaNs/Infs with None for clean JSON serialization
            sanitized[col] = sanitized[col].replace([np.inf, -np.inf, np.nan], None)
        else:
            sanitized[col] = sanitized[col].where(pd.notnull(sanitized[col]), None)
    return sanitized.to_dict(orient="records")


@router.get("/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "provider": config.llm_provider,
        "model": config.llm_model_id,
        "max_query_result_rows": config.max_query_result_rows,
        "duckdb_memory_limit": config.duckdb_memory_limit,
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
    }


@router.post("/sessions", response_model=SessionInfo)
def create_session(session_id: Optional[str] = None) -> SessionInfo:
    session = session_manager.get_or_create(session_id)
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        datasets=session.get_datasets(),
        relationships=session.get_relationships(),
        has_agent=session.agent is not None,
        conversation_history_length=len(session.conversation_history),
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str) -> SessionInfo:
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        datasets=session.get_datasets(),
        relationships=session.get_relationships(),
        has_agent=session.agent is not None,
        conversation_history_length=len(session.conversation_history),
    )


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, bool]:
    session_manager.remove(session_id)
    return {"success": True}


@router.post("/upload", response_model=SessionInfo)
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
) -> SessionInfo:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session = session_manager.get_or_create(session_id)
    # Release previous session state
    session.release()

    catalog = DatasetCatalog()
    records = []
    try:
        for file in files:
            record = catalog.add(file.file, file.filename)
            records.append(record)

        if not records:
            catalog.close()
            raise HTTPException(status_code=400, detail="Select at least one valid CSV file.")

        first = records[0]
        dataframe = pd.DataFrame(first.profile.sample_rows, columns=first.profile.column_names)
        agent = DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=catalog.profile,
            max_result_rows=config.max_query_result_rows,
            dataset_catalog=catalog,
        )

        session.catalog = catalog
        session.dataframe = dataframe
        session.dataset_profile = catalog.profile
        session.agent = agent

        return SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            datasets=session.get_datasets(),
            relationships=session.get_relationships(),
            has_agent=True,
            conversation_history_length=0,
        )

    except (DatasetResourceError, DataLoaderError, CSVEncodingError, CSVParsingError, EmptyFileError, FileTooLargeError) as exc:
        catalog.close()
        raise HTTPException(status_code=400, detail=f"Dataset error: {str(exc)}")
    except Exception as exc:
        catalog.close()
        logger.exception("Unexpected error while preparing uploaded datasets")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while preparing datasets.")


@router.post("/load-sample", response_model=SessionInfo)
def load_sample_dataset(req: SampleLoadRequest) -> SessionInfo:
    session = session_manager.get_or_create(req.session_id)
    session.release()

    sample_dir = PROJECT_ROOT / "data" / "sample"
    file_paths: List[Path] = []

    if req.sample_key == "ecommerce_single":
        file_paths = [sample_dir / "customer_shopping_behavior.csv"]
    elif req.sample_key == "item_catalog":
        file_paths = [sample_dir / "item_catalog.csv"]
    elif req.sample_key == "multi_retail":
        file_paths = [
            sample_dir / "customer_shopping_behavior.csv",
            sample_dir / "item_catalog.csv",
        ]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sample key: {req.sample_key}")

    catalog = DatasetCatalog()
    records = []
    try:
        for path in file_paths:
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Sample file not found: {path.name}")
            with open(path, "rb") as f:
                record = catalog.add(f, path.name)
                records.append(record)

        first = records[0]
        dataframe = pd.DataFrame(first.profile.sample_rows, columns=first.profile.column_names)
        agent = DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=catalog.profile,
            max_result_rows=config.max_query_result_rows,
            dataset_catalog=catalog,
        )

        session.catalog = catalog
        session.dataframe = dataframe
        session.dataset_profile = catalog.profile
        session.agent = agent

        return SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            datasets=session.get_datasets(),
            relationships=session.get_relationships(),
            has_agent=True,
            conversation_history_length=0,
        )
    except Exception as exc:
        catalog.close()
        logger.exception("Failed to load sample dataset: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(exc)}")


@router.post("/relationships/status")
def update_relationship_status(req: RelationshipUpdateRequest) -> Dict[str, Any]:
    session = session_manager.get(req.session_id)
    if not session or not session.catalog:
        raise HTTPException(status_code=404, detail="Active dataset catalog not found.")

    try:
        session.catalog.set_relationship_status(
            req.left_table,
            req.left_column,
            req.right_table,
            req.right_column,
            req.status,
        )
        return {
            "success": True,
            "relationships": [r.model_dump() for r in session.catalog.profile.relationships],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error updating relationship status")
        raise HTTPException(status_code=500, detail="Failed to update relationship status.")


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_query(req: AnalysisRequest) -> AnalysisResponse:
    session = session_manager.get(req.session_id)
    if not session or not session.agent:
        raise HTTPException(
            status_code=400,
            detail="Please upload one or more CSV files before asking questions.",
        )

    started_at = perf_counter()

    try:
        agent_result = session.agent.run(
            question=req.question,
            generate_explanation=True,
            generate_chart_spec=True,
            chart_type=req.chart_type,
            conversation_context=session.conversation_history,
        )
    except AgentDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AgentCapacityError:
        raise HTTPException(
            status_code=503,
            detail="The service is currently at capacity processing another analysis. Please retry in a moment.",
        )
    except AgentPlanningError as exc:
        user_msg = _planning_error_message(exc)
        logger.exception("Planning error: %s", exc)
        raise HTTPException(status_code=502, detail=user_msg)
    except AgentValidationError as exc:
        logger.warning("Security validation rejected generated SQL: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Security check rejected the query: {str(exc)}",
        )
    except AgentExecutionError as exc:
        logger.exception("SQL execution error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Query execution error: {str(exc)}",
        )
    except QueryInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail=f"Unexpected analysis failure: {str(exc)}")

    execution_time_ms = round((perf_counter() - started_at) * 1000, 2)

    # Process query result dataframe
    df = agent_result.query_result.dataframe
    columns = [str(c) for c in df.columns]
    rows = _dataframe_to_records(df)

    # Process chart specification
    chart_payload = None
    if agent_result.chart:
        chart_dict = dict(agent_result.chart)
        fig = chart_dict.get("figure")
        if fig is not None and hasattr(fig, "to_plotly_json"):
            chart_dict["plotly_spec"] = json.loads(fig.to_json())
            # Remove non-serializable figure object
            del chart_dict["figure"]
        chart_payload = chart_dict

    # Append to conversation history (bounded to MAX_CONVERSATION_TURNS)
    session.conversation_history.append({"role": "user", "content": req.question})
    if agent_result.explanation:
        session.conversation_history.append({"role": "assistant", "content": agent_result.explanation})

    if len(session.conversation_history) > MAX_CONVERSATION_TURNS * 2:
        session.conversation_history = session.conversation_history[-MAX_CONVERSATION_TURNS * 2:]

    response_data = AnalysisResponse(
        session_id=session.session_id,
        question=agent_result.question,
        sql=agent_result.sql,
        validation=agent_result.validation.model_dump(),
        query_result={
            "columns": columns,
            "rows": rows,
            "row_count": agent_result.query_result.row_count,
            "truncated": agent_result.query_result.truncated,
        },
        explanation=agent_result.explanation,
        chart=chart_payload,
        warnings=agent_result.warnings,
        execution_time_ms=execution_time_ms,
        grounded_verified=True,
    )

    session.last_result = response_data.model_dump()
    return response_data


@router.post("/custom-chart")
def generate_custom_chart(req: CustomChartRequest) -> Dict[str, Any]:
    session = session_manager.get(req.session_id)
    if not session or not session.catalog:
        raise HTTPException(status_code=400, detail="Active dataset not found.")

    target_df = None
    if req.dataset_id:
        record = session.catalog._records.get(req.dataset_id)
        if record:
            conn = session.catalog.path
            import duckdb
            c = duckdb.connect(str(conn), read_only=True)
            try:
                target_df = c.execute(f'SELECT * FROM "{record.table_name}" LIMIT 5000').df()
            finally:
                c.close()

    if target_df is None and session.dataframe is not None:
        target_df = session.dataframe

    if target_df is None:
        raise HTTPException(status_code=400, detail="No data available for charting.")

    try:
        custom_res = build_custom_dataset_figure(
            df=target_df,
            chart_type=req.chart_type,
            x_col=req.x_col,
            y_col=req.y_col,
            color_col=req.color_col,
            agg_func=req.agg_func or "Sum",
        )
        if custom_res is None:
            raise HTTPException(status_code=400, detail="Could not generate figure for chosen columns.")

        fig, metrics = custom_res if isinstance(custom_res, tuple) else (custom_res, [])
        if fig is None:
            raise HTTPException(status_code=400, detail="Could not generate figure for chosen columns.")

        return {
            "success": True,
            "plotly_spec": json.loads(fig.to_json()),
            "chart_type": req.chart_type,
            "metrics": metrics,
        }
    except Exception as exc:
        logger.exception("Custom chart generation failed")
        raise HTTPException(status_code=400, detail=f"Custom chart failed: {str(exc)}")


@router.get("/suggested-questions/{session_id}")
def get_suggested_questions(session_id: str) -> List[str]:
    session = session_manager.get(session_id)
    if not session or not session.catalog:
        return [
            "What are the top 5 product categories by revenue?",
            "What is the average purchase amount across locations?",
            "How does customer review rating vary with subscription status?",
            "What are the most frequent shipping types used?",
        ]

    datasets = session.get_datasets()
    relationships = [r for r in session.get_relationships() if r.status == "confirmed"]

    suggestions = []
    if relationships:
        suggestions.append(
            f"Join {datasets[0].filename} and {datasets[1].filename} to analyze sales performance by category."
        )
        suggestions.append(
            "What are the top 10 items purchased and their corresponding categories?"
        )

    # Dataset-specific suggestions
    col_names = []
    for d in datasets:
        col_names.extend([c.name.lower() for c in d.profile.columns])

    if "purchase amount (usd)" in col_names or "sales" in col_names:
        suggestions.append("Which customer segment or location generated the highest total purchase amount?")
    if "category" in col_names or "item category" in col_names:
        suggestions.append("What is the revenue breakdown across different categories?")
    if "review rating" in col_names:
        suggestions.append("What is the average review rating grouped by category and gender?")

    if not suggestions:
        suggestions = [
            "Show a high-level summary of all records.",
            "What are the top categories by total count?",
            "Find the average values of the numeric columns.",
        ]

    return suggestions[:5]
