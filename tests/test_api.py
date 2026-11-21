"""Integration tests for FastAPI REST API endpoints."""

import io
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.app import app
from app.api.session import session_manager
from app.models.schemas import (
    CatalogProfile,
    ColumnProfile,
    DatasetMetadata,
    DatasetProfile,
    QueryResult,
    RelationshipSuggestion,
    ValidationResult,
)


@pytest.fixture
def client():
    session_manager.clear_all()
    with TestClient(app) as test_client:
        yield test_client
    session_manager.clear_all()


def test_health_endpoint(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "provider" in data
    assert "model" in data
    assert "max_query_result_rows" in data


def test_session_lifecycle(client: TestClient):
    # Create session
    create_res = client.post("/api/sessions")
    assert create_res.status_code == 200
    data = create_res.json()
    session_id = data["session_id"]
    assert session_id

    # Get session
    get_res = client.get(f"/api/sessions/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == session_id

    # Delete session
    del_res = client.delete(f"/api/sessions/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Get after delete should 404
    not_found_res = client.get(f"/api/sessions/{session_id}")
    assert not_found_res.status_code == 404


def test_load_sample_dataset(client: TestClient):
    # Test single-file sample load
    single_res = client.post("/api/load-sample", json={"sample_key": "ecommerce_single"})
    assert single_res.status_code == 200
    single_data = single_res.json()
    assert len(single_data["datasets"]) == 1
    assert single_data["has_agent"] is True

    # Test multi-file sample load
    multi_res = client.post("/api/load-sample", json={"sample_key": "multi_retail"})
    assert multi_res.status_code == 200
    multi_data = multi_res.json()
    assert len(multi_data["datasets"]) == 2
    assert len(multi_data["relationships"]) == 1
    rel = multi_data["relationships"][0]
    assert rel["status"] == "suggested"


def test_relationship_status_update(client: TestClient):
    res = client.post("/api/load-sample", json={"sample_key": "multi_retail"})
    data = res.json()
    session_id = data["session_id"]
    rel = data["relationships"][0]

    # Confirm relationship
    confirm_res = client.post(
        "/api/relationships/status",
        json={
            "session_id": session_id,
            "left_table": rel["left_table"],
            "left_column": rel["left_column"],
            "right_table": rel["right_table"],
            "right_column": rel["right_column"],
            "status": "confirmed",
        },
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["relationships"][0]["status"] == "confirmed"

    # Reject relationship
    reject_res = client.post(
        "/api/relationships/status",
        json={
            "session_id": session_id,
            "left_table": rel["left_table"],
            "left_column": rel["left_column"],
            "right_table": rel["right_table"],
            "right_column": rel["right_column"],
            "status": "rejected",
        },
    )
    assert reject_res.status_code == 200
    assert reject_res.json()["relationships"][0]["status"] == "rejected"


def test_csv_upload_endpoint(client: TestClient):
    csv_content = b"id,name,value\n1,Alpha,100\n2,Beta,200\n3,Gamma,300\n"
    files = [
        ("files", ("test_upload.csv", io.BytesIO(csv_content), "text/csv"))
    ]

    response = client.post(
        "/api/upload?session_id=test_upload_sess",
        files=files,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_upload_sess"
    assert len(data["datasets"]) == 1
    assert data["has_agent"] is True
    assert data["datasets"][0]["row_count"] == 3


def test_analyze_without_dataset_fails(client: TestClient):
    response = client.post(
        "/api/analyze",
        json={"session_id": "nonexistent_sess", "question": "Show me total count"},
    )
    assert response.status_code == 400


def test_suggested_questions(client: TestClient):
    # Without session
    empty_res = client.get("/api/suggested-questions/none")
    assert empty_res.status_code == 200
    assert len(empty_res.json()) > 0

    # With multi-dataset loaded
    load_res = client.post("/api/load-sample", json={"sample_key": "multi_retail"})
    sess_id = load_res.json()["session_id"]
    questions_res = client.get(f"/api/suggested-questions/{sess_id}")
    assert questions_res.status_code == 200
    questions = questions_res.json()
    assert len(questions) > 0


def test_custom_chart_generation(client: TestClient):
    res = client.post("/api/load-sample", json={"sample_key": "ecommerce_single"})
    sess_id = res.json()["session_id"]

    chart_res = client.post(
        "/api/custom-chart",
        json={
            "session_id": sess_id,
            "chart_type": "bar",
            "x_col": "Category",
            "y_col": "Purchase Amount (USD)",
            "agg_func": "Sum",
        },
    )
    assert chart_res.status_code == 200
    assert chart_res.json()["success"] is True
    assert "plotly_spec" in chart_res.json()
