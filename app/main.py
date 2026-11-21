"""
QueryPilotAI Streamlit application entry point.

Application pipeline:

    CSV Upload
        ↓
    Data Loading
        ↓
    Dataset Profiling
        ↓
    DataAnalystAgent
        ↓
    Query Planning
        ↓
    SQL Security Validation
        ↓
    DuckDB Execution
        ↓
    Chart Generation
        ↓
    LLM Explanation

The Streamlit layer is responsible for:

- user interaction
- displaying application state
- passing data into the core pipeline
- rendering results

Business logic remains inside app.core.
"""

import hmac
import html
from io import BytesIO
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# Ensure project root is importable
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# Plotly charting
# --------------------------------------------------------------------------

import plotly.express as px
# --------------------------------------------------------------------------
# Application imports
# --------------------------------------------------------------------------

from app.ui.analysis import _run_analysis, display_analysis_result
from app.ui.theme import apply_theme, style_figure as _style_plotly_figure, CHART_COLORS as PLOTLY_COLORWAY
from app.file_handlers.csv_handler import delimiter_for, load_csv_bytes
from app.file_handlers.detect_format import detect_format, read_upload
from app.file_handlers.excel_handler import load_excel_bytes, sheet_names
from app.file_handlers.pdf_handler import load_pdf_bytes
from app.file_handlers.txt_handler import load_txt_bytes
from app.config import config
from app.core.agent import (
    AgentCapacityError,
    AgentExecutionError,
    AgentPlanningError,
    AgentValidationError,
    DataAnalystAgent,
)
from app.core.data_loader import (
    CSVEncodingError,
    CSVParsingError,
    DataLoaderError,
    EmptyFileError,
    FileTooLargeError,
    load_csv,
)
from app.core.data_profiler import profile_dataframe
from app.core.chart_generator import (
    classify_dataset_columns,
    generate_dataset_auto_visualizations,
    build_custom_dataset_figure,
)
from app.core.query_planner import MAX_CONVERSATION_TURNS
from app.utils.logger import get_logger
from app.utils.dataset_resources import DatasetResourceError, dataset_preparation_slot
from app.preprocessing.ui import render_preprocessing


logger = get_logger(__name__)

ACCESS_PASSWORD_ENVIRONMENT_VARIABLE = "APP_ACCESS_PASSWORD"


# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=config.app_name,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

apply_theme()


def _get_suggested_questions(profile) -> list[str]:
    """Generate contextual example questions from the active dataset profile."""
    if profile is None or not getattr(profile, "columns", None):
        return [
            "What are the top 10 records by value?",
            "Summarize the key metrics across this dataset",
            "Show distribution and breakdown by category",
        ]

    # Identifiers are poor default measures or groupings. Prefer additive
    # business measures when present, then offer concrete schema-based prompts.
    cols = [c for c in profile.columns if c.name.lower().replace('_', ' ').split()[-1] not in {'id', 'identifier'}]
    numeric_candidates = [
        c.name for c in cols
        if str(c.inferred_type).lower() in ("integer", "float", "numeric", "decimal")
        or any(k in c.name.lower() for k in ("sales", "revenue", "price", "amount", "total", "cost", "score", "count", "profit", "qty", "quantity"))
    ]
    cat_candidates = [
        c.name for c in cols
        if c.name not in numeric_candidates
        and (str(c.inferred_type).lower() in ("string", "category", "text") or c.unique_count <= 250)
    ]
    numeric_candidates.sort(key=lambda name: not any(word in name.lower() for word in ("sales", "revenue", "amount", "cost", "quantity")))
    cat_candidates.sort(key=lambda name: "category" not in name.lower())

    suggestions = []
    if numeric_candidates and cat_candidates:
        suggestions.append(f"Top 5 {cat_candidates[0]} by total {numeric_candidates[0]}")
        if len(numeric_candidates) > 1:
            suggestions.append(f"Average {numeric_candidates[0]} by {cat_candidates[0]}")
        elif len(cat_candidates) > 1:
            suggestions.append(f"Average {numeric_candidates[0]} grouped by {cat_candidates[1]}")
        else:
            suggestions.append(f"Total {numeric_candidates[0]} breakdown by {cat_candidates[0]}")
    elif numeric_candidates:
        suggestions.append(f"Top 10 records with highest {numeric_candidates[0]}")
        suggestions.append(f"Total and average {numeric_candidates[0]}")
    elif cat_candidates:
        suggestions.append(f"Most frequent {cat_candidates[0]} values")
        suggestions.append(f"Count of records grouped by {cat_candidates[0]}")

    fallbacks = [
        "How many records are in this dataset?",
        "Show the first 10 records",
        "Which columns have missing values?",
    ]
    for fb in fallbacks:
        if len(suggestions) < 3 and fb not in suggestions:
            suggestions.append(fb)

    return suggestions[:3]


def _apply_suggested_question(query: str) -> None:
    """Stage a suggested question to be loaded into query_input_composer on rerun."""
    st.session_state.pending_query = query


def _workspace_status() -> tuple[str, str, str]:
    """Return sidebar status copy, CSS modifier, and header chip class."""

    if st.session_state.get("upload_error"):
        return "Paused", "is-paused", "status-chip--paused"
    if st.session_state.get("dataframe") is not None or st.session_state.get("documents"):
        return "Ready", "is-ready", "status-chip--ready"
    return "No dataset", "", "status-chip--empty"


def _initialize_session_state() -> None:
    """Initialize Streamlit session-state values."""

    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None
    if "dataset_catalog" not in st.session_state:
        st.session_state.dataset_catalog = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "pdf_page_errors" not in st.session_state:
        st.session_state.pdf_page_errors = []
    if "pdf_notices" not in st.session_state:
        st.session_state.pdf_notices = []
    if "show_preprocess" not in st.session_state:
        st.session_state.show_preprocess = False
    if "preprocess_report" not in st.session_state:
        st.session_state.preprocess_report = None

    if "upload_error" not in st.session_state:
        st.session_state.upload_error = False
    if "failed_upload_signature" not in st.session_state:
        st.session_state.failed_upload_signature = None

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None

    if "uploaded_file_signature" not in st.session_state:
        st.session_state.uploaded_file_signature = None

    if "uploader_key_version" not in st.session_state:
        st.session_state.uploader_key_version = 0

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "demo_access_granted" not in st.session_state:
        st.session_state.demo_access_granted = False


def _render_public_demo_access_gate() -> bool:
    """Return whether this session may use an optionally protected demo."""

    expected_password = os.getenv(ACCESS_PASSWORD_ENVIRONMENT_VARIABLE, "")

    # Local development remains frictionless until a deployment explicitly
    # supplies a non-empty secret through its environment or secret manager.
    if not expected_password.strip():
        if config.is_production:
            st.error("The demo is locked until APP_ACCESS_PASSWORD is configured.")
            return False
        return True

    if st.session_state.demo_access_granted:
        return True

    st.html(
        """
        <div class="access-card">
            <div class="access-icon">🔒</div>
            <div class="hero-eyebrow">PROTECTED WORKSPACE</div>
        </div>
        """
    )
    st.title("Private Portfolio Demo")
    st.caption(
        "This demo is access-controlled to protect the AI service quota "
        "used for analysis."
    )

    access_col, _ = st.columns([1, 1])
    with access_col:
        access_password = st.text_input(
            "Demo access code",
            type="password",
            key="demo_access_password",
        )

        if st.button(
            "Unlock demo",
            type="primary",
            use_container_width=True,
        ):
            if hmac.compare_digest(access_password, expected_password):
                st.session_state.demo_access_granted = True
                st.session_state.pop("demo_access_password", None)
                logger.info("Portfolio demo access granted")
                st.rerun()
            else:
                logger.warning("Portfolio demo access denied")
                st.error("The access code is incorrect. Please try again.")

    return False


def _uploaded_file_signature(uploaded_file) -> tuple:
    """Return Streamlit upload metadata that changes for a new upload."""
    return (
        getattr(uploaded_file, "file_id", None),
        getattr(uploaded_file, "name", None),
        getattr(uploaded_file, "size", None),
    )


def _clear_dataset() -> None:
    """Clear all analysis state and force Streamlit to create a fresh uploader."""
    if st.session_state.agent is not None:
        st.session_state.agent.release_dataset()
    st.session_state.upload_error = False
    st.session_state.failed_upload_signature = None
    st.session_state.dataframe = None
    st.session_state.dataset_profile = None
    st.session_state.dataset_catalog = None
    st.session_state.documents = []
    st.session_state.pdf_page_errors = []
    st.session_state.pdf_notices = []
    st.session_state.show_preprocess = False
    st.session_state.preprocess_report = None
    st.session_state.agent = None
    st.session_state.last_result = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_file_signature = None
    st.session_state.conversation_history = []
    st.session_state.uploader_key_version += 1
    st.session_state.pop("query_input_composer", None)
    st.session_state.pending_query = None

    logger.info("Dataset cleared from Streamlit session")


def _clear_conversation_history() -> None:
    """Clear session-local follow-up context without clearing the dataset."""

    st.session_state.conversation_history = []
    logger.info("Conversation context cleared from Streamlit session")


def _load_uploaded_file(
    uploaded_file,
    upload_signature: tuple,
) -> None:
    """
    Load and profile an uploaded CSV file.

    Session state is updated only after the complete pipeline succeeds.
    """

    st.session_state.upload_error = True
    st.session_state.failed_upload_signature = upload_signature
    st.session_state.last_result = None
    st.session_state.conversation_history = []
    try:
        with dataset_preparation_slot():
            # Small uploads retain the original DataFrame workflow.
            # Larger uploads avoid materializing the full CSV in pandas.
            disk_dataset = None
            if getattr(uploaded_file, "size", 0) > 1024**2:
                from app.core.disk_dataset import DiskDataset
                status = st.empty()
                try:
                    disk_dataset = DiskDataset(uploaded_file, progress=status.caption)
                    dataframe = disk_dataset.preview
                    dataset_profile = disk_dataset.raw_profile
                finally:
                    status.empty()
            else:
                try:
                    dataframe = load_csv(uploaded_file)
                    dataset_profile = profile_dataframe(dataframe)
                except DatasetResourceError:
                    # A compact CSV can have too many rows for the pandas path.
                    from app.core.disk_dataset import DiskDataset
                    status = st.empty()
                    try:
                        disk_dataset = DiskDataset(uploaded_file, progress=status.caption)
                        dataframe = disk_dataset.preview
                        dataset_profile = disk_dataset.raw_profile
                    finally:
                        status.empty()
            try:
                agent = DataAnalystAgent(
                    dataframe=dataframe,
                    dataset_profile=dataset_profile,
                    max_result_rows=config.max_query_result_rows,
                    **({"disk_dataset": disk_dataset} if disk_dataset else {}),
                )
            except Exception:
                if disk_dataset is not None:
                    disk_dataset.close()
                raise

    except DatasetResourceError as exc:
        st.error(str(exc))
        return

    except FileTooLargeError as exc:
        st.error(f"📦 File is too large: {exc}")
        return

    except EmptyFileError as exc:
        st.error(f"📭 Empty file: {exc}")
        return

    except CSVEncodingError as exc:
        st.error(f"🔤 CSV encoding error: {exc}")
        return

    except CSVParsingError as exc:
        st.error(f"📄 CSV parsing error: {exc}")
        return

    except DataLoaderError as exc:
        st.error(f"❌ Could not load the CSV file: {exc}")
        return

    except Exception as exc:
        logger.exception(
            "Unexpected error while preparing uploaded dataset"
        )

        st.error(
            "❌ An unexpected error occurred while preparing the dataset."
        )
        return

    previous_agent = st.session_state.agent
    if previous_agent is not None:
        previous_agent.release_dataset()
    st.session_state.upload_error = False
    st.session_state.failed_upload_signature = None
    st.session_state.dataframe = dataframe
    st.session_state.dataset_profile = dataset_profile
    # A single-file workspace must not retain controls or table counts from
    # a previously active catalog. The retired agent above owns and closes
    # that catalog before this reference is cleared.
    st.session_state.dataset_catalog = None
    st.session_state.show_preprocess = False
    st.session_state.preprocess_report = None
    st.session_state.documents = []
    st.session_state.pdf_page_errors = []
    st.session_state.pdf_notices = []
    st.session_state.agent = agent
    st.session_state.last_result = None
    st.session_state.conversation_history = []

    st.session_state.uploaded_filename = getattr(
        uploaded_file,
        "name",
        "uploaded.csv",
    )
    st.session_state.uploaded_file_signature = upload_signature

    logger.info(
        "Dataset loaded into Streamlit session | filename=%s | "
        "rows=%d | columns=%d",
        st.session_state.uploaded_filename,
        len(dataframe),
        len(dataframe.columns),
    )

    st.success(
        f"✅ Successfully loaded "
        f"**{st.session_state.uploaded_filename}** "
        f"({dataset_profile.row_count:,} rows × {dataset_profile.column_count} columns)."
    )


def _load_multiple_files(uploaded_files, upload_signature: tuple) -> None:
    """Atomically replace the session catalog after all CSVs validate."""
    from app.core.dataset_catalog import DatasetCatalog

    st.session_state.upload_error = True
    st.session_state.failed_upload_signature = upload_signature
    st.session_state.last_result = None
    st.session_state.conversation_history = []
    catalog = DatasetCatalog()
    try:
        status = st.empty()
        try:
            records = [catalog.add(item, getattr(item, "name", "upload.csv"), progress=status.caption) for item in uploaded_files]
        finally:
            status.empty()
        if not records:
            raise DataLoaderError("Select at least one CSV file.")
        first = records[0]
        dataframe = pd.DataFrame(first.profile.sample_rows, columns=first.profile.column_names)
        agent = DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=catalog.profile,
            max_result_rows=config.max_query_result_rows,
            dataset_catalog=catalog,
        )
    except (DatasetResourceError, DataLoaderError, CSVEncodingError, CSVParsingError, EmptyFileError, FileTooLargeError) as exc:
        catalog.close()
        st.error(f"❌ Could not load the selected CSV files: {exc}")
        return
    except Exception:
        catalog.close()
        logger.exception("Unexpected error while preparing dataset catalog")
        st.error("❌ An unexpected error occurred while preparing the selected datasets.")
        return
    previous_agent = st.session_state.agent
    if previous_agent is not None:
        previous_agent.release_dataset()
    st.session_state.upload_error = False
    st.session_state.failed_upload_signature = None
    st.session_state.dataframe = dataframe
    st.session_state.dataset_profile = first.profile
    st.session_state.dataset_catalog = catalog
    st.session_state.show_preprocess = False
    st.session_state.preprocess_report = None
    st.session_state.documents = []
    st.session_state.pdf_page_errors = []
    st.session_state.pdf_notices = []
    st.session_state.agent = agent
    st.session_state.last_result = None
    st.session_state.conversation_history = []
    st.session_state.uploaded_filename = f"{len(records)} datasets"
    st.session_state.uploaded_file_signature = upload_signature
    st.success(f"✅ Loaded {len(records)} isolated CSV datasets. Confirm a suggested relationship before asking for a join.")


def _inspect_format_files(uploaded_files):
    inspected = []
    for index, upload in enumerate(uploaded_files):
        name = getattr(upload, "name", "uploaded file")
        data = read_upload(upload)
        kind, warning = detect_format(name, data)
        if warning:
            st.warning(warning)
        sheets = []
        if kind == "xlsx":
            available = sheet_names(data)
            sheets = st.multiselect(
                f"Sheets from {name}", available, default=available,
                key=f"excel_sheets_{index}_{getattr(upload, 'file_id', name)}",
            )
        inspected.append((name, data, kind, sheets))
    return inspected


def _uses_original_csv_path(inspected):
    for _, data, kind, _ in inspected:
        if kind != "csv":
            return False
        try:
            if delimiter_for(data.decode("utf-8-sig")) != ",":
                return False
        except UnicodeDecodeError:
            return False
    return True


def _parse_format_files(inspected):
    parts = []
    for name, data, kind, sheets in inspected:
        if kind == "csv":
            parts.extend(load_csv_bytes(data, name))
        elif kind == "xlsx":
            parts.extend(load_excel_bytes(data, name, sheets))
        elif kind == "txt":
            parts.extend(load_txt_bytes(data, name))
        elif kind == "pdf":
            parts.extend(load_pdf_bytes(data, name))
    return parts


def _catalog_from_parts(tables):
    from app.core.dataset_catalog import DatasetCatalog
    catalog = DatasetCatalog()
    try:
        for part in tables:
            data = part["dataframe"].to_csv(index=False).encode("utf-8-sig")
            upload = BytesIO(data)
            upload.name = part["name"].replace("/", "_").replace("\\", "_") + ".csv"
            record = catalog.add(upload, upload.name)
            record.filename = part["name"]
        return catalog
    except Exception:
        catalog.close()
        raise


def _activate_format_files(catalog, documents, signature, failures, notices):
    records = catalog.profile.datasets if catalog is not None else []
    first = records[0] if records else None
    frame = pd.DataFrame(first.profile.sample_rows, columns=first.profile.column_names) if first else None
    agent = DataAnalystAgent(dataframe=frame, dataset_profile=catalog.profile,
                             max_result_rows=config.max_query_result_rows,
                             dataset_catalog=catalog) if catalog else None
    if st.session_state.agent is not None:
        st.session_state.agent.release_dataset()
    st.session_state.agent = agent
    st.session_state.dataset_catalog = catalog
    st.session_state.show_preprocess = False
    st.session_state.preprocess_report = None
    st.session_state.dataframe = frame
    st.session_state.dataset_profile = first.profile if first else None
    st.session_state.documents = documents
    st.session_state.pdf_page_errors = failures
    st.session_state.pdf_notices = notices
    st.session_state.uploaded_filename = f"{len(records)} tables, {len(documents)} documents"
    st.session_state.uploaded_file_signature = signature
    st.session_state.failed_upload_signature = None
    st.session_state.upload_error = False
    st.session_state.last_result = None
    st.session_state.conversation_history = []
    if records or documents:
        st.success(f"Loaded {len(records)} table(s) and {len(documents)} document(s).")


def _load_format_files(inspected, signature):
    st.session_state.upload_error = True
    st.session_state.failed_upload_signature = signature
    st.session_state.last_result = None
    st.session_state.conversation_history = []
    catalog = None
    try:
        with dataset_preparation_slot():
            parts = _parse_format_files(inspected)
            tables = [part for part in parts if "dataframe" in part]
            documents = [part for part in parts if "text" in part]
            failures = [failure for part in parts for failure in part.get("page_errors", [])]
            notices = [part["notice"] for part in parts if "notice" in part]
            if tables:
                catalog = _catalog_from_parts(tables)
            _activate_format_files(catalog, documents, signature, failures, notices)
    except (ValueError, DatasetResourceError, DataLoaderError) as exc:
        if catalog is not None:
            catalog.close()
        st.error(f"Could not load the selected files: {exc}")
    except Exception:
        if catalog is not None:
            catalog.close()
        logger.exception("Unexpected error while preparing uploaded files")
        st.error("Could not prepare the selected files. Check their contents and retry.")


def _render_document_previews():
    for notice in st.session_state.pdf_notices:
        st.warning(notice)
    if st.session_state.pdf_page_errors:
        st.warning("Pages that failed")
        for failure in st.session_state.pdf_page_errors:
            st.caption(f"{failure['name']} · page {failure['page']}: {failure['message']}")
    if not st.session_state.documents:
        return
    st.subheader("Document previews")
    for index, document in enumerate(st.session_state.documents):
        with st.expander(document["name"], expanded=True):
            st.text_area("Text preview", value=document["text"][:4000],
                         height=150, disabled=True, key=f"document_preview_{index}")
            if document.get("format") == "pdf":
                st.caption("PDF page text may repeat extracted table cells.")
            if len(document["text"]) > 4000:
                st.caption("Preview shows the first 4,000 characters; the full text remains in this session.")


def _refresh_catalog_session() -> None:
    """Update UI/agent views after a safe catalog mutation."""
    # Approval buttons rerun before the composer renders. Preserve the draft
    # across Streamlit's cleanup of widgets absent from that interrupted run.
    if st.session_state.get("query_input_composer") is not None:
        st.session_state.pending_query = st.session_state.query_input_composer
    catalog = st.session_state.dataset_catalog
    records = catalog.profile.datasets
    if not records:
        _clear_dataset()
        return
    report = st.session_state.get("preprocess_report")
    if report and report["processed_id"] not in {record.dataset_id for record in records}:
        st.session_state.preprocess_report = None
    first = records[0]
    dataframe = pd.DataFrame(first.profile.sample_rows, columns=first.profile.column_names)
    st.session_state.agent.refresh_catalog(dataframe)
    st.session_state.dataframe = dataframe
    st.session_state.dataset_profile = first.profile
    st.session_state.uploaded_filename = f"{len(records)} datasets"
    st.session_state.last_result = None
    st.session_state.conversation_history = []


def _render_catalog_controls() -> None:
    """Render per-dataset removal/replacement controls for the active session."""
    catalog = st.session_state.dataset_catalog
    if catalog is None:
        return
    records = catalog.profile.datasets
    st.caption("Active session datasets")
    st.dataframe(
        [
            {
                "File": record.filename,
                "Table": record.table_name,
                "Rows": record.row_count,
                "Columns": record.profile.column_count,
            }
            for record in records
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected_id = st.selectbox(
        "Manage dataset",
        options=[record.dataset_id for record in records],
        format_func=lambda item: next(record.filename for record in records if record.dataset_id == item),
        key="catalog_selected_dataset",
    )
    selected = next(record for record in records if record.dataset_id == selected_id)
    st.caption(f"Previewing {selected.filename}. Analysis can query every registered dataset.")
    st.session_state.dataset_profile = selected.profile
    st.session_state.dataframe = pd.DataFrame(selected.profile.sample_rows, columns=selected.profile.column_names)
    with st.expander("Registered schemas", expanded=False):
        for record in records:
            st.markdown(f"**{record.filename}**")
            st.code(record.table_name, language="text")
            st.dataframe([
                {"Column": c.name, "Type": c.pandas_dtype, "Missing": c.null_count}
                for c in record.profile.columns
            ], hide_index=True, use_container_width=True)
            st.caption("Preview (first five rows)")
            st.dataframe(pd.DataFrame(record.profile.sample_rows, columns=record.profile.column_names),
                         hide_index=True, use_container_width=True)
    replacement = st.file_uploader(
        "Replacement CSV",
        type=["csv"],
        key=f"catalog_replace_{selected.dataset_id}",
    )
    first_col, second_col = st.columns(2)
    with first_col:
        if st.button("Remove selected", use_container_width=True):
            catalog.remove(selected_id)
            _refresh_catalog_session()
            st.rerun()
    with second_col:
        if st.button("Replace selected", disabled=replacement is None, use_container_width=True):
            try:
                status = st.empty()
                try:
                    catalog.replace(selected_id, replacement, getattr(replacement, "name", selected.filename), progress=status.caption)
                finally:
                    status.empty()
                _refresh_catalog_session()
                st.success("✅ Dataset replaced. Schema and relationship suggestions were refreshed.")
                st.rerun()
            except (DatasetResourceError, DataLoaderError, CSVEncodingError, CSVParsingError, EmptyFileError, FileTooLargeError) as exc:
                st.error(f"❌ Replacement was rejected; the existing dataset is unchanged: {exc}")
            except Exception:
                logger.exception("Catalog dataset replacement failed")
                st.error("❌ Replacement failed; the existing dataset is unchanged.")

    relationships = catalog.profile.relationships
    st.subheader("Relationships")
    if not relationships:
        st.info("No compatible shared keys were found. You can analyze each dataset separately; a join needs a real shared key.")
        return
    st.caption("Review the shared keys below. Confirmation checks for a complete unique key on at least one side; every generated join is checked again before execution.")
    names = {record.table_name: record.filename for record in records}
    for index, relation in enumerate(relationships):
        st.caption(
            f"{names[relation.left_table]} / {relation.left_column} ↔ "
            f"{names[relation.right_table]} / {relation.right_column} · "
            f"{relation.status.upper()} · {relation.overlap_percentage:.0f}% observed key overlap (up to 10,001 distinct values per table)"
        )
        confirm_col, reject_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm join", key=f"confirm_rel_{index}", disabled=relation.status == "confirmed", use_container_width=True):
                try:
                    catalog.set_relationship_status(
                        relation.left_table, relation.left_column,
                        relation.right_table, relation.right_column, "confirmed",
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    _refresh_catalog_session()
                    st.rerun()
        with reject_col:
            if st.button("Revoke join" if relation.status == "confirmed" else "Reject join", key=f"reject_rel_{index}", disabled=relation.status == "rejected", use_container_width=True):
                catalog.set_relationship_status(
                    relation.left_table, relation.left_column,
                    relation.right_table, relation.right_column, "rejected",
                )
                _refresh_catalog_session()
                st.rerun()


def _display_dataset_overview() -> None:
    """Display basic information about the loaded dataset."""

    dataframe = st.session_state.dataframe
    profile = st.session_state.dataset_profile

    if dataframe is None or profile is None:
        return

    st.html(
        """
        <div class="section-label">
            DATASET OVERVIEW
        </div>
        """
    )

    st.subheader("Dataset overview")
    if profile.statistics_sampled:
        st.caption(f"SQL analyzes all {profile.row_count:,} rows. Preview and unique/min/max/mean statistics use the first {profile.profiled_row_count:,} rows; missing-value counts cover the full dataset.")

    catalog = st.session_state.dataset_catalog
    table_count = len(catalog.profile.datasets) if catalog is not None else 1
    total_missing = sum(column.null_count for column in profile.columns)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{profile.row_count:,}")

    with col2:
        st.metric("Columns", f"{profile.column_count:,}")

    with col3:
        st.metric("Missing values", f"{total_missing:,}")

    with col4:
        st.metric("Active tables", f"{table_count:,}")

    preview_col, profile_col = st.columns(2)

    with preview_col:
        with st.expander("Preview data", expanded=False):
            st.dataframe(
                dataframe.head(10),
                use_container_width=True,
                hide_index=True,
            )

    with profile_col:
        with st.expander("Dataset profile", expanded=False):
            profile_data = [
                {
                    "Column": column.name,
                    "Pandas Type": column.pandas_dtype,
                    "Inferred Type": column.inferred_type,
                    "Null Count": column.null_count,
                    "Null %": round(column.null_percentage, 2),
                    "Unique Values": column.unique_count,
                }
                for column in profile.columns
            ]

            st.dataframe(
                profile_data,
                use_container_width=True,
                hide_index=True,
            )


def _display_explore_dataset() -> None:
    """
    Render the 'Explore Dataset' visualization section directly from the loaded dataset,
    independent of AI natural language queries.
    """
    dataframe = st.session_state.dataframe
    profile = st.session_state.dataset_profile

    if dataframe is None or dataframe.empty:
        return

    st.html(
        """
        <div class="section-label">
            EXPLORE DATASET
        </div>
        """
    )

    with st.container(border=True):
        st.html(
            """
            <div class="card-header-badge chart-badge">
                <span class="badge-icon">📈</span>
                <span>EXPLORE DATASET VISUALIZATIONS</span>
            </div>
            """
        )
        st.subheader("Explore Dataset")
        st.caption(
            "Visual exploration generated directly from your uploaded dataset, independent of AI query results."
        )
        if st.session_state.dataset_catalog is not None:
            st.caption(f"Overview charts query the selected table where supported. Custom charts use its {len(dataframe):,} preview rows; ask a question for complete-table analysis.")

        tab_auto, tab_custom = st.tabs(["Overview charts", "Build a chart"])

        with tab_auto:
            auto_charts = generate_dataset_auto_visualizations(
                dataframe,
                profile=profile,
                max_charts=4,
                agent=st.session_state.agent,
                table_name=(
                    next(record.table_name for record in st.session_state.dataset_catalog.profile.datasets
                         if record.dataset_id == st.session_state.catalog_selected_dataset)
                    if st.session_state.dataset_catalog is not None else None
                ),
            )
            if not auto_charts:
                st.caption("Fewer than 2 rows or no visualizable columns found in the dataset.")
            else:
                cols = st.columns(2, gap="medium")
                for idx, chart_info in enumerate(auto_charts):
                    with cols[idx % 2]:
                        with st.container(border=True):
                            st.caption(f"**{chart_info['title']}**")
                            st.caption(chart_info["description"])
                            if chart_info.get("sample_label"):
                                st.caption(f"*{chart_info['sample_label']}*")

                            metrics = chart_info.get("metrics")
                            if metrics:
                                m_cols = st.columns(len(metrics))
                                for m_idx, m in enumerate(metrics):
                                    m_cols[m_idx].metric(label=m["label"], value=m["value"])

                            st.plotly_chart(
                                _style_plotly_figure(chart_info["figure"]),
                                use_container_width=True,
                                key=f"auto_chart_{chart_info['id']}_{idx}",
                            )

        with tab_custom:
            classified = classify_dataset_columns(dataframe, profile=profile)
            all_cols = classified["all"]
            cat_cols = classified["categorical"]
            num_cols = classified["numeric"]
            date_cols = classified["datetime"]

            ctrl_row1 = st.columns([1.4, 1.2, 1.2, 1.1])
            with ctrl_row1[0]:
                custom_chart_type = st.selectbox(
                    "Chart Type",
                    options=[
                        "Bar Chart",
                        "Donut / Pie Chart",
                        "Line Chart",
                        "Histogram",
                        "Scatter Plot",
                        "Grouped Bar",
                    ],
                    key="explore_custom_chart_type",
                )

            x_col = None
            y_col = None
            color_col = None
            agg_func = "sum"
            max_cats = 12
            hist_bins = 25
            show_box = True
            orientation = "vertical"

            if custom_chart_type == "Bar Chart":
                with ctrl_row1[1]:
                    x_col = st.selectbox("Category Column (X)", options=cat_cols if cat_cols else all_cols, key="custom_bar_x")
                with ctrl_row1[2]:
                    y_options = ["(Count of Records)"] + num_cols
                    y_col = st.selectbox("Metric Column (Y)", options=y_options, key="custom_bar_y")
                with ctrl_row1[3]:
                    if y_col != "(Count of Records)":
                        agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Max", "Min"], key="custom_bar_agg")
                    else:
                        orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"], key="custom_bar_orient").lower()

                ctrl_row2 = st.columns([1.4, 1.2, 1.2, 1.1])
                with ctrl_row2[0]:
                    max_cats = st.slider("Max Categories", min_value=5, max_value=30, value=10, key="custom_bar_cats")
                with ctrl_row2[1]:
                    if y_col != "(Count of Records)":
                        orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"], key="custom_bar_orient_2").lower()

            elif custom_chart_type == "Donut / Pie Chart":
                with ctrl_row1[1]:
                    x_col = st.selectbox("Category / Slice", options=cat_cols if cat_cols else all_cols, key="custom_pie_x")
                with ctrl_row1[2]:
                    y_options = ["(Count of Records)"] + num_cols
                    y_col = st.selectbox("Values", options=y_options, key="custom_pie_y")
                with ctrl_row1[3]:
                    if y_col != "(Count of Records)":
                        agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Max", "Min"], key="custom_pie_agg")
                max_cats = st.slider("Max Slices", min_value=3, max_value=15, value=7, key="custom_pie_slices")

            elif custom_chart_type == "Line Chart":
                with ctrl_row1[1]:
                    x_col = st.selectbox("Date / Dimension (X)", options=date_cols if date_cols else all_cols, key="custom_line_x")
                with ctrl_row1[2]:
                    y_options = (num_cols if num_cols else []) + ["(Count of Records)"]
                    y_col = st.selectbox("Metric Column (Y)", options=y_options, key="custom_line_y")
                with ctrl_row1[3]:
                    if y_col != "(Count of Records)":
                        agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Max", "Min"], key="custom_line_agg")
                if cat_cols:
                    color_choice = st.selectbox("Group by Series (Optional)", options=["None"] + [c for c in cat_cols if c != x_col], key="custom_line_group")
                    color_col = None if color_choice == "None" else color_choice

            elif custom_chart_type == "Histogram":
                with ctrl_row1[1]:
                    x_col = st.selectbox("Numeric Column", options=num_cols if num_cols else all_cols, key="custom_hist_x")
                with ctrl_row1[2]:
                    hist_bins = st.slider("Bins", min_value=5, max_value=60, value=25, key="custom_hist_bins")
                with ctrl_row1[3]:
                    show_box = st.checkbox("Show Box Plot", value=True, key="custom_hist_box")

            elif custom_chart_type == "Scatter Plot":
                with ctrl_row1[1]:
                    x_col = st.selectbox("X-Axis (Numeric)", options=num_cols if num_cols else all_cols, key="custom_scatter_x")
                with ctrl_row1[2]:
                    y_options = [c for c in num_cols if c != x_col] if len(num_cols) > 1 else num_cols
                    y_col = st.selectbox("Y-Axis (Numeric)", options=y_options if y_options else all_cols, key="custom_scatter_y")
                with ctrl_row1[3]:
                    color_options = ["None"] + [c for c in cat_cols if c not in (x_col, y_col)]
                    color_choice = st.selectbox("Color Group (Optional)", options=color_options, key="custom_scatter_color")
                    color_col = None if color_choice == "None" else color_choice

            elif custom_chart_type == "Grouped Bar":
                with ctrl_row1[1]:
                    x_col = st.selectbox("Category (X)", options=cat_cols if cat_cols else all_cols, key="custom_grp_x")
                with ctrl_row1[2]:
                    selected_metrics = st.multiselect("Metrics (Y)", options=num_cols, default=num_cols[:2] if len(num_cols) >= 2 else num_cols, key="custom_grp_metrics")
                    y_col = ",".join(selected_metrics) if selected_metrics else None
                with ctrl_row1[3]:
                    agg_func = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Max", "Min"], key="custom_grp_agg")
                max_cats = st.slider("Max Categories", min_value=3, max_value=20, value=8, key="custom_grp_cats")

            if x_col:
                fig, cust_metrics = build_custom_dataset_figure(
                    df=dataframe,
                    chart_type=custom_chart_type,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=color_col,
                    agg_func=agg_func,
                    max_categories=max_cats,
                    bins=hist_bins,
                    show_box=show_box,
                    orientation=orientation,
                )
                if fig is not None:
                    if cust_metrics:
                        m_cols = st.columns(len(cust_metrics))
                        for m_idx, m in enumerate(cust_metrics):
                            m_cols[m_idx].metric(label=m["label"], value=m["value"])
                    st.plotly_chart(_style_plotly_figure(fig), use_container_width=True, key="explore_custom_plotly")
                else:
                    st.info("Select suitable columns above to render your visualization.")
            else:
                st.info("No suitable columns available for this chart type.")


def _display_chart(chart_spec, query_result=None) -> None:
    """
    Render a chart specification returned by the agent's Visualization
    Intelligence stage, or a subtle "not needed" notice when the engine
    decided a chart wouldn't meaningfully add anything for this result.

    `chart_spec` is either None (no chart was useful for this result) or
    a JSON-safe dict produced by app.core.chart_generator.generate_chart.
    """

    st.html(
        """
        <div class="result-header">
            VISUALIZATION
        </div>
        """
    )

    data = (chart_spec or {}).get("data", [])

    if not chart_spec or not data or len(data) <= 1:
        # Gracefully handle single-row, single-column, or scalar metrics
        if (
            query_result is not None
            and getattr(query_result, "dataframe", None) is not None
            and not query_result.dataframe.empty
        ):
            df = query_result.dataframe

            # Case 1: Exactly 1 row and 1 column (e.g. single scalar metric like total, count, average)
            if len(df) == 1 and len(df.columns) == 1:
                col_name = str(df.columns[0]).replace("_", " ").title()
                raw_val = df.iloc[0, 0]
                if pd.isna(raw_val):
                    display_val = "N/A"
                elif isinstance(raw_val, (float, np.floating)):
                    display_val = f"{raw_val:,.2f}"
                elif isinstance(raw_val, (int, np.integer)):
                    display_val = f"{raw_val:,}"
                else:
                    display_val = str(raw_val)
                st.caption("Key metric summary:")
                st.metric(label=col_name, value=display_val)
                return

            # Case 2: Exactly 1 row and 2-4 columns (e.g. min, max, avg, sum metrics)
            if len(df) == 1 and 1 < len(df.columns) <= 4:
                st.caption("Key metric summary:")
                metric_cols = st.columns(len(df.columns))
                for idx, col in enumerate(df.columns):
                    raw_val = df.iloc[0, idx]
                    if pd.isna(raw_val):
                        display_val = "N/A"
                    elif isinstance(raw_val, (float, np.floating)):
                        display_val = f"{raw_val:,.2f}"
                    elif isinstance(raw_val, (int, np.integer)):
                        display_val = f"{raw_val:,}"
                    else:
                        display_val = str(raw_val)
                    metric_cols[idx].metric(
                        label=str(col).replace("_", " ").title(),
                        value=display_val,
                    )
                return

            # Case 3: Exactly 1 column with multiple rows (single list)
            if len(df.columns) == 1 and len(df) > 1:
                col_name = str(df.columns[0]).replace("_", " ").title()
                st.caption(
                    f"📋 Single-column result for **{col_name}** ({len(df):,} items) — "
                    "best viewed in the structured table below."
                )
                return

        # Case 4: 1 single data point in chart spec
        if data and len(data) == 1:
            item = data[0]
            label = item.get("category") or item.get("x") or item.get("label") or "Result"
            val = item.get("value") or item.get("y") or item.get("val")
            if val is not None:
                display_val = f"{val:,.2f}" if isinstance(val, (float, np.floating)) else (f"{val:,}" if isinstance(val, (int, np.integer)) else str(val))
                st.caption("Single record result:")
                st.metric(label=str(label).replace("_", " ").title(), value=display_val)
                return

        st.caption(
            "📉 Visualization isn't necessary for this result — the "
            "table below communicates it clearly."
        )
        return

    chart_type = chart_spec.get("chart_type")
    rendered = False

    try:

        if chart_type == "bar":

            category_col = chart_spec.get("x_column", "category")
            value_col = chart_spec.get("y_column", "value")
            is_horizontal = chart_spec.get("orientation") == "horizontal"

            chart_df = pd.DataFrame(
                {
                    category_col: [item["category"] for item in data],
                    value_col: [item["value"] for item in data],
                }
            )

            if is_horizontal:

                # chart_generator.py already returns ranking data in
                # descending metric order.
                #
                # Plotly horizontal categorical axes are displayed
                # bottom-to-top, so reversing the category array places
                # the highest-ranked category at the top.
                fig = px.bar(
                    chart_df,
                    x=value_col,
                    y=category_col,
                    orientation="h",
                )

                fig.update_layout(
                    yaxis=dict(
                        categoryorder="array",
                        categoryarray=list(
                            reversed(
                                chart_df[category_col].tolist()
                            )
                        ),
                    ),
                    xaxis_title=value_col,
                    yaxis_title="",
                )
                fig.update_xaxes(rangemode="tozero")
                _style_plotly_figure(fig)

            else:

                fig = px.bar(
                    chart_df,
                    x=category_col,
                    y=value_col,
                )

                fig.update_layout(
                    xaxis_title=category_col,
                    yaxis_title=value_col,
                )
                fig.update_yaxes(rangemode="tozero")
                _style_plotly_figure(fig)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            rendered = True

        elif chart_type == "grouped_bar":

            _display_grouped_bar_chart(chart_spec, data)

            rendered = True

        elif chart_type == "line":

            y_columns = chart_spec.get("y_columns")

            if y_columns and len(y_columns) > 1:

                # Multi-series time chart: every data point already
                # carries "x" plus one key per numeric series.
                chart_df = pd.DataFrame(data)
                fig = px.line(chart_df, x="x", y=y_columns)
                fig.update_yaxes(autorange=True, rangemode="normal")
                _style_plotly_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

            else:

                chart_df = pd.DataFrame(
                    {
                        "x": [item["x"] for item in data],
                        "y": [item["y"] for item in data],
                    }
                )
                fig = px.line(chart_df, x="x", y="y")
                fig.update_yaxes(autorange=True, rangemode="normal")
                _style_plotly_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

            rendered = True

        elif chart_type == "scatter":

            chart_df = pd.DataFrame(
                {
                    "x": [item["x"] for item in data],
                    "y": [item["y"] for item in data],
                }
            )
            fig = px.scatter(chart_df, x="x", y="y")
            _style_plotly_figure(fig)
            st.plotly_chart(fig, use_container_width=True)

            rendered = True

        elif chart_type == "pie":

            _display_pie_chart(data)

            rendered = True

        else:

            logger.warning(
                "Unknown chart type returned by chart generator: %s",
                chart_type,
            )

    except Exception as exc:

        logger.exception("Failed to render chart")

        st.warning(
            "⚠️ The analysis succeeded, but the chart could not be rendered."
        )

        return

    # "Other" is a summary bucket, not a dropped tail — make it visible
    # whenever the chart actually gained one (bar, grouped_bar, or pie;
    # line/scatter never set this key, so this is a no-op for them).
    if rendered and chart_spec.get("omitted_rows", 0):
        st.caption(f"Showing {chart_spec['displayed_rows']:,} plotted points; {chart_spec['omitted_rows']:,} rows are omitted. The table contains the complete returned result.")
    if rendered and chart_spec.get("sampled"):
        st.caption("This chart uses evenly spaced rows across the full range; intermediate values and outliers may be absent.")
    if rendered and chart_spec.get("omitted_series", 0):
        st.caption(f"{chart_spec['omitted_series']} additional numeric series are available in the table.")
    if rendered and chart_spec.get("has_other"):

        other_count = chart_spec.get("other_count", 0)

        st.caption(
            f"ℹ️ Showing the top categories by value — {other_count:,} "
            "additional categor"
            + ("y is" if other_count == 1 else "ies are")
            + ' grouped into "Other".'
        )


def _display_grouped_bar_chart(chart_spec: dict, data: list) -> None:
    """
    Render a grouped (multi-series) bar chart via Plotly: one categorical
    dimension on the x-axis, with every requested numeric metric drawn
    as its own series of bars within each category's group.
    """

    category_col = chart_spec.get("x_column", "category")
    value_columns = chart_spec.get("y_columns", [])

    chart_df = pd.DataFrame(
        {
            category_col: [item["category"] for item in data],
            **{
                column: [item[column] for item in data]
                for column in value_columns
            },
        }
    )

    fig = px.bar(
        chart_df,
        x=category_col,
        y=value_columns,
        barmode="group",
    )

    fig.update_layout(
        legend_title_text="",
        xaxis_title=category_col,
        yaxis_title="",
    )
    fig.update_yaxes(rangemode="tozero")
    _style_plotly_figure(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def _display_pie_chart(data: list) -> None:
    """Render a pie/donut chart from chart_generator's pie data points."""

    labels = [item["label"] for item in data]
    values = [item["value"] for item in data]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.45,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(showlegend=True)
    _style_plotly_figure(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def _display_analysis_result() -> None:
    """Display the latest analysis result."""
    display_analysis_result(st.session_state.last_result, chart_renderer=_display_chart)


def _render_dataset_workspace() -> None:
    """Prominent upload and relationship controls shared by both workflows."""
    st.subheader("Your dataset workspace")

    st.caption(
        "Upload CSV, XLSX, TXT or PDF files. Confirm suggested relationships before asking for joins."
    )
    st.caption("Questions, schema samples, and bounded result rows may be sent to Gemini.")

    uploaded_file = st.file_uploader(
        "Upload CSV, XLSX, TXT or PDF files",
        type=["csv", "xlsx", "txt", "pdf"],
        accept_multiple_files=True,
        key=f"csv_uploader_{st.session_state.uploader_key_version}",
        help=(
            f"Maximum upload size: "
            f"{config.max_upload_size_mb} MB"
        ),
    )

    # Streamlit returns a list for multi-file mode. Tests and older
    # adapters may return one UploadedFile, so normalize both.
    uploaded_files = uploaded_file if isinstance(uploaded_file, list) else ([uploaded_file] if uploaded_file is not None else [])

    inspected = None
    if uploaded_files:
        try:
            inspected = _inspect_format_files(uploaded_files)
            current_upload_signature = (
                tuple(_uploaded_file_signature(item) for item in uploaded_files)
                + tuple((index, tuple(sheets)) for index, (_, _, kind, sheets)
                        in enumerate(inspected) if kind == "xlsx")
            )
            if (current_upload_signature != st.session_state.uploaded_file_signature
                    and current_upload_signature != st.session_state.failed_upload_signature):
                if _uses_original_csv_path(inspected):
                    if len(uploaded_files) == 1:
                        _load_uploaded_file(uploaded_files[0], current_upload_signature)
                    else:
                        _load_multiple_files(uploaded_files, current_upload_signature)
                else:
                    _load_format_files(inspected, current_upload_signature)
        except (ValueError, DatasetResourceError, DataLoaderError) as exc:
            st.session_state.upload_error = True
            st.error(f"Could not inspect the selected file: {exc}")
        except Exception:
            st.session_state.upload_error = True
            logger.exception("Unexpected upload inspection error")
            st.error("Could not inspect the selected file. Check its contents and retry.")

    if st.session_state.upload_error and uploaded_files and inspected is not None:
        if st.button("Retry selected upload", use_container_width=True):
            if _uses_original_csv_path(inspected) and len(uploaded_files) == 1:
                _load_uploaded_file(uploaded_files[0], current_upload_signature)
            elif _uses_original_csv_path(inspected):
                _load_multiple_files(uploaded_files, current_upload_signature)
            else:
                _load_format_files(inspected, current_upload_signature)
    if not uploaded_files:
        st.session_state.upload_error = False
        st.session_state.failed_upload_signature = None

    status_label, status_modifier, _ = _workspace_status()
    filename = html.escape(
        str(st.session_state.uploaded_filename or "No dataset loaded")
    )
    st.html(
        f"""
        <div class="sidebar-status {status_modifier}">
            <span class="sidebar-status-dot"></span>
            <span>{status_label} · {filename}</span>
        </div>
        """
    )

    _render_catalog_controls()
    def prepare_preprocessing_catalog():
        from app.core.dataset_catalog import DatasetCatalog
        if len(uploaded_files) != 1:
            raise ValueError("The original CSV upload is no longer available. Upload it again.")
        catalog = DatasetCatalog()
        try:
            record = catalog.add(uploaded_files[0], getattr(uploaded_files[0], "name", "dataset.csv"))
            catalog.rename_single_table(record.dataset_id)
            return catalog, record.dataset_id
        except Exception:
            catalog.close()
            raise

    render_preprocessing(prepare_preprocessing_catalog, _refresh_catalog_session,
                         _activate_format_files)
    _render_document_previews()

    if st.button(
        "Clear active dataset",
        disabled=st.session_state.dataframe is None and not st.session_state.documents,
        use_container_width=True,
    ):
        _clear_dataset()
        st.rerun()



def main() -> None:
    """Run the Streamlit application."""

    _initialize_session_state()

    if not _render_public_demo_access_gate():
        return

    st.title(config.app_name)
    st.caption("Explore uploaded data, verify relationships, and turn questions into validated SQL insights.")
    st.html('<div id="your-data"></div>')
    with st.container(border=True):
        _render_dataset_workspace()

    logger.info(
        "QueryPilotAI started | env=%s | provider=%s",
        config.app_env,
        config.llm_provider,
    )

    # ----------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------

    with st.sidebar:

        st.subheader("Workspace")
        st.markdown("[Data](#your-data) · [Analysis](#analysis) · [Charts](#charts)")
        st.caption("Visualization")

        chart_option = st.selectbox(
            "Chart type",
            options=[
                "Automatic",
                "Bar",
                "Grouped Bar",
                "Line",
                "Pie",
                "Scatter",
            ],
            index=0,
        )

        chart_type_map = {
            "Automatic": None,
            "Bar": "bar",
            "Grouped Bar": "grouped_bar",
            "Line": "line",
            "Pie": "pie",
            "Scatter": "scatter",
        }

        selected_chart_type = chart_type_map[chart_option]

        st.caption("Session")

        if st.button(
            "Clear follow-up context",
            disabled=not st.session_state.conversation_history,
            use_container_width=True,
            help=(
                "Forget prior questions from this browser session while "
                "keeping the current dataset loaded."
            ),
        ):
            _clear_conversation_history()
            st.rerun()

    if st.session_state.dataframe is None:
        if any(document.get("format") == "pdf" for document in st.session_state.documents):
            st.info("PDF text preview is available above. Add a table to ask data questions.")
        elif st.session_state.documents:
            st.info("TXT document preview is available above. Add a table to ask data questions.")
        elif st.session_state.pdf_notices or st.session_state.pdf_page_errors:
            st.info("No queryable table was extracted from this PDF.")
        elif not st.session_state.upload_error:
            st.info("Upload a CSV, XLSX, TXT or PDF file above to begin, or select multiple files to build a connected workspace.")
        return

    # ----------------------------------------------------------------------
    # Dataset overview
    # ----------------------------------------------------------------------

    if st.session_state.upload_error:
        st.warning("The selected CSV was not loaded. Analysis is paused. Retry, choose a valid CSV, or remove the rejected upload to resume the previous dataset.")
        return

    _display_dataset_overview()

    # ----------------------------------------------------------------------
    # Natural-language question
    # ----------------------------------------------------------------------

    st.html(
        """
        <div class="section-label">
            NATURAL LANGUAGE QUERY
        </div>
        """
    )

    st.html('<div id="analysis"></div>')
    composer_col, context_col = st.columns((2.2, 1), gap="medium")

    with composer_col:
        with st.container(border=True):
            st.subheader("Ask a question about your data")
            if st.session_state.get("pending_query") is not None:
                st.session_state.query_input_composer = st.session_state.pop("pending_query")

            question = st.text_area(
                "Your question",
                placeholder=(
                    "Example: What are the top 10 products by total sales?"
                ),
                height=110,
                max_chars=config.max_question_chars,
                label_visibility="collapsed",
                key="query_input_composer",
            )

            suggestions = _get_suggested_questions(st.session_state.dataset_profile)
            if suggestions:
                st.html('<div class="suggested-prompts-header">Suggested Questions</div>')
                sug_cols = st.columns(len(suggestions))
                for idx, sug in enumerate(suggestions):
                    with sug_cols[idx]:
                        st.button(
                            f"💡 {sug}",
                            key=f"sug_btn_{idx}",
                            use_container_width=True,
                            help=f"Click to load: {sug}",
                            on_click=_apply_suggested_question,
                            args=(sug,),
                        )

            analyze_clicked = st.button(
                "Analyze question",
                type="primary",
                use_container_width=True,
            )

    with context_col:
        with st.container(border=True):
            st.subheader("Session context")
            history_count = len(st.session_state.conversation_history)
            if history_count:
                st.caption(
                    f"Follow-up context is active from the last "
                    f"{history_count} successful "
                    f"{'question' if history_count == 1 else 'questions'} in "
                    "this browser session."
                )
            else:
                st.caption("No follow-up context yet. Each successful analysis can inform the next question.")
            if st.session_state.last_result is not None:
                st.caption("Latest analysis is shown below.")
            else:
                st.caption("Run a question to populate insights, charts, and SQL.")

    if analyze_clicked:

        if not question.strip():

            st.warning(
                "Please enter a question before clicking Analyze."
            )

        else:

            _run_analysis(
                question=question.strip(),
                chart_type=selected_chart_type,
            )

    # ----------------------------------------------------------------------
    # Latest result
    # ----------------------------------------------------------------------

    _display_analysis_result()

    # Keep the question and latest result ahead of optional dataset exploration.
    st.html('<div id="charts"></div>')
    _display_explore_dataset()




# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()
