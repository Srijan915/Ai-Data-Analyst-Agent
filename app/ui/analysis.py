"""Analysis action, results display, and safe user-facing error mapping."""
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import streamlit as st
from app.core.agent import (
    AgentCapacityError,
    AgentDataError,
    AgentExecutionError,
    AgentPlanningError,
    AgentValidationError,
)
from app.core.query_planner import MAX_CONVERSATION_TURNS
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
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _planning_error_message(error):
    messages = {
        LLMCredentialsError: "The AI service credentials are missing or invalid. The app owner needs to check the private deployment settings.",
        LLMAccessDeniedError: "The configured AI model is unavailable for this account. The app owner needs to check model access.",
        LLMThrottlingError: "The AI request quota or hourly service budget has been reached. Try again later.",
        LLMConfigurationError: "The configured AI model or request is unavailable. The app owner needs to check the model setting.",
        LLMTLSVerificationError: "The secure connection to the AI service could not be verified. The app owner needs to configure a trusted TLS certificate.",
        LLMTimeoutError: "The AI service request timed out. Check the network connection and try again.",
        LLMNetworkError: "The AI service could not be reached. Check the network or proxy connection and try again.",
        LLMAPIError: "The AI service could not be reached or rejected the request. Try again shortly; if it persists, contact the app owner.",
    }
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for kind, message in messages.items():
            if isinstance(error, kind):
                return message
        error = error.__cause__ or error.__context__
    return "❌ The AI model could not generate a valid analysis plan."


def _run_analysis(
    question: str,
    chart_type: str | None,
) -> None:
    """Run the complete DataAnalystAgent pipeline."""

    if st.session_state.get("upload_error", False):
        st.error("The selected CSV was not loaded. Choose a valid file before analyzing.")
        return
    agent = st.session_state.agent

    if agent is None:
        st.error(
            "Please upload a CSV file before asking a question."
        )
        return

    st.session_state.last_result = None

    with st.spinner("Analyzing your question..."):

        try:

            result = agent.run(
                question=question,
                generate_explanation=True,
                generate_chart_spec=True,
                chart_type=chart_type,
                conversation_context=(
                    st.session_state.conversation_history
                ),
            )

        except AgentDataError as exc:
            logger.warning("Analysis stopped by conversion fidelity check")
            st.warning(str(exc))
            return

        except AgentCapacityError:

            logger.warning("Analysis request rejected because capacity is full")

            st.warning(
                "⏳ The app is busy processing another analysis. "
                "Please try again in a moment."
            )
            return

        except AgentPlanningError as exc:

            logger.exception("Agent planning failed")

            st.error(
                _planning_error_message(exc)
            )
            return

        except AgentValidationError as exc:

            logger.exception("SQL validation failed")

            st.error(
                "🛡️ The generated SQL did not pass the security checks."
            )
            return

        except AgentExecutionError as exc:

            logger.exception("SQL execution failed")

            st.error(
                "❌ The validated SQL could not be executed."
            )
            return

        except Exception as exc:

            logger.exception(
                "Unexpected agent execution error"
            )

            st.error(
                "❌ The analysis could not be completed."
            )
            return

    st.session_state.last_result = result

    # Only completed, validated work becomes context for a later follow-up.
    # The planner itself applies the same bound again as a defense in depth.
    if result.validation.is_valid and result.validation.cleaned_sql:
        history = list(st.session_state.conversation_history)
        history.append(
            {
                "question": result.question,
                "sql": result.validation.cleaned_sql,
            }
        )
        st.session_state.conversation_history = history[
            -MAX_CONVERSATION_TURNS:
        ]

        logger.info(
            "Conversation context updated | retained_turns=%d",
            len(st.session_state.conversation_history),
        )


def _build_column_config(df: pd.DataFrame) -> dict:
    """Build column_config mapping for formatted numeric and date displays."""
    col_config = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return col_config
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            col_config[col] = st.column_config.NumberColumn(
                col,
                format="%d",
            )
        elif pd.api.types.is_float_dtype(dtype):
            col_config[col] = st.column_config.NumberColumn(
                col,
                format="%.2f",
            )
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_config[col] = st.column_config.DatetimeColumn(
                col,
                format="YYYY-MM-DD",
            )
    return col_config


def _determine_result_state(result: Any) -> Tuple[str, str, str]:
    """
    Determine the result status: (state_type, icon, explanatory_sentence).
    state_type is one of 'success', 'warning', 'refusal'.
    """
    if result is None:
        return "refusal", "🛡️", "No analysis result is available for this request."

    validation = getattr(result, "validation", None)
    if validation is not None and not validation.is_valid:
        return "refusal", "🛡️", "The request was refused because it did not pass security validation."

    query_result = getattr(result, "query_result", None)
    if query_result is None:
        return "refusal", "🛡️", "The request could not be fulfilled as an executable query."

    warnings = getattr(result, "warnings", [])
    if warnings:
        return "warning", "⚠️", "Analysis completed with advisory warnings or caveats."

    if getattr(query_result, "truncated", False):
        return "warning", "⚠️", "Analysis result reached the maximum row limit and was truncated."

    return "success", "✅", "Analysis completed successfully and verified against the dataset."


def display_analysis_result(result: Any, chart_renderer: Optional[Callable] = None) -> None:
    """
    Render analysis result area in priority order:
    1. Answer card
    2. Result table
    3. Chart
    4. Collapsible SQL
    """
    if result is None:
        return

    st.html(
        """
        <div class="result-header-bar">
            <div class="section-label">
                ANALYSIS RESULT
            </div>
            <div class="result-badge">
                Verified Output
            </div>
        </div>
        """
    )

    state_type, icon, sentence = _determine_result_state(result)
    st.html(
        f"""
        <div class="result-status-banner status-{state_type}">
            <span class="status-icon">{icon}</span>
            <span class="status-message">{sentence}</span>
        </div>
        """
    )

    for warning in getattr(result, "warnings", []):
        st.warning(f"⚠️ {warning}")

    # 1. Answer card
    with st.container(border=True):
        st.html(
            """
            <div class="card-header-badge ai-badge">
                <span class="badge-icon">✦</span>
                <span>AI EXECUTIVE INSIGHT</span>
            </div>
            """
        )
        st.subheader("Key insight")
        if getattr(result, "explanation", None):
            st.markdown(result.explanation)
        else:
            st.caption("No explanation was generated for this result.")

    # 2. Result table
    with st.container(border=True):
        st.html(
            """
            <div class="card-header-badge table-badge">
                <span class="badge-icon">📋</span>
                <span>STRUCTURED QUERY RESULT</span>
            </div>
            """
        )
        st.subheader("Query result")

        query_result = getattr(result, "query_result", None)
        if query_result is not None and getattr(query_result, "dataframe", None) is not None:
            df = query_result.dataframe
            if getattr(query_result, "truncated", False):
                st.caption(
                    f"Rows returned: {query_result.row_count:,} "
                    "| Reached the result limit; additional rows may be available."
                )
            else:
                st.caption(f"Rows returned: {query_result.row_count:,}")

            col_cfg = _build_column_config(df)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg,
            )
        else:
            st.caption("No tabular result is available for this question.")

    # 3. Chart
    chart_spec = getattr(result, "chart", None)
    if chart_renderer is not None and (chart_spec is not None or getattr(result, "query_result", None) is not None):
        with st.container(border=True):
            st.html(
                """
                <div class="card-header-badge chart-badge">
                    <span class="badge-icon">📊</span>
                    <span>VISUALIZATION ENGINE</span>
                </div>
                """
            )
            st.subheader("Visualization")
            try:
                chart_renderer(chart_spec, query_result=getattr(result, "query_result", None))
            except Exception:
                logger.exception("Failed to render chart")
                st.warning("Chart could not be rendered")

    # 4. Collapsible SQL
    sql_text = getattr(result, "sql", None)
    if sql_text:
        with st.expander("Generated SQL", expanded=False):
            st.caption("Executed safely in sandboxed DuckDB query engine")
            st.code(sql_text, language="sql")
