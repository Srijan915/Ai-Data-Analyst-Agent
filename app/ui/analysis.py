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

    with st.spinner("Planning and checking SQL, then preparing your results..."):

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
            logger.warning("Analysis stopped because the request could not be answered safely")
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
                "🛡️ The generated SQL did not pass the security checks. Try rephrasing your question."
            )
            return

        except AgentExecutionError as exc:

            logger.exception("SQL execution failed")

            st.error(
                "❌ The validated SQL could not be executed. Try a simpler question or check the dataset columns."
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
    """Keep integer IDs readable without forcing float or timestamp precision."""
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

    return "success", "✅", "Query completed. Review the result and SQL below."


def _download_results(dataframe):
    st.download_button(
        "Export results as CSV",
        data=dataframe.to_csv(index=False).encode("utf-8-sig"),
        file_name="query_results.csv", mime="text/csv", key="export_query_results",
    )


def display_analysis_result(result: Any, chart_renderer: Optional[Callable] = None) -> None:
    """Render the latest grounded result in three compact cards."""
    if result is None:
        columns = st.columns(3)
        for column, title in zip(columns, ("Query result", "Chart", "Key insights")):
            with column, st.container(border=True):
                st.subheader(title)
                st.caption("Ask a question to see results.")
        return
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
    table_col, chart_col, insight_col = st.columns(3, gap="medium")
    query_result = getattr(result, "query_result", None)
    with table_col, st.container(border=True):
        st.subheader("Query result")
        if query_result is not None and getattr(query_result, "dataframe", None) is not None:
            df = query_result.dataframe
            if getattr(query_result, "truncated", False):
                st.caption(
                    f"Rows returned: {query_result.row_count:,} "
                    "| Reached the result limit; additional rows may be available."
                )
            else:
                st.caption(f"Rows returned: {query_result.row_count:,}")

            if df.empty:
                st.info("No rows matched this query. Try adjusting the question or filters.")

            col_cfg = _build_column_config(df)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg,
            )
            _download_results(df)
        else:
            st.caption("No tabular result is available for this question.")
    chart_spec = getattr(result, "chart", None)
    with chart_col, st.container(border=True):
        st.subheader("Chart")
        if chart_renderer is not None and (chart_spec is not None or query_result is not None):
            try:
                chart_renderer(chart_spec, query_result=query_result)
            except Exception:
                logger.exception("Failed to render chart")
                st.warning("Chart could not be rendered")
        else:
            st.caption("No chart is available for this result.")
    with insight_col, st.container(border=True):
        st.subheader("Key insights")
        explanation = getattr(result, "explanation", None)
        if explanation:
            parts = [part.strip() for part in explanation.split(";") if part.strip()]
            st.markdown("\n".join(f"- {part}" for part in parts[:3]))
            if len(parts) > 3:
                st.markdown("- " + "; ".join(parts[3:]))
        else:
            st.caption("No explanation was generated for this result.")
    validation = getattr(result, "validation", None)
    executed_sql = getattr(validation, "cleaned_sql", None)
    sql_text = executed_sql if isinstance(executed_sql, str) and executed_sql else getattr(result, "sql", None)
    if sql_text:
        with st.expander("Generated SQL", expanded=False):
            if validation is not None and validation.is_valid and executed_sql:
                st.html('<span class="validated-tag">Validated: read-only</span>')
            st.caption("Validated SQL sent to DuckDB, including any enforced row limit.")
            st.code(sql_text, language="sql")
