"""
Chart generation utilities — the "Visualization Intelligence" stage of the
pipeline.

Converts a QueryResult containing a pandas DataFrame into a chart-friendly
specification that the Streamlit UI can render.

This module intentionally does NOT render charts itself. It only inspects
query results (and, optionally, light context about the question/SQL that
produced them) and determines:

- whether the result is suitable for visualization at all,
- which columns should be used for the x-axis and y-axis,
- which chart type is most appropriate,
- rendering hints (e.g. horizontal vs. vertical bar orientation), and
- a JSON-safe representation of the result data.

Supported chart types:
    - bar         (vertical or horizontal, via the "orientation" hint)
    - grouped_bar (one categorical dimension + 2+ numeric metrics, e.g.
                   "revenue vs cost by region" — each metric becomes its
                   own series within each category's group of bars)
    - line        (single-series, or multi-series when a time axis
                   carries several numeric columns)
    - pie
    - scatter

Core product rule
------------------
"If the result can meaningfully communicate information visually,
automatically visualize it. Do NOT generate meaningless charts."

The DataFrame supplies the chart dimensions. Validated SQL supplies conservative
additive-metric evidence: only simple SUM/non-distinct COUNT outputs authorize
combining duplicate categories or an "Other" bucket. Unknown/non-additive metrics
remain uncombined, and capped subsets disclose omissions. Extra grouping
dimensions are never silently collapsed. Line/scatter sampling spans the full
returned range and reports omitted rows and series.

Design goals:
    - Never mutate the input DataFrame.
    - Never make network or external provider calls.
    - Never execute SQL.
    - Keep chart-selection deterministic and testable.
    - Return JSON-safe chart specifications.
    - Never raise merely because a result isn't visualizable — return None.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.models.schemas import QueryResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DARK_THEME_COLORWAY = [
    "#0EA5E9",  # Electric sky / teal
    "#06B6D4",  # Cyan
    "#6366F1",  # Indigo
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
]

DEFAULT_MAX_CATEGORIES = 20
DEFAULT_MAX_ROWS = 1000

# Cardinality thresholds for choosing pie/donut over bar. A slightly higher
# ceiling applies when the question explicitly signals a "proportion of
# total" framing (e.g. "percentage of orders by category").
PIE_MAX_CATEGORIES_DEFAULT = 6
PIE_MAX_CATEGORIES_PROPORTION = 8

# Cap on how many numeric columns are drawn as separate lines in a
# multi-series time chart, so the chart stays legible.
DEFAULT_MAX_LINE_SERIES = 4

# Same idea for a grouped bar chart's metric series.
DEFAULT_MAX_GROUPED_BAR_SERIES = 4

SUPPORTED_CHART_TYPES = {
    "bar",
    "grouped_bar",
    "line",
    "pie",
    "scatter",
}

# Lightweight keyword signals used only to *tie-break* between chart types
# that are otherwise equally valid for the same DataFrame shape (e.g.
# category+numeric could reasonably be a bar or a pie). These never decide
# *whether* to chart — only *which* chart — and the DataFrame shape always
# has final say.
_RANKING_KEYWORDS = (
    "top",
    "highest",
    "lowest",
    "best",
    "worst",
    "most",
    "least",
    "rank",
    "ranked",
    "ranking",
    "bottom",
    "largest",
    "smallest",
)

_PROPORTION_KEYWORDS = (
    "percentage",
    "percent",
    "%",
    "share",
    "proportion",
    "distribution",
    "breakdown",
    "split",
    "composition",
)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_chart(
    query_result: QueryResult,
    chart_type: Optional[str] = None,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_rows: int = DEFAULT_MAX_ROWS,
    question: Optional[str] = None,
    sql: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a chart specification from a QueryResult.

    Parameters
    ----------
    query_result:
        QueryResult containing the pandas DataFrame to visualize. This is
        the primary — and in "automatic" mode, the only — signal used to
        decide whether and how to visualize the result.

    chart_type:
        Optional explicit chart type. Supported values are:
        "bar", "grouped_bar", "line", "pie", and "scatter".

        If omitted (None / "Automatic" in the UI), the chart type — and
        whether to chart at all — is inferred automatically from the
        DataFrame's shape.

        If provided, that type is attempted against the DataFrame. When
        the data isn't compatible with the requested type (e.g. "scatter"
        with only one numeric column), this returns None rather than
        raising or fabricating data, so the caller can fail gracefully.

    max_categories:
        Maximum number of categorical values to include in a chart.

    max_rows:
        Maximum number of rows copied into the chart specification.

    question:
        Optional natural-language question that produced this result.
        Used only as a *tie-breaker* between otherwise-equally-valid chart
        types for the same DataFrame shape (e.g. preferring a ranked bar
        chart over a pie chart for "top 10 ..." questions). Never expands
        *whether* a chart is shown — that is decided from the DataFrame.

    sql:
        Optional generated SQL for the same purpose as `question` (e.g.
        detecting an `ORDER BY ... LIMIT` ranking query).

    Returns
    -------
    Optional[Dict[str, Any]]
        A JSON-safe chart specification, or None when the result cannot
        reasonably be visualized (or isn't useful to visualize).

    Raises
    ------
    ValueError
        If chart_type or limits are invalid.
    """

    if not isinstance(query_result, QueryResult):
        raise ValueError("query_result must be a QueryResult")

    if max_categories <= 0:
        raise ValueError("max_categories must be greater than 0")

    if max_rows <= 0:
        raise ValueError("max_rows must be greater than 0")

    if chart_type is not None:
        chart_type = chart_type.strip().lower()

        if chart_type not in SUPPORTED_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart_type '{chart_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHART_TYPES)}"
            )

    df = query_result.dataframe

    if not isinstance(df, pd.DataFrame):
        raise ValueError("query_result.dataframe must be a pandas DataFrame")

    if df.empty or len(df.columns) == 0:
        logger.info("Skipping chart generation because query result is empty")
        return None

    if len(df.columns) <= 1:
        logger.info(
            "Skipping chart generation because query result has only %d column",
            len(df.columns),
        )
        return None

    numeric_columns = _numeric_columns(df)
    categorical_columns = _categorical_columns(df)
    datetime_columns = _datetime_columns(df)

    is_ranking = _looks_like_ranking(question, sql)
    is_proportion = _looks_like_proportion(question)

    selected_type = chart_type or _infer_chart_type(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        is_ranking=is_ranking,
        is_proportion=is_proportion,
    )

    if selected_type is None:
        logger.info(
            "No suitable chart type found | rows=%d | columns=%d",
            len(df),
            len(df.columns),
        )
        return None

    additive = _additive_columns(sql)
    if selected_type == "pie" and sql and numeric_columns[0] not in additive:
        # Rates and averages are comparisons, not parts of an additive whole.
        if chart_type is not None:
            return None
        selected_type = "bar"
    category_type = selected_type in {"bar", "pie", "grouped_bar"}
    if category_type and (len(categorical_columns) != 1 or datetime_columns):
        return None
    if selected_type == "line" and any(df[c].nunique(dropna=False) > 1 for c in categorical_columns):
        return None
    if selected_type == "scatter":
        spec = _build_scatter_chart(
            df,
            numeric_columns,
            max_rows=max_rows,
        )

    elif selected_type == "line":
        spec = _build_line_chart(
            df,
            numeric_columns,
            datetime_columns,
            max_rows=max_rows,
        )

    elif selected_type == "pie":
        spec = _build_pie_chart(
            df,
            categorical_columns,
            numeric_columns,
            max_categories=max_categories,
            additive_columns=additive,
        )

    elif selected_type == "bar":
        spec = _build_bar_chart(
            df,
            categorical_columns,
            numeric_columns,
            max_categories=max_categories,
            orientation="horizontal" if is_ranking else "vertical",
            additive_columns=additive,
        )

    elif selected_type == "grouped_bar":
        spec = _build_grouped_bar_chart(
            df,
            categorical_columns,
            numeric_columns,
            max_categories=max_categories,
            additive_columns=additive,
        )

    else:
        # Defensive guard. The value is already validated above.
        return None

    if spec is None:
        logger.info(
            "Chart generation produced no usable specification | type=%s",
            selected_type,
        )
        return None

    logger.info(
        "Generated chart | type=%s | rows=%d | columns=%d",
        selected_type,
        len(df),
        len(df.columns),
    )

    spec["displayed_rows"] = len(spec["data"])
    spec["input_rows"] = len(df)
    spec.setdefault("omitted_rows", max(0, len(df) - len(spec["data"]) - spec.get("other_count", 0)))
    selected_metrics = spec.get("y_columns") or [spec.get("y_column", spec.get("value_column"))]
    spec["omitted_series"] = max(0, len(numeric_columns) - (2 if selected_type == "scatter" else len(selected_metrics)))
    return spec


# --------------------------------------------------------------------------
# Chart type inference
# --------------------------------------------------------------------------


def _infer_chart_type(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    datetime_columns: List[str],
    is_ranking: bool = False,
    is_proportion: bool = False,
) -> Optional[str]:
    """
    Infer the most useful chart type from DataFrame structure.

    This is the core "should we chart, and with what" decision for
    Automatic mode. `is_ranking`/`is_proportion` only ever choose between
    chart types that the DataFrame shape already supports — they cannot
    conjure a chart out of data that doesn't support one.

    Priority:
        1. A single row (or no rows) carries nothing to visually compare —
           a metric or table communicates it better than a chart. No chart.
        2. line for datetime + numeric (single or multi-series).
        3. scatter for two or more numeric columns (no datetime axis).
        4. categorical + 2 or more numeric metrics: grouped_bar (e.g.
           "revenue vs cost by region") — a single metric could reasonably
           be squeezed into a pie, but two or more genuinely need their
           own series, not a two-point scatter that discards the category.
        5. categorical + exactly one numeric metric: pie for a small,
           proportion-flavored distribution; horizontal-bar-flagged bar
           for a ranking question; otherwise pie for small cardinality,
           bar for larger cardinality.
        6. None — no combination of columns says anything visually.
    """

    # A single row (e.g. "which customer had the highest purchase?") or an
    # empty frame has nothing to compare visually — a metric/table is the
    # right presentation, not a chart with one bar/slice/point.
    if len(df) <= 1:
        return None

    if datetime_columns and numeric_columns:
        return "line"

    if len(numeric_columns) >= 2 and not categorical_columns:
        return "scatter"

    if categorical_columns and numeric_columns:

        if len(numeric_columns) >= 2:
            return "grouped_bar"

        category = categorical_columns[0]
        cardinality = df[category].nunique(dropna=True)

        if is_proportion and cardinality <= PIE_MAX_CATEGORIES_PROPORTION:
            return "pie"

        if is_ranking:
            return "bar"

        if cardinality <= PIE_MAX_CATEGORIES_DEFAULT:
            return "pie"

        return "bar"

    return None


# --------------------------------------------------------------------------
# Question/SQL context signals (tie-breakers only, see module docstring)
# --------------------------------------------------------------------------


def _looks_like_ranking(
    question: Optional[str],
    sql: Optional[str],
) -> bool:
    """Detect a "top N" / ranking framing from the question or SQL."""

    combined = f"{question or ''} {sql or ''}".strip().lower()

    if not combined:
        return False

    if any(keyword in combined for keyword in _RANKING_KEYWORDS):
        return True

    # A generated `ORDER BY ... LIMIT ...` is a strong structural signal
    # of a ranking query even when the question's wording doesn't use an
    # obvious ranking word (e.g. "which products sell the most?").
    return "order by" in combined and "limit" in combined


def _looks_like_proportion(question: Optional[str]) -> bool:
    """Detect a "share of the whole" framing from the question."""

    if not question:
        return False

    lowered = question.lower()

    return any(keyword in lowered for keyword in _PROPORTION_KEYWORDS)


# --------------------------------------------------------------------------
# Individual chart builders
# --------------------------------------------------------------------------


def _additive_columns(sql):
    if not sql:
        return set()
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError:
        return set()
    result = set()
    if not isinstance(tree, exp.Select):
        return result
    group = tree.args.get("group")
    if group is not None and any(group.args.get(key) for key in ("rollup", "cube", "grouping_sets")):
        return result
    for projection in tree.expressions:
        body = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(body, (exp.Sum, exp.Count)) and not body.find(exp.Distinct):
            result.add(projection.alias_or_name)
    return result


def _sample_rows(working, limit):
    if len(working) <= limit:
        return working
    positions = np.linspace(0, len(working)-1, num=limit, dtype=int)
    return working.iloc[positions]


def _category_values(working, column):
    if working[column].isna().any():
        label = "Missing (NULL)"
        while label in set(working[column].dropna()):
            label += " (NULL)"
        working[column] = working[column].astype(object).fillna(label)
    return working


def _apply_top_n_with_other(
    working: pd.DataFrame,
    category_column: str,
    value_columns: List[str],
    max_categories: int,
    allow_other: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Cap an already-aggregated, descending-sorted-by-primary-metric
    DataFrame to at most `max_categories` rows.

    - If the data already fits within `max_categories`, it is returned
      unchanged. This is what keeps an explicitly-limited ranking query
      (e.g. a generated `ORDER BY ... LIMIT 10` with the default
      max_categories=20) from ever gaining a fabricated "Other" row —
      there is nothing left over to aggregate, because the DataFrame
      itself doesn't contain any more categories than we're charting.

    - If there ARE more categories than `max_categories` allows, the
      top `max_categories - 1` categories are kept as-is and everything
      else is summed into one synthetic "Other" row only when allow_other
      explicitly authorizes additive metrics. Otherwise the capped subset is
      reported as omitted rows. An additive bucket preserves the
      total. "Other" is always appended last — it's a catch-all bucket,
      not a ranked value, so it is never re-sorted into the list by its
      (possibly large) aggregate value.

    Returns the possibly-truncated DataFrame plus the count of original
    categories that were folded into "Other" (0 when nothing was).
    """

    total_categories = len(working)

    if total_categories <= max_categories:
        return working.reset_index(drop=True), 0

    if not allow_other:
        return working.head(max_categories).reset_index(drop=True), 0
    if "Other" in set(working[category_column]):
        # Never collide a genuine category with the synthetic summary bucket.
        return working.head(max_categories).reset_index(drop=True), 0
    keep = max(max_categories - 1, 0)
    top = working.head(keep)
    remainder = working.iloc[keep:]

    if remainder.empty:
        return top.reset_index(drop=True), 0

    other_row: Dict[str, Any] = {category_column: "Other"}

    for column in value_columns:
        other_row[column] = remainder[column].sum(min_count=1)

    combined = pd.concat(
        [top, pd.DataFrame([other_row])],
        ignore_index=True,
    )

    return combined, len(remainder)


def _build_bar_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
    orientation: str = "vertical",
    additive_columns=None,
) -> Optional[Dict[str, Any]]:
    """
    Build a bar chart from categorical and numeric columns.

    `orientation` is a rendering hint only ("horizontal" or "vertical") —
    it does not affect which rows are selected. Callers typically request
    "horizontal" for ranking/top-N results, where long category labels
    (e.g. restaurant names) read better on a horizontal axis.

    When there are more distinct categories than `max_categories`, the
    remainder is folded into a single "Other" row rather than dropped —
    see `_apply_top_n_with_other`.
    """

    if not categorical_columns or not numeric_columns:
        return None

    category_column = categorical_columns[0]
    value_column = numeric_columns[0]

    working = df[[category_column, value_column]].copy()

    working = _category_values(working, category_column).dropna(subset=[value_column])

    if working.empty:
        return None

    # Aggregate duplicate categories so the chart remains meaningful.
    allow_sum = set([value_column]).issubset(additive_columns or set())
    if working[category_column].duplicated().any():
        if not allow_sum:
            return None
        working = working.groupby(category_column, as_index=False)[[value_column]].sum(min_count=1)
    working = working.sort_values(value_column, ascending=False).reset_index(drop=True)
    available = len(working)


    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=[value_column],
        max_categories=max_categories,
        allow_other=allow_sum,
    )

    if working.empty:
        return None

    data = [
        {
            "category": _json_safe(row[category_column]),
            "value": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "bar",
        "title": f"{value_column} by {category_column}",
        "x_column": category_column,
        "y_column": value_column,
        "orientation": (
            "horizontal" if orientation == "horizontal" else "vertical"
        ),
        "data": data,
        "has_other": other_count > 0,
        "other_count": other_count,
        "omitted_rows": len(df) - available + (max(0, available - len(working)) if not other_count else 0),
    }


def _build_grouped_bar_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
    max_series: int = DEFAULT_MAX_GROUPED_BAR_SERIES,
    additive_columns=None,
) -> Optional[Dict[str, Any]]:
    """
    Build a grouped (multi-series) bar chart: one categorical dimension
    plus two or more numeric metrics, e.g. "revenue vs cost by region"
    or "actual vs budget by department". Every metric becomes its own
    series within each category's group of bars.

    High-cardinality categories are handled the same way as a plain bar
    chart — see `_apply_top_n_with_other`. Metrics beyond `max_series`
    are dropped (not aggregated) purely to keep the chart legible; the
    metrics kept are always the leading `numeric_columns`, in order.
    """

    if not categorical_columns or len(numeric_columns) < 2:
        return None

    category_column = categorical_columns[0]
    value_columns = numeric_columns[:max_series]

    working = df[[category_column, *value_columns]].copy()
    working = _category_values(working, category_column).dropna(subset=value_columns, how="all")

    if working.empty:
        return None

    # Aggregate duplicate categories, ranking by the first metric.
    allow_sum = set(value_columns).issubset(additive_columns or set())
    if working[category_column].duplicated().any():
        if not allow_sum:
            return None
        working = working.groupby(category_column, as_index=False)[value_columns].sum(min_count=1)
    working = working.sort_values(value_columns[0], ascending=False).reset_index(drop=True)
    available = len(working)


    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=value_columns,
        max_categories=max_categories,
        allow_other=allow_sum,
    )

    if working.empty:
        return None

    data = [
        {
            "category": _json_safe(row[category_column]),
            **{
                column: _json_safe(row[column])
                for column in value_columns
            },
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "grouped_bar",
        "title": f"{', '.join(value_columns)} by {category_column}",
        "x_column": category_column,
        "y_columns": value_columns,
        "data": data,
        "has_other": other_count > 0,
        "other_count": other_count,
        "omitted_rows": len(df) - available + (max(0, available - len(working)) if not other_count else 0),
    }


def _build_line_chart(
    df: pd.DataFrame,
    numeric_columns: List[str],
    datetime_columns: List[str],
    max_rows: int,
    max_series: int = DEFAULT_MAX_LINE_SERIES,
) -> Optional[Dict[str, Any]]:
    """
    Build a line chart using datetime/numeric data.

    When a time axis is available and the result carries more than one
    numeric column (e.g. "revenue" and "profit" over "month"), this
    produces a multi-series line chart: every data point is a single
    dict keyed by column name (plus "x"), and `y_columns` lists the
    series in order. Callers that only care about a single series can
    keep reading `y_column` (the first series) as before.

    With exactly one numeric column, the output is unchanged from before:
    each data point is `{"x": ..., "y": ...}` and only `y_column` is set.
    """

    if not numeric_columns:
        return None

    if datetime_columns:
        x_column = datetime_columns[0]
    else:
        # For purely numeric data, use the DataFrame index as x-axis.
        x_column = "__index__"

    use_multi_series = bool(datetime_columns) and len(numeric_columns) > 1
    value_columns = (
        numeric_columns[:max_series] if use_multi_series else numeric_columns[:1]
    )

    if x_column == "__index__":
        working = df[value_columns].copy()
        working.insert(0, x_column, range(len(working)))
    else:
        working = df[[x_column, *value_columns]].copy()
        working[x_column] = pd.to_datetime(
            working[x_column],
            errors="coerce",
        )
        working = working.dropna(subset=[x_column])

    if working.empty:
        return None

    working = working.sort_values(x_column)
    if not use_multi_series:
        working = working.dropna(subset=value_columns)
    available = len(working)
    working = _sample_rows(working, max_rows)

    if use_multi_series:
        data = [
            {
                "x": _json_safe(row[x_column]),
                **{
                    column: _json_safe(row[column])
                    for column in value_columns
                },
            }
            for _, row in working.iterrows()
        ]

        if not data:
            return None

        return {
            "chart_type": "line",
        "sampled": available > len(working),
        "omitted_rows": len(df) - len(working),
            "title": f"{', '.join(value_columns)} over {x_column}",
            "x_column": x_column,
            "y_column": value_columns[0],
            "y_columns": value_columns,
            "data": data,
        }

    value_column = value_columns[0]

    working = working.dropna(subset=[value_column])

    if working.empty:
        return None

    data = [
        {
            "x": _json_safe(row[x_column]),
            "y": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "line",
        "sampled": available > len(working),
        "omitted_rows": len(df) - len(working),
        "title": f"{value_column} over {x_column}",
        "x_column": x_column,
        "y_column": value_column,
        "data": data,
    }


def _build_pie_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
    additive_columns=None,
) -> Optional[Dict[str, Any]]:
    """
    Build a pie chart for a small categorical distribution.

    When there are more distinct categories than `max_categories`
    (relevant mainly for an explicit user override on a high-cardinality
    result, since automatic inference already keeps pie to small
    cardinalities), the remainder is folded into a single "Other" slice
    rather than dropped — see `_apply_top_n_with_other`.
    """

    if not categorical_columns or not numeric_columns:
        return None

    category_column = categorical_columns[0]
    value_column = numeric_columns[0]

    working = df[[category_column, value_column]].copy()
    working = _category_values(working, category_column).dropna(subset=[value_column])

    if working.empty:
        return None

    allow_sum = set([value_column]).issubset(additive_columns or set())
    if working[category_column].duplicated().any():
        if not allow_sum:
            return None
        working = working.groupby(category_column, as_index=False)[[value_column]].sum(min_count=1)
    working = working.sort_values(value_column, ascending=False).reset_index(drop=True)
    if (working[value_column] < 0).any():
        return None
    available = len(working)


    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=[value_column],
        max_categories=max_categories,
        allow_other=allow_sum,
    )

    if working.empty:
        return None

    data = [
        {
            "label": _json_safe(row[category_column]),
            "value": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "pie",
        "title": f"{value_column} by {category_column}",
        "label_column": category_column,
        "value_column": value_column,
        "data": data,
        "has_other": other_count > 0,
        "other_count": other_count,
        "omitted_rows": len(df) - available + (max(0, available - len(working)) if not other_count else 0),
    }


def _build_scatter_chart(
    df: pd.DataFrame,
    numeric_columns: List[str],
    max_rows: int,
) -> Optional[Dict[str, Any]]:
    """Build a scatter chart using the first two numeric columns."""

    if len(numeric_columns) < 2:
        return None

    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    working = df[[x_column, y_column]].copy()
    working = working.dropna(subset=[x_column, y_column])
    available = len(working)
    working = _sample_rows(working, max_rows)

    if working.empty:
        return None

    data = [
        {
            "x": _json_safe(row[x_column]),
            "y": _json_safe(row[y_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "scatter",
        "sampled": available > len(working),
        "omitted_rows": len(df) - len(working),
        "title": f"{y_column} vs {x_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": data,
    }


# --------------------------------------------------------------------------
# Column classification helpers
# --------------------------------------------------------------------------


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns excluding boolean columns."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def _datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that are already datetime-like."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Return columns suitable as categorical chart dimensions.

    Numeric, datetime, and boolean columns are excluded.
    """

    columns: List[str] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            columns.append(str(column))

    return columns


# --------------------------------------------------------------------------
# JSON safety
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy values into ordinary JSON-safe Python values.

    NaN and NaT become None.
    """

    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None

    except (TypeError, ValueError):
        pass

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    # Convert numpy arrays or other obvious numpy objects conservatively.
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


# --------------------------------------------------------------------------
# Dataset Exploration & Visualization Intelligence
# --------------------------------------------------------------------------


def style_dark_figure(
    fig: go.Figure,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> go.Figure:
    """Apply the workspace dark analytics chart theme to a Plotly figure."""

    layout_updates: Dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.55)",
        "font": dict(
            color="#94A3B8",
            size=12,
            family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif",
        ),
        "colorway": DARK_THEME_COLORWAY,
        "legend": dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=11),
        ),
        "margin": dict(t=16, b=64, l=16, r=16),
        "autosize": True,
        "title": None,
        "hoverlabel": dict(
            bgcolor="#0F172A",
            bordercolor="#0EA5E9",
            font=dict(color="#F8FAFC", size=12),
        ),
    }

    fig.update_layout(**layout_updates)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.07)",
        zeroline=False,
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.07)",
        zeroline=False,
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1"),
    )
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)

    return fig


def classify_dataset_columns(
    df: pd.DataFrame,
    profile: Optional[Any] = None,
) -> Dict[str, List[str]]:
    """
    Classify columns of a dataset into semantic categories for exploration.

    Returns a dict with keys:
        - "numeric": numeric columns (int, float, excluding pure booleans)
        - "categorical": low-to-medium cardinality categorical columns
        - "datetime": date/time columns (datetime64 or parseable dates)
        - "all": all columns
    """
    if df is None or df.empty:
        return {"numeric": [], "categorical": [], "datetime": [], "all": []}

    all_cols = [str(c) for c in df.columns]
    num_cols: List[str] = []
    cat_cols: List[str] = []
    date_cols: List[str] = []

    profile_type_map: Dict[str, str] = {}
    if profile is not None and getattr(profile, "columns", None):
        for col_p in profile.columns:
            profile_type_map[col_p.name] = str(getattr(col_p, "inferred_type", ""))

    for col in all_cols:
        series = df[col]
        inferred = profile_type_map.get(col, "")

        # Datetime check
        if inferred == "datetime" or pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            name_lower = col.lower()
            if any(term in name_lower for term in ("date", "time", "timestamp", "year_month", "period")):
                sample = series.dropna().head(30)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().sum() / len(sample) >= 0.8:
                            date_cols.append(col)
                            continue
                    except Exception:
                        pass

        # Numeric check
        if (
            (inferred in ("integer", "float") or pd.api.types.is_numeric_dtype(series))
            and not pd.api.types.is_bool_dtype(series)
        ):
            num_cols.append(col)
            continue

        # Categorical check
        if (
            inferred in ("string", "category", "boolean")
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            n_unique = series.nunique(dropna=True)
            if 1 < n_unique <= max(100, int(len(df) * 0.8)):
                cat_cols.append(col)
            elif n_unique > 0 and len(cat_cols) == 0:
                cat_cols.append(col)

    def _num_score(c: str) -> int:
        c_lower = c.lower()
        if c_lower in ("id", "index") or c_lower.endswith("_id") or c_lower.startswith("id_"):
            return 10
        return 0

    num_cols.sort(key=_num_score)

    def _cat_score(c: str) -> int:
        n_u = df[c].nunique(dropna=True)
        if 2 <= n_u <= 25:
            return 0
        if 26 <= n_u <= 60:
            return 1
        return 2

    cat_cols.sort(key=_cat_score)

    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "datetime": date_cols,
        "all": all_cols,
    }


def _query_readonly_duckdb(agent: Any, sql: str) -> Optional[pd.DataFrame]:
    """Execute read-only SQL via the agent's existing DuckDB executor context."""
    if agent is None:
        return None
    try:
        from app.utils.security import validate_sql

        if getattr(agent, "_dataset_catalog", None) is not None:
            from app.core.dataset_catalog import CatalogSQLExecutor
            context = CatalogSQLExecutor(agent._dataset_catalog, agent._max_result_rows)
        elif getattr(agent, "_disk_dataset", None) is not None:
            from app.core.disk_dataset import DiskSQLExecutor
            context = DiskSQLExecutor(agent._disk_dataset, agent._max_result_rows)
        elif getattr(agent, "_execution_dataframe", None) is not None:
            from app.core.sql_executor import SQLExecutor
            context = SQLExecutor(agent._execution_dataframe, max_result_rows=agent._max_result_rows)
        else:
            return None

        with context as executor:
            table_names = executor.table_names if hasattr(executor, "table_names") else [executor.table_name]
            validation = validate_sql(sql, allowed_tables=table_names, max_rows=executor.max_result_rows)
            if not validation.is_valid or not validation.cleaned_sql:
                logger.warning("Insight SQL rejected by security: %s", [e.code for e in validation.errors])
                return None
            res = executor.execute(validation)
            return res.dataframe
    except Exception as exc:
        logger.warning("Insight query failed: %s", exc)
        return None


def _get_agent_table_name(agent: Any) -> str:
    """Get the primary table name for DuckDB queries."""
    if agent is not None and getattr(agent, "_dataset_catalog", None) is not None:
        names = getattr(agent._dataset_catalog, "table_names", None)
        if names:
            return names[0]
    return "dataset"


def generate_dataset_auto_visualizations(
    df: pd.DataFrame,
    profile: Optional[Any] = None,
    max_charts: int = 4,
    agent: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Automatically identify suitable columns and generate meaningful visualizations.
    Queries the full dataset through DuckDB whenever possible.
    """
    if df is None or df.empty or len(df) < 2:
        return []

    classified = classify_dataset_columns(df, profile)
    num_cols = classified["numeric"]
    cat_cols = classified["categorical"]
    date_cols = classified["datetime"]

    # Check whether df is a sample vs full dataset
    tbl_name = _get_agent_table_name(agent)
    profile_rows = getattr(profile, "row_count", None) or getattr(profile, "total_rows", None)
    is_sample_fallback = bool(profile_rows and profile_rows > len(df))
    sample_notice = f"Based on first {len(df):,} rows" if is_sample_fallback else None

    charts: List[Dict[str, Any]] = []

    # 1. Categorical Breakdown
    if cat_cols:
        primary_cat = cat_cols[0]
        cat_df = None
        tot_count = None

        if agent is not None:
            sql_cat = f'SELECT "{primary_cat}" AS cat, COUNT(*) AS count FROM "{tbl_name}" WHERE "{primary_cat}" IS NOT NULL GROUP BY "{primary_cat}" ORDER BY count DESC LIMIT 15'
            cat_df = _query_readonly_duckdb(agent, sql_cat)
            if cat_df is not None and not cat_df.empty:
                sql_tot = f'SELECT COUNT(*) AS total_count FROM "{tbl_name}" WHERE "{primary_cat}" IS NOT NULL'
                tot_res = _query_readonly_duckdb(agent, sql_tot)
                tot_count = int(tot_res.iloc[0]["total_count"]) if tot_res is not None and not tot_res.empty else int(cat_df["count"].sum())

        if cat_df is not None and len(cat_df) >= 2:
            num_cats = len(cat_df)
            if num_cats == 2:
                # Replace 2-slice donut with sorted horizontal bar
                pcts = [round(c / tot_count * 100, 1) if tot_count and tot_count > 0 else 0 for c in cat_df["count"]]
                cat_df["pct"] = pcts
                fig = px.bar(cat_df, x="count", y="cat", orientation="h")
                fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed([str(v) for v in cat_df["cat"]]))))
                fig.update_xaxes(rangemode="tozero")
                fig.update_traces(marker_color="#0EA5E9", hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>")
                style_dark_figure(fig, title=f"Distribution: {primary_cat}", x_title="Count", y_title="")
                charts.append({
                    "id": f"bar_{primary_cat}",
                    "title": f"Distribution: {primary_cat}",
                    "chart_type": "bar",
                    "figure": fig,
                    "description": f"Breakdown across categories in {primary_cat}.",
                    "metrics": [
                        {"label": f"{cat_df.iloc[0]['cat']} Share", "value": f"{pcts[0]}%"},
                        {"label": f"{cat_df.iloc[1]['cat']} Share", "value": f"{pcts[1]}%"},
                        {"label": "Total Analyzed", "value": f"{tot_count:,}"},
                    ],
                })
            elif num_cats <= 7:
                labels = [str(k) for k in cat_df["cat"]]
                values = cat_df["count"].tolist()
                fig = px.pie(names=labels, values=values, hole=0.45)
                fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>")
                style_dark_figure(fig, title=f"Proportion by {primary_cat}")
                charts.append({
                    "id": f"donut_{primary_cat}",
                    "title": f"Proportion by {primary_cat}",
                    "chart_type": "donut",
                    "figure": fig,
                    "description": f"Composition across top {len(labels)} categories.",
                    "metrics": [
                        {"label": "Categories", "value": f"{num_cats:,}"},
                        {"label": "Top Category", "value": f"{labels[0]} ({values[0]:,})"},
                    ],
                })
            else:
                top_10 = cat_df.head(10)
                fig = px.bar(top_10, x="count", y="cat", orientation="h")
                fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed([str(v) for v in top_10["cat"]]))))
                fig.update_xaxes(rangemode="tozero")
                fig.update_traces(marker_color="#0EA5E9", hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>")
                style_dark_figure(fig, title=f"Top {primary_cat} by Count", x_title="Count", y_title="")
                charts.append({
                    "id": f"bar_{primary_cat}",
                    "title": f"Top Categories: {primary_cat}",
                    "chart_type": "bar",
                    "figure": fig,
                    "description": f"Frequency ranking of top {len(top_10)} categories.",
                    "metrics": [
                        {"label": "Top Leader", "value": f"{top_10.iloc[0]['cat']}"},
                        {"label": "Total Analyzed", "value": f"{tot_count:,}"},
                    ],
                })
        else:
            # Fallback to pandas
            val_counts = df[primary_cat].value_counts().dropna()
            if len(val_counts) >= 2:
                num_cats = len(val_counts)
                total_sample_c = int(val_counts.sum())
                desc_suffix = f" ({sample_notice})" if sample_notice else ""
                if num_cats == 2:
                    labels = [str(k) for k in val_counts.index]
                    pcts = [round(v / total_sample_c * 100, 1) for v in val_counts.values]
                    two_df = pd.DataFrame({primary_cat: labels, "count": val_counts.values})
                    fig = px.bar(two_df, x="count", y=primary_cat, orientation="h")
                    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels))))
                    fig.update_xaxes(rangemode="tozero")
                    fig.update_traces(marker_color="#0EA5E9", hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>")
                    style_dark_figure(fig, title=f"Distribution: {primary_cat}", x_title="Count", y_title="")
                    charts.append({
                        "id": f"bar_{primary_cat}",
                        "title": f"Distribution: {primary_cat}",
                        "chart_type": "bar",
                        "figure": fig,
                        "description": f"Breakdown across {primary_cat}{desc_suffix}.",
                        "sample_label": sample_notice,
                        "metrics": [
                            {"label": f"{labels[0]} Share", "value": f"{pcts[0]}%"},
                            {"label": f"{labels[1]} Share", "value": f"{pcts[1]}%"},
                            {"label": "Sample Rows" if sample_notice else "Total Count", "value": f"{total_sample_c:,}"},
                        ],
                    })
                elif num_cats <= 7:
                    labels = [str(k) for k in val_counts.index]
                    values = val_counts.values.tolist()
                    fig = px.pie(names=labels, values=values, hole=0.45)
                    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>")
                    style_dark_figure(fig, title=f"Proportion by {primary_cat}")
                    charts.append({
                        "id": f"donut_{primary_cat}",
                        "title": f"Proportion by {primary_cat}",
                        "chart_type": "donut",
                        "figure": fig,
                        "description": f"Composition across top {len(labels)} categories{desc_suffix}.",
                        "sample_label": sample_notice,
                        "metrics": [
                            {"label": "Categories", "value": f"{len(val_counts):,}"},
                            {"label": "Top Category", "value": f"{labels[0]} ({values[0]:,})"},
                        ],
                    })
                else:
                    top_10 = val_counts.head(10).reset_index()
                    top_10.columns = [primary_cat, "count"]
                    fig = px.bar(top_10, x="count", y=primary_cat, orientation="h")
                    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed([str(v) for v in top_10[primary_cat]]))))
                    fig.update_xaxes(rangemode="tozero")
                    fig.update_traces(marker_color="#0EA5E9", hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>")
                    style_dark_figure(fig, title=f"Top {primary_cat} by Count", x_title="Count", y_title="")
                    charts.append({
                        "id": f"bar_{primary_cat}",
                        "title": f"Top Categories: {primary_cat}",
                        "chart_type": "bar",
                        "figure": fig,
                        "description": f"Frequency ranking of top {len(top_10)} categories{desc_suffix}.",
                        "sample_label": sample_notice,
                        "metrics": [
                            {"label": "Top Leader", "value": f"{top_10.iloc[0][primary_cat]}"},
                            {"label": "Sample Rows" if sample_notice else "Total Unique", "value": f"{total_sample_c:,}"},
                        ],
                    })

    # 2. Date/Time Trend (Aggregated by week or month)
    if date_cols and len(charts) < max_charts:
        date_col = date_cols[0]
        trend_df = None
        trunc_name = "Monthly"

        if agent is not None:
            sql_bounds = f'SELECT MIN("{date_col}") AS min_d, MAX("{date_col}") AS max_d FROM "{tbl_name}" WHERE "{date_col}" IS NOT NULL'
            bounds_res = _query_readonly_duckdb(agent, sql_bounds)
            if bounds_res is not None and not bounds_res.empty and pd.notna(bounds_res.iloc[0]["min_d"]) and pd.notna(bounds_res.iloc[0]["max_d"]):
                d_min = pd.to_datetime(bounds_res.iloc[0]["min_d"])
                d_max = pd.to_datetime(bounds_res.iloc[0]["max_d"])
                days_span = (d_max - d_min).days
                trunc_unit = "month" if days_span > 90 else "week"
                trunc_name = "Monthly" if trunc_unit == "month" else "Weekly"

                if num_cols:
                    metric_col = num_cols[0]
                    sql_trend = f'SELECT CAST(DATE_TRUNC(\'{trunc_unit}\', "{date_col}") AS DATE) AS period, SUM("{metric_col}") AS val, COUNT(*) AS count FROM "{tbl_name}" WHERE "{date_col}" IS NOT NULL GROUP BY period ORDER BY period LIMIT 500'
                    metric_label = f"Total {metric_col}"
                else:
                    sql_trend = f'SELECT CAST(DATE_TRUNC(\'{trunc_unit}\', "{date_col}") AS DATE) AS period, COUNT(*) AS val FROM "{tbl_name}" WHERE "{date_col}" IS NOT NULL GROUP BY period ORDER BY period LIMIT 500'
                    metric_label = "Record Count"

                trend_df = _query_readonly_duckdb(agent, sql_trend)

        if trend_df is not None and len(trend_df) >= 2:
            trend_df = trend_df.dropna(subset=["period", "val"]).sort_values("period")
            fig = px.line(trend_df, x="period", y="val", markers=len(trend_df) <= 60)
            fig.update_traces(line=dict(width=2.5, color="#06B6D4"), hovertemplate="<b>Date: %{x}</b><br>Value: %{y:,.2f}<extra></extra>")
            fig.update_yaxes(autorange=True, rangemode="normal")
            style_dark_figure(fig, title=f"{trunc_name} {metric_label} Trend ({date_col})", x_title="Date", y_title=metric_label)
            charts.append({
                "id": f"trend_{date_col}",
                "title": f"{trunc_name} Trend ({date_col})",
                "chart_type": "line",
                "figure": fig,
                "description": f"Chronological {trunc_name.lower()} trend across the full dataset.",
                "metrics": [
                    {"label": f"{trunc_name} Periods", "value": f"{len(trend_df):,}"},
                    {"label": "Peak Value", "value": f"{trend_df['val'].max():,.2f}" if trend_df['val'].max() > 10 else f"{trend_df['val'].max():,}"},
                ],
            })
        else:
            # Fallback to pandas
            safe_df = df.dropna(subset=[date_col]).copy()
            try:
                safe_df["_parsed_date"] = pd.to_datetime(safe_df[date_col], errors="coerce")
                safe_df = safe_df.dropna(subset=["_parsed_date"])
                if len(safe_df) >= 2:
                    d_min = safe_df["_parsed_date"].min()
                    d_max = safe_df["_parsed_date"].max()
                    freq = "M" if (d_max - d_min).days > 90 else "W"
                    trunc_name = "Monthly" if freq == "M" else "Weekly"
                    safe_df["_period"] = safe_df["_parsed_date"].dt.to_period(freq).dt.to_timestamp()
                    if num_cols:
                        metric_col = num_cols[0]
                        trend_df = safe_df.groupby("_period")[metric_col].agg("sum").reset_index()
                        trend_df.columns = ["date", "val"]
                        trend_df = trend_df.sort_values("date")
                        metric_label = f"Total {metric_col}"
                    else:
                        trend_df = safe_df.groupby("_period").size().reset_index(name="val")
                        trend_df.columns = ["date", "val"]
                        trend_df = trend_df.sort_values("date")
                        metric_label = "Record Count"

                    if len(trend_df) >= 2:
                        fig = px.line(trend_df, x="date", y="val", markers=len(trend_df) <= 60)
                        fig.update_traces(line=dict(width=2.5, color="#06B6D4"), hovertemplate="<b>Date: %{x}</b><br>Value: %{y:,.2f}<extra></extra>")
                        fig.update_yaxes(autorange=True, rangemode="normal")
                        style_dark_figure(fig, title=f"{trunc_name} {metric_label} Trend ({date_col})", x_title="Date", y_title=metric_label)
                        desc_suffix = f" ({sample_notice})" if sample_notice else ""
                        charts.append({
                            "id": f"trend_{date_col}",
                            "title": f"{trunc_name} Trend ({date_col})",
                            "chart_type": "line",
                            "figure": fig,
                            "description": f"Chronological {trunc_name.lower()} trend{desc_suffix}.",
                            "sample_label": sample_notice,
                            "metrics": [
                                {"label": f"{trunc_name} Periods", "value": f"{len(trend_df):,}"},
                                {"label": "Peak Value", "value": f"{trend_df['val'].max():,.2f}" if trend_df['val'].max() > 10 else f"{trend_df['val'].max():,}"},
                            ],
                        })
            except Exception as exc:
                logger.debug("Failed auto-generating datetime line chart: %s", exc)

    # 3. Numeric Column Distribution & Summary Metrics
    if num_cols and len(charts) < max_charts:
        primary_num = num_cols[0]
        summ_df = None
        if agent is not None:
            sql_summ = f'SELECT COUNT(*) AS total_n, AVG("{primary_num}") AS avg_v, MIN("{primary_num}") AS min_v, MAX("{primary_num}") AS max_v FROM "{tbl_name}" WHERE "{primary_num}" IS NOT NULL'
            summ_df = _query_readonly_duckdb(agent, sql_summ)

        num_series = df[primary_num].dropna()
        if len(num_series) >= 2 and num_series.nunique() > 1:
            if summ_df is not None and not summ_df.empty:
                mean_v = float(summ_df.iloc[0]["avg_v"]) if pd.notna(summ_df.iloc[0]["avg_v"]) else float(num_series.mean())
                min_v = float(summ_df.iloc[0]["min_v"]) if pd.notna(summ_df.iloc[0]["min_v"]) else float(num_series.min())
                max_v = float(summ_df.iloc[0]["max_v"]) if pd.notna(summ_df.iloc[0]["max_v"]) else float(num_series.max())
                total_n = int(summ_df.iloc[0]["total_n"]) if pd.notna(summ_df.iloc[0]["total_n"]) else len(num_series)
                metrics_list = [
                    {"label": "Mean (Full Dataset)", "value": f"{mean_v:,.2f}" if abs(mean_v) < 1e6 else f"{mean_v:,.0f}"},
                    {"label": "Min - Max", "value": f"{min_v:,.1f} to {max_v:,.1f}"},
                    {"label": "Total Analyzed", "value": f"{total_n:,}"},
                ]
                sample_lbl = None
            else:
                mean_v = float(num_series.mean())
                median_v = float(num_series.median())
                min_v = float(num_series.min())
                max_v = float(num_series.max())
                metrics_list = [
                    {"label": "Mean", "value": f"{mean_v:,.2f}" if abs(mean_v) < 1e6 else f"{mean_v:,.0f}"},
                    {"label": "Median", "value": f"{median_v:,.2f}" if abs(median_v) < 1e6 else f"{median_v:,.0f}"},
                    {"label": "Min - Max", "value": f"{min_v:,.1f} to {max_v:,.1f}"},
                ]
                sample_lbl = sample_notice

            hist_df = pd.DataFrame({primary_num: num_series})
            hist_sample = hist_df.sample(5000, random_state=42) if len(hist_df) > 5000 else hist_df

            fig = px.histogram(hist_sample, x=primary_num, nbins=25, marginal="box")
            fig.update_traces(marker_color="#06B6D4", marker_line_color="#0EA5E9", marker_line_width=1, opacity=0.85, hovertemplate=f"Range: %{{x}}<br>Count: %{{y:,}}<extra></extra>")
            fig.update_yaxes(rangemode="tozero")
            style_dark_figure(fig, title=f"Distribution: {primary_num}", x_title=primary_num, y_title="Frequency")
            charts.append({
                "id": f"hist_{primary_num}",
                "title": f"Distribution: {primary_num}",
                "chart_type": "histogram",
                "figure": fig,
                "description": f"Histogram and spread box plot showing distribution variance.",
                "sample_label": sample_lbl,
                "metrics": metrics_list,
            })

    # 4. Category + Numeric Value: Aggregated Bar Chart
    if cat_cols and num_cols and len(charts) < max_charts:
        agg_cat = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
        agg_num = num_cols[0]
        grp_df = None

        if agent is not None:
            sql_grp = f'SELECT "{agg_cat}" AS cat, AVG("{agg_num}") AS avg_v, COUNT(*) AS count FROM "{tbl_name}" WHERE "{agg_cat}" IS NOT NULL AND "{agg_num}" IS NOT NULL GROUP BY "{agg_cat}" ORDER BY avg_v DESC LIMIT 10'
            grp_df = _query_readonly_duckdb(agent, sql_grp)

        if grp_df is not None and len(grp_df) >= 2:
            fig = px.bar(grp_df, x="cat", y="avg_v")
            fig.update_yaxes(rangemode="tozero")
            fig.update_traces(marker_color="#10B981", hovertemplate=f"<b>%{{x}}</b><br>Average {agg_num}: %{{y:,.2f}}<extra></extra>")
            style_dark_figure(fig, title=f"Avg {agg_num} by {agg_cat}", x_title=agg_cat, y_title=f"Average {agg_num}")
            charts.append({
                "id": f"agg_bar_{agg_cat}_{agg_num}",
                "title": f"Average {agg_num} by {agg_cat}",
                "chart_type": "bar",
                "figure": fig,
                "description": f"Aggregated metric comparison across categories on full dataset.",
                "metrics": [
                    {"label": "Top Category", "value": str(grp_df.iloc[0]["cat"])},
                    {"label": "Highest Avg", "value": f"{grp_df.iloc[0]['avg_v']:,.2f}"},
                ],
            })
        else:
            # Fallback to pandas
            grouped = df.groupby(agg_cat)[agg_num].mean().dropna().sort_values(ascending=False).head(10).reset_index()
            if len(grouped) >= 2:
                fig = px.bar(grouped, x=agg_cat, y=agg_num)
                fig.update_yaxes(rangemode="tozero")
                fig.update_traces(marker_color="#10B981", hovertemplate=f"<b>%{{x}}</b><br>Average {agg_num}: %{{y:,.2f}}<extra></extra>")
                style_dark_figure(fig, title=f"Avg {agg_num} by {agg_cat}", x_title=agg_cat, y_title=f"Average {agg_num}")
                desc_suffix = f" ({sample_notice})" if sample_notice else ""
                charts.append({
                    "id": f"agg_bar_{agg_cat}_{agg_num}",
                    "title": f"Average {agg_num} by {agg_cat}",
                    "chart_type": "bar",
                    "figure": fig,
                    "description": f"Aggregated metric comparison across categories{desc_suffix}.",
                    "sample_label": sample_notice,
                    "metrics": [
                        {"label": "Top Category", "value": str(grouped.iloc[0][agg_cat])},
                        {"label": "Highest Avg", "value": f"{grouped.iloc[0][agg_num]:,.2f}"},
                    ],
                })

    # 5. Two Numeric Columns: Scatter Plot (Correlation)
    if len(num_cols) >= 2 and len(charts) < max_charts:
        x_num = num_cols[0]
        y_num = num_cols[1]
        pair_df = df[[x_num, y_num]].dropna()
        if len(pair_df) >= 5 and pair_df[x_num].nunique() > 1 and pair_df[y_num].nunique() > 1:
            pair_sample = pair_df.sample(1000, random_state=42) if len(pair_df) > 1000 else pair_df
            corr_val = pair_df[x_num].corr(pair_df[y_num])
            corr_str = f"{corr_val:.2f}" if pd.notna(corr_val) else "N/A"

            fig = px.scatter(pair_sample, x=x_num, y=y_num, opacity=0.75)
            fig.update_traces(marker=dict(size=7, color="#6366F1", line=dict(width=1, color="#A5B4FC")), hovertemplate=f"<b>{x_num}:</b> %{{x:,.2f}}<br><b>{y_num}:</b> %{{y:,.2f}}<extra></extra>")
            style_dark_figure(fig, title=f"{x_num} vs {y_num}", x_title=x_num, y_title=y_num)
            charts.append({
                "id": f"scatter_{x_num}_{y_num}",
                "title": f"Correlation: {x_num} vs {y_num}",
                "chart_type": "scatter",
                "figure": fig,
                "description": f"Scatter correlation between {x_num} and {y_num} (Pearson r = {corr_str}).",
                "sample_label": sample_notice,
                "metrics": [
                    {"label": "Correlation (r)", "value": corr_str},
                    {"label": "Points Plotted", "value": f"{len(pair_sample):,}"},
                ],
            })

    return charts[:max_charts]

    return charts[:max_charts]


def build_custom_dataset_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    agg_func: str = "sum",
    max_categories: int = 15,
    bins: int = 25,
    show_box: bool = True,
    orientation: str = "vertical",
) -> Tuple[Optional[go.Figure], Optional[List[Dict[str, str]]]]:
    """
    Build a customized Plotly figure based on user interaction in Explore Dataset.

    Returns:
        (figure, summary_metrics_list)
    """
    if df is None or df.empty or not x_col or x_col not in df.columns:
        return None, None

    chart_type = chart_type.lower()
    metrics: List[Dict[str, str]] = []

    try:
        if chart_type in ("bar", "bar chart"):
            if not y_col or y_col == "(Count of Records)":
                vc = df[x_col].value_counts().dropna().head(max_categories).reset_index()
                vc.columns = [x_col, "Count"]
                if orientation == "horizontal":
                    fig = px.bar(vc, x="Count", y=x_col, orientation="h")
                    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed([str(v) for v in vc[x_col]]))))
                else:
                    fig = px.bar(vc, x=x_col, y="Count")
                style_dark_figure(fig, title=f"Record Count by {x_col}", x_title="Count" if orientation == "horizontal" else x_col, y_title="" if orientation == "horizontal" else "Count")
                metrics = [{"label": "Total Categories", "value": f"{df[x_col].nunique():,}"}]
                return fig, metrics

            clean_df = df[[x_col, y_col] + ([color_col] if color_col and color_col in df.columns and color_col != x_col else [])].dropna()
            clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[y_col])

            if color_col and color_col in clean_df.columns and color_col != x_col:
                grp = clean_df.groupby([x_col, color_col])[y_col].agg(agg_func.lower()).reset_index()
                top_cats = clean_df.groupby(x_col)[y_col].sum().nlargest(max_categories).index
                grp = grp[grp[x_col].isin(top_cats)]
                fig = px.bar(grp, x=x_col, y=y_col, color=color_col, barmode="group")
                fig.update_yaxes(rangemode="tozero")
                style_dark_figure(fig, title=f"{agg_func.capitalize()} of {y_col} by {x_col} and {color_col}")
                return fig, None

            grp = clean_df.groupby(x_col)[y_col].agg(agg_func.lower()).reset_index()
            grp = grp.sort_values(y_col, ascending=False).head(max_categories)
            if orientation == "horizontal":
                fig = px.bar(grp, x=y_col, y=x_col, orientation="h")
                fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(reversed([str(v) for v in grp[x_col]]))))
                fig.update_xaxes(rangemode="tozero")
            else:
                fig = px.bar(grp, x=x_col, y=y_col)
                fig.update_yaxes(rangemode="tozero")
            style_dark_figure(fig, title=f"{agg_func.capitalize()} of {y_col} by {x_col}", x_title=y_col if orientation == "horizontal" else x_col, y_title="" if orientation == "horizontal" else f"{agg_func.capitalize()} of {y_col}")
            if len(grp) > 0:
                metrics = [
                    {"label": "Top Category", "value": str(grp.iloc[0][x_col])},
                    {"label": f"Top {agg_func.capitalize()}", "value": f"{grp.iloc[0][y_col]:,.2f}"},
                ]
            return fig, metrics

        elif chart_type in ("pie", "donut", "pie / donut", "donut / pie chart"):
            if not y_col or y_col == "(Count of Records)":
                vc = df[x_col].value_counts().dropna().head(max_categories)
                labels = [str(k) for k in vc.index]
                values = vc.values.tolist()
            else:
                clean_df = df[[x_col, y_col]].dropna()
                clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
                clean_df = clean_df.dropna(subset=[y_col])
                grp = clean_df.groupby(x_col)[y_col].agg(agg_func.lower()).sort_values(ascending=False).head(max_categories)
                labels = [str(k) for k in grp.index]
                values = grp.values.tolist()

            fig = px.pie(names=labels, values=values, hole=0.45)
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>Share: %{percent}<extra></extra>",
            )
            style_dark_figure(fig, title=f"Proportion by {x_col}")
            return fig, None

        elif chart_type in ("line", "line chart"):
            clean_df = df.dropna(subset=[x_col]).copy()
            is_date = False
            try:
                clean_df["_x_sort"] = pd.to_datetime(clean_df[x_col], errors="coerce")
                if clean_df["_x_sort"].notna().sum() > 0:
                    clean_df = clean_df.dropna(subset=["_x_sort"]).sort_values("_x_sort")
                    is_date = True
            except Exception:
                pass

            if not y_col or y_col == "(Count of Records)":
                grp = clean_df.groupby(x_col).size().reset_index(name="Count")
                if is_date:
                    grp["_date"] = pd.to_datetime(grp[x_col], errors="coerce")
                    grp = grp.sort_values("_date")
                fig = px.line(grp, x=x_col, y="Count", markers=len(grp) <= 60)
                fig.update_yaxes(autorange=True, rangemode="normal")
                style_dark_figure(fig, title=f"Count Trend across {x_col}")
                return fig, None

            clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[y_col])

            if color_col and color_col in clean_df.columns and color_col != x_col:
                grp = clean_df.groupby([x_col, color_col])[y_col].agg(agg_func.lower()).reset_index()
                fig = px.line(grp, x=x_col, y=y_col, color=color_col)
                fig.update_yaxes(autorange=True, rangemode="normal")
                style_dark_figure(fig, title=f"{y_col} by {x_col} & {color_col}")
                return fig, None

            grp = clean_df.groupby(x_col)[y_col].agg(agg_func.lower()).reset_index()
            if is_date:
                grp["_date"] = pd.to_datetime(grp[x_col], errors="coerce")
                grp = grp.sort_values("_date")
            fig = px.line(grp, x=x_col, y=y_col, markers=len(grp) <= 60)
            fig.update_yaxes(autorange=True, rangemode="normal")
            style_dark_figure(fig, title=f"{agg_func.capitalize()} of {y_col} across {x_col}")
            return fig, None

        elif chart_type in ("histogram", "distribution"):
            clean_series = pd.to_numeric(df[x_col], errors="coerce").dropna()
            if len(clean_series) < 2 or clean_series.nunique() <= 1:
                return None, None

            mean_v = float(clean_series.mean())
            median_v = float(clean_series.median())
            std_v = float(clean_series.std()) if len(clean_series) > 1 else 0.0
            min_v = float(clean_series.min())
            max_v = float(clean_series.max())

            hist_df = pd.DataFrame({x_col: clean_series})
            if len(hist_df) > 5000:
                hist_sample = hist_df.sample(5000, random_state=42)
            else:
                hist_sample = hist_df

            fig = px.histogram(
                hist_sample,
                x=x_col,
                nbins=bins,
                marginal="box" if show_box else None,
            )
            fig.update_traces(
                marker_color="#06B6D4",
                marker_line_color="#0EA5E9",
                marker_line_width=1,
                opacity=0.85,
                hovertemplate=f"Range: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
            )
            style_dark_figure(fig, title=f"Distribution of {x_col}", x_title=x_col, y_title="Frequency")
            metrics = [
                {"label": "Mean", "value": f"{mean_v:,.2f}" if abs(mean_v) < 1e6 else f"{mean_v:,.0f}"},
                {"label": "Median", "value": f"{median_v:,.2f}" if abs(median_v) < 1e6 else f"{median_v:,.0f}"},
                {"label": "Std Dev", "value": f"{std_v:,.2f}"},
                {"label": "Range", "value": f"{min_v:,.1f} - {max_v:,.1f}"},
            ]
            return fig, metrics

        elif chart_type in ("scatter", "scatter plot"):
            if not y_col or y_col not in df.columns:
                return None, None

            cols_to_keep = [x_col, y_col]
            has_color = color_col and color_col in df.columns and color_col not in (x_col, y_col)
            if has_color:
                cols_to_keep.append(color_col)

            clean_df = df[cols_to_keep].dropna()
            clean_df[x_col] = pd.to_numeric(clean_df[x_col], errors="coerce")
            clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
            clean_df = clean_df.dropna(subset=[x_col, y_col])

            if len(clean_df) > 1000:
                plot_sample = clean_df.sample(1000, random_state=42)
            else:
                plot_sample = clean_df

            fig = px.scatter(
                plot_sample,
                x=x_col,
                y=y_col,
                color=color_col if has_color else None,
                opacity=0.75,
            )
            fig.update_traces(
                marker=dict(size=7),
                hovertemplate=f"<b>{x_col}:</b> %{{x:,.2f}}<br><b>{y_col}:</b> %{{y:,.2f}}<extra></extra>",
            )
            corr_val = clean_df[x_col].corr(clean_df[y_col])
            corr_str = f"{corr_val:.2f}" if pd.notna(corr_val) else "N/A"
            style_dark_figure(fig, title=f"{x_col} vs {y_col}")
            metrics = [
                {"label": "Correlation (r)", "value": corr_str},
                {"label": "Points", "value": f"{len(plot_sample):,}"},
            ]
            return fig, metrics

        elif chart_type in ("grouped_bar", "grouped bar"):
            metrics_list = [c.strip() for c in (y_col or "").split(",") if c.strip() in df.columns]
            if not metrics_list:
                return None, None
            clean_df = df[[x_col] + metrics_list].dropna()
            for m in metrics_list:
                clean_df[m] = pd.to_numeric(clean_df[m], errors="coerce")
            clean_df = clean_df.dropna()
            grp = clean_df.groupby(x_col)[metrics_list].agg(agg_func.lower()).reset_index()
            grp = grp.head(max_categories)
            fig = px.bar(grp, x=x_col, y=metrics_list, barmode="group")
            fig.update_yaxes(rangemode="tozero")
            style_dark_figure(fig, title=f"Grouped Metrics by {x_col}")
            return fig, None

    except Exception as exc:
        logger.exception("Failed to build custom dataset figure: %s", exc)
        return None, None

    return None, None
