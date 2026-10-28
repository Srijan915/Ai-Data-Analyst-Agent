"""Deterministic first-pass answerability gate for analytical questions."""
from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from app.models.schemas import CatalogProfile, DatasetProfile


class Answerability(StrEnum):
    ANSWERABLE = "ANSWERABLE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNANSWERABLE = "UNANSWERABLE"


class AnswerabilityResult(BaseModel):
    outcome: Answerability
    message: str
    missing_terms: list[str] = []


_STOPWORDS = {"a", "an", "and", "are", "by", "for", "from", "how", "in", "is", "of", "show", "the", "to", "what", "with"}
_ANALYTIC_WORDS = {"average", "count", "highest", "lowest", "max", "maximum", "mean", "min", "minimum", "records", "sum", "total", "top"}


def assess_answerability(question: str, profile: DatasetProfile | CatalogProfile) -> AnswerabilityResult:
    """Refuse absent data and flag ambiguous multi-table requests pre-LLM.

    This deliberately stays conservative: it only declares a question
    unanswerable for explicit column-like terms that match nothing, and asks
    for clarification before an unconfirmed multi-table join can be planned.
    """
    if not isinstance(question, str) or not question.strip():
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="What would you like to analyze?")
    if isinstance(profile, CatalogProfile) and not profile.datasets:
        return AnswerabilityResult(outcome=Answerability.UNANSWERABLE, message="No datasets are available. Upload a CSV first.")
    columns = []
    table_count = 1
    confirmed = []
    if isinstance(profile, CatalogProfile):
        table_count = len(profile.datasets)
        columns = [column.name for dataset in profile.datasets for column in dataset.profile.columns]
        confirmed = [r for r in profile.relationships if r.status == "confirmed"]
    else:
        columns = profile.column_names
    folded = {column.casefold(): column for column in columns}
    column_terms = {
        token.rstrip("s")
        for column in columns
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", column.casefold())
    }
    quoted = re.findall(r'["`\[]([^"`\]]+)["`\]]', question)
    missing = [term for term in quoted if term.casefold() not in folded]
    if missing:
        return AnswerabilityResult(outcome=Answerability.UNANSWERABLE, message="The requested column is not in the active dataset(s): " + ", ".join(missing) + ".", missing_terms=missing)
    words = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", question.casefold()) if word not in _STOPWORDS]
    recognized = {word for word in words if word in folded or word.rstrip("s") in column_terms or word in _ANALYTIC_WORDS}
    if question.casefold().lstrip().startswith("show "):
        recognized.add("show")
    if not recognized:
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="Please name a metric, column, or analytical question to answer.")
    mentions = sum(1 for dataset in getattr(profile, "datasets", []) if dataset.table_name.casefold() in question.casefold() or dataset.filename.casefold() in question.casefold())
    if table_count > 1 and (mentions > 1 or "join" in question.casefold()) and not confirmed:
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="Which datasets and join key should I use? No relationship has been confirmed yet.")
    return AnswerabilityResult(outcome=Answerability.ANSWERABLE, message="The request can be planned against the active schema.")
