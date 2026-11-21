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
_ANALYTIC_WORDS = {"average", "count", "delete", "highest", "lowest", "max", "maximum", "mean", "min", "minimum", "records", "rows", "sum", "total", "top"}


def _missing_data_concepts(question: str, column_terms: set[str]) -> list[str]:
    """Identify common data requirements that the active schema cannot meet.

    This is intentionally a small semantic guard, not an attempt to turn
    natural language into SQL.  It catches requests that otherwise look
    analytical merely because they include words such as ``average`` or
    ``highest`` but require data the file plainly does not contain.
    """
    text = question.casefold()
    requirements: list[tuple[bool, str, set[str]]] = [
        (
            bool(re.search(r"\b(?:daily|weekly|monthly|quarterly|yearly)\b.*\btrend\b|\btrend\b.*\b(?:date|time)\b", text)),
            "date or time",
            {"date", "time", "timestamp"},
        ),
        ("profit" in text or "margin" in text, "profit, margin, or cost", {"profit", "margin", "cost"}),
        (bool(re.search(r"\bcosts?\b", text)), "cost", {"cost"}),
        (bool(re.search(r"\be-?mails?\b", text)), "email address", {"email", "mail"}),
        (bool(re.search(r"\b(?:return|returns|returned|refund|refunded)\b", text)), "return or refund", {"return", "refund"}),
        (bool(re.search(r"\b(?:customer )?names?\b", text)) and not bool(re.search(r"\b(?:item|product)\s+names?\b", text)), "customer name", {"name"}),
        (bool(re.search(r"\bdelivery\s+time\b|\bdelivered\s+late\b", text)), "delivery date or time", {"delivery", "date", "time"}),
        (bool(re.search(r"\bwritten\s+(?:customer\s+)?reviews?\b", text)), "written review text", {"comment", "feedback", "text"}),
        (bool(re.search(r"\b(?:inventory|stock)\b", text)), "inventory or stock", {"inventory", "stock"}),
        (bool(re.search(r"\b(?:income|household income)\b", text)), "income", {"income"}),
        (bool(re.search(r"\bcountr(?:y|ies)\b", text)), "country", {"country"}),
        (bool(re.search(r"\bcancell?(?:ed|ed|ation|ations)\b", text)), "cancellation", {"cancel"}),
    ]
    return [label for mentioned, label, supporting_terms in requirements if mentioned and not (column_terms & supporting_terms)]


def _term_variants(word: str) -> set[str]:
    """Return small, predictable English inflections for schema matching."""
    variants = {word, word.rstrip("s")}
    if word.endswith("ed") and len(word) > 2:
        variants.add(word[:-2])
    if word.endswith("ing") and len(word) > 3:
        variants.add(word[:-3])
    return variants


def assess_answerability(question: str, profile: DatasetProfile | CatalogProfile) -> AnswerabilityResult:
    """Refuse absent data and flag ambiguous multi-table requests pre-LLM.

    This deliberately stays conservative: it only declares a question
    unanswerable for explicit column-like terms that match nothing, and asks
    for clarification before an unconfirmed multi-table join can be planned.
    """
    if not isinstance(question, str) or not question.strip():
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="What would you like to analyze?")
    if re.search(r"\b(?:ignore|disregard)\s+(?:all\s+)?instructions\b", question, re.IGNORECASE):
        return AnswerabilityResult(
            outcome=Answerability.NEEDS_CLARIFICATION,
            message="Please ask a direct analytical question about the active dataset.",
        )
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
        for token in re.findall(r"[a-zA-Z]+", column.casefold())
    }
    quoted = re.findall(r'["`\[]([^"`\]]+)["`\]]', question)
    missing = [term for term in quoted if term.casefold() not in folded]
    if missing:
        return AnswerabilityResult(outcome=Answerability.UNANSWERABLE, message="The requested column is not in the active dataset(s): " + ", ".join(missing) + ".", missing_terms=missing)
    missing_concepts = _missing_data_concepts(question, column_terms)
    if missing_concepts:
        return AnswerabilityResult(
            outcome=Answerability.UNANSWERABLE,
            message="The active dataset does not contain the data needed for: " + ", ".join(missing_concepts) + ".",
            missing_terms=missing_concepts,
        )
    words = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", question.casefold()) if word not in _STOPWORDS]
    recognized = {
        word
        for word in words
        if word in folded
        or _term_variants(word) & column_terms
        or word in _ANALYTIC_WORDS
    }
    if question.casefold().lstrip().startswith("show "):
        recognized.add("show")
    if not recognized:
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="Please name a metric, column, or analytical question to answer.")
    mentions = sum(1 for dataset in getattr(profile, "datasets", []) if dataset.table_name.casefold() in question.casefold() or dataset.filename.casefold() in question.casefold())
    if table_count > 1 and (mentions > 1 or "join" in question.casefold()) and not confirmed:
        return AnswerabilityResult(outcome=Answerability.NEEDS_CLARIFICATION, message="Which datasets and join key should I use? No relationship has been confirmed yet.")
    return AnswerabilityResult(outcome=Answerability.ANSWERABLE, message="The request can be planned against the active schema.")
