from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# Resolve the project root regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_PATH = PROJECT_ROOT / "data" / "project_kb.json"


class KnowledgeRecord(BaseModel):
    """
    Represents one entry loaded from the local project knowledge base.
    """

    id: str
    title: str
    category: str
    keywords: list[str]
    content: str


class KnowledgeMatch(BaseModel):
    """
    Represents one matching knowledge-base record and its calculated score.
    """

    id: str
    title: str
    category: str
    content: str
    score: int
    matched_terms: list[str]


class KnowledgeSearchResult(BaseModel):
    """
    Structured result returned by the knowledge-base search tool.
    """

    query: str
    category: str | None
    matches: list[KnowledgeMatch]
    total_matches: int


def normalize_text(text: str) -> str:
    """
    Converts text to lowercase and replaces punctuation with spaces.

    Example:
        "Actor-Component Blueprint!" becomes
        "actor component blueprint"
    """

    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
    return " ".join(normalized.split())


def tokenize(text: str) -> set[str]:
    """
    Converts normalized text into a set of searchable terms.
    """

    normalized = normalize_text(text)
    return set(normalized.split())


def load_knowledge_base(
    kb_path: Path = DEFAULT_KB_PATH,
) -> list[KnowledgeRecord]:
    """
    Loads and validates knowledge-base records from a JSON file.

    Raises:
        FileNotFoundError:
            When the knowledge-base file does not exist.

        ValueError:
            When the JSON file is invalid or does not contain a list.
    """

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge-base file was not found: {kb_path}"
        )

    try:
        with kb_path.open("r", encoding="utf-8") as file:
            raw_data: Any = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Knowledge-base file contains invalid JSON: {kb_path}"
        ) from error

    if not isinstance(raw_data, list):
        raise ValueError(
            "Knowledge-base JSON must contain a list of records."
        )

    return [
        KnowledgeRecord.model_validate(record)
        for record in raw_data
    ]


def calculate_match_score(
    query: str,
    record: KnowledgeRecord,
) -> tuple[int, list[str]]:
    """
    Calculates a deterministic match score.

    Weighting:
        Exact full-query phrase in title:    6 points
        Exact full-query phrase in keyword:  5 points
        Individual title-term match:         3 points
        Individual keyword-term match:       2 points
        Individual content-term match:       1 point

    Returns:
        A tuple containing:
            - total score
            - sorted unique matched terms
    """

    normalized_query = normalize_text(query)
    query_terms = tokenize(query)

    normalized_title = normalize_text(record.title)
    title_terms = tokenize(record.title)

    normalized_keywords = [
        normalize_text(keyword)
        for keyword in record.keywords
    ]

    keyword_terms: set[str] = set()
    for keyword in record.keywords:
        keyword_terms.update(tokenize(keyword))

    content_terms = tokenize(record.content)

    score = 0
    matched_terms: set[str] = set()

    # Reward a complete phrase match in the title.
    if normalized_query and normalized_query in normalized_title:
        score += 6
        matched_terms.update(query_terms)

    # Reward a complete phrase match in a keyword.
    if normalized_query and any(
        normalized_query in keyword
        for keyword in normalized_keywords
    ):
        score += 5
        matched_terms.update(query_terms)

    for term in query_terms:
        if term in title_terms:
            score += 3
            matched_terms.add(term)

        if term in keyword_terms:
            score += 2
            matched_terms.add(term)

        if term in content_terms:
            score += 1
            matched_terms.add(term)

    return score, sorted(matched_terms)


def _search_records(
    query: str,
    records: list[KnowledgeRecord],
    category: str | None,
) -> list[KnowledgeMatch]:
    normalized_category = (
        normalize_text(category)
        if category is not None
        else None
    )

    matches: list[KnowledgeMatch] = []

    for record in records:
        if normalized_category is not None:
            record_category = normalize_text(record.category)

            if record_category != normalized_category:
                continue

        score, matched_terms = calculate_match_score(
            query=query,
            record=record,
        )

        if score == 0:
            continue

        matches.append(
            KnowledgeMatch(
                id=record.id,
                title=record.title,
                category=record.category,
                content=record.content,
                score=score,
                matched_terms=matched_terms,
            )
        )

    matches.sort(
        key=lambda match: (
            -match.score,
            match.title.lower(),
            match.id,
        )
    )

    return matches


def _has_sufficient_query_coverage(
    query: str,
    matches: list[KnowledgeMatch],
    minimum_coverage: float = 0.5,
) -> bool:
    """
    Returns True when the highest-ranked result matches enough of
    the meaningful query terms to trust the category-filtered search.
    """

    if not matches:
        return False

    query_terms = tokenize(query)

    if not query_terms:
        return False


    best_match_terms = set(matches[0].matched_terms)


    coverage = len(
        query_terms.intersection(best_match_terms)
    ) / len(query_terms)

    return coverage >= minimum_coverage


def search_project_kb(
    query: str,
    category: str | None = None,
    max_results: int = 3,
    kb_path: Path = DEFAULT_KB_PATH,
) -> KnowledgeSearchResult:
    """
    Searches the local project knowledge base.

    Args:
        query:
            Natural-language search query.

        category:
            Optional category filter such as "naming" or "testing".

        max_results:
            Maximum number of matches to return.

        kb_path:
            Path to the JSON knowledge-base file.

    Returns:
        A validated KnowledgeSearchResult.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    records = load_knowledge_base(kb_path)

    normalized_category = (
        normalize_text(category)
        if category is not None
        else None
    )

    matches = _search_records(
        query=cleaned_query,
        records=records,
        category=category,
    )

    if (
            category is not None
            and not _has_sufficient_query_coverage(
        query=cleaned_query,
        matches=matches,
    )
    ):
        matches = _search_records(
            query=cleaned_query,
            records=records,
            category=None,
        )

    limited_matches = matches[:max_results]

    return KnowledgeSearchResult(
        query=cleaned_query,
        category=category,
        matches=limited_matches,
        total_matches=len(matches),
    )

class KnowledgeSearchInput(BaseModel):
    """
    Validated input accepted by the knowledge-base search tool.
    """

    query: str = Field(
        min_length=1,
        max_length=500,
        description="Natural-language query for the project knowledge base.",
    )

    category: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Optional exact knowledge-base category filter. "
            "Omit this field unless the user explicitly requests "
            "a category. For naming or prefix questions, use "
            "'naming' or omit the category."
        ),
    )

    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of matching records to return.",
    )