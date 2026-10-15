from pathlib import Path

import pytest

from app.tools.kb_search import (
    DEFAULT_KB_PATH,
    load_knowledge_base,
    search_project_kb,
)


def test_knowledge_base_loads_expected_records() -> None:
    records = load_knowledge_base()

    assert len(records) == 5
    assert records[0].id == "naming-001"
    assert records[0].title == "Actor Blueprint Naming"


def test_actor_component_query_returns_bpc_prefix() -> None:
    result = search_project_kb(
        query="actor component prefix",
    )

    assert result.total_matches > 0
    assert len(result.matches) > 0

    best_match = result.matches[0]

    assert best_match.id == "naming-002"
    assert "BPC_" in best_match.content
    assert best_match.score > 0


def test_category_filter_limits_results() -> None:
    result = search_project_kb(
        query="feature validation testing",
        category="testing",
    )

    assert result.total_matches == 1
    assert result.matches[0].category == "testing"
    assert result.matches[0].id == "testing-001"


def test_unknown_query_returns_no_matches() -> None:
    result = search_project_kb(
        query="quantum navigation asset policy",
    )

    assert result.total_matches == 0
    assert result.matches == []


def test_empty_query_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        search_project_kb(query="   ")


def test_missing_knowledge_base_raises_file_error(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing_kb.json"

    with pytest.raises(FileNotFoundError):
        load_knowledge_base(missing_path)


def test_max_results_limits_returned_matches() -> None:
    result = search_project_kb(
        query="blueprint naming prefix",
        max_results=2,
    )

    assert result.total_matches >= 2
    assert len(result.matches) == 2


def test_invalid_max_results_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="max_results must be at least 1",
    ):
        search_project_kb(
            query="blueprint",
            max_results=0,
        )


def test_default_knowledge_base_path_exists() -> None:
    assert DEFAULT_KB_PATH.exists()
    assert DEFAULT_KB_PATH.name == "project_kb.json"


def test_incorrect_category_falls_back_to_all_records() -> None:
    result = search_project_kb(
        query="Actor Component Blueprint prefix",
        category="blueprint",
    )

    assert result.total_matches > 0
    assert result.matches[0].id == "naming-002"
    assert "BPC_" in result.matches[0].content