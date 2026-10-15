import pytest
from pydantic import ValidationError

from app.tools.task_estimator import (
    calculate_estimated_max_hours,
    calculate_risk_score,
    classify_risk,
    estimate_task,
)


def test_task_estimate_returns_expected_values() -> None:
    result = estimate_task(
        task_name="Add three vehicle types",
        complexity=3,
        estimated_hours=6.0,
        dependencies=2,
        uncertainty=0.25,
    )

    assert result.task_name == "Add three vehicle types"
    assert result.base_hours == 6.0
    assert result.estimated_min_hours == 6.0
    assert result.estimated_max_hours == 9.25
    assert result.risk_score == 4.85
    assert result.risk_level == "Moderate"
    assert result.review_priority == "Planned"


def test_maximum_hours_formula() -> None:
    result = calculate_estimated_max_hours(
        base_hours=8.0,
        complexity=4,
        dependencies=3,
        uncertainty=0.20,
    )

    assert result == 12.1


def test_risk_score_formula() -> None:
    result = calculate_risk_score(
        complexity=4,
        dependencies=3,
        uncertainty=0.20,
    )

    assert result == 6.2


@pytest.mark.parametrize(
    ("score", "expected_level", "expected_priority"),
    [
        (2.99, "Low", "Routine"),
        (3.0, "Moderate", "Planned"),
        (5.99, "Moderate", "Planned"),
        (6.0, "High", "Early Review"),
        (7.99, "High", "Early Review"),
        (8.0, "Critical", "Immediate Review"),
        (10.0, "Critical", "Immediate Review"),
    ],
)
def test_risk_classification_thresholds(
    score: float,
    expected_level: str,
    expected_priority: str,
) -> None:
    risk_level, review_priority = classify_risk(score)

    assert risk_level == expected_level
    assert review_priority == expected_priority


def test_risk_score_is_capped_at_ten() -> None:
    result = calculate_risk_score(
        complexity=5,
        dependencies=10,
        uncertainty=1.0,
    )

    assert result == 10.0


def test_task_name_whitespace_is_removed() -> None:
    result = estimate_task(
        task_name="  Implement save system  ",
        complexity=2,
        estimated_hours=10.0,
        dependencies=1,
        uncertainty=0.10,
    )

    assert result.task_name == "Implement save system"


def test_blank_task_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="   ",
            complexity=2,
            estimated_hours=5.0,
            dependencies=1,
            uncertainty=0.10,
        )


def test_complexity_below_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="Invalid complexity",
            complexity=0,
            estimated_hours=5.0,
            dependencies=1,
            uncertainty=0.10,
        )


def test_complexity_above_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="Invalid complexity",
            complexity=6,
            estimated_hours=5.0,
            dependencies=1,
            uncertainty=0.10,
        )


def test_negative_dependencies_are_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="Invalid dependencies",
            complexity=3,
            estimated_hours=5.0,
            dependencies=-1,
            uncertainty=0.10,
        )


def test_uncertainty_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="Invalid uncertainty",
            complexity=3,
            estimated_hours=5.0,
            dependencies=1,
            uncertainty=1.1,
        )


def test_estimated_hours_below_minimum_are_rejected() -> None:
    with pytest.raises(ValidationError):
        estimate_task(
            task_name="Invalid hours",
            complexity=3,
            estimated_hours=0.0,
            dependencies=1,
            uncertainty=0.10,
        )


def test_explanation_contains_estimate_information() -> None:
    result = estimate_task(
        task_name="Create inventory screen",
        complexity=3,
        estimated_hours=7.0,
        dependencies=2,
        uncertainty=0.20,
    )

    assert "Create inventory screen" in result.explanation
    assert "base estimate" in result.explanation
    assert "upper estimate" in result.explanation