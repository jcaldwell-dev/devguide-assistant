from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TaskEstimateInput(BaseModel):
    """
    Validated input for the deterministic task-estimation tool.
    """

    task_name: str = Field(
        min_length=1,
        max_length=100,
    )

    complexity: int = Field(
        ge=1,
        le=5,
        description="Technical complexity from 1 to 5.",
    )

    estimated_hours: float = Field(
        ge=0.5,
        le=500.0,
        description="Initial base estimate in hours.",
    )

    dependencies: int = Field(
        ge=0,
        le=10,
        description="Number of blocking tasks or systems.",
    )

    uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncertainty as a decimal from 0.0 to 1.0.",
    )

    @field_validator("task_name")
    @classmethod
    def task_name_must_not_be_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("task_name cannot be blank.")

        return cleaned_value


class TaskEstimateResult(BaseModel):
    """
    Structured result returned by the task-estimation tool.
    """

    task_name: str

    complexity: int
    base_hours: float
    dependency_count: int
    uncertainty: float

    estimated_min_hours: float
    estimated_max_hours: float

    risk_score: float
    risk_level: str
    review_priority: str

    explanation: str


def calculate_estimated_max_hours(
    base_hours: float,
    complexity: int,
    dependencies: int,
    uncertainty: float,
) -> float:
    """
    Calculates the upper effort estimate.

    Formula:
        base_hours * (1 + uncertainty)
        + dependencies * 0.5
        + complexity * 0.25
    """

    maximum_hours = (
        base_hours * (1.0 + uncertainty)
        + dependencies * 0.5
        + complexity * 0.25
    )

    return round(maximum_hours, 2)


def calculate_risk_score(
    complexity: int,
    dependencies: int,
    uncertainty: float,
) -> float:
    """
    Calculates a deterministic risk score from 0.0 to 10.0.

    Formula:
        complexity * 1.1
        + dependencies * 0.4
        + uncertainty * 3.0
    """

    raw_score = (
        complexity * 1.1
        + dependencies * 0.4
        + uncertainty * 3.0
    )

    capped_score = min(raw_score, 10.0)

    return round(capped_score, 2)


def classify_risk(risk_score: float) -> tuple[str, str]:
    """
    Converts the numeric risk score into a risk level and
    planning-review priority.
    """

    if risk_score < 3.0:
        return "Low", "Routine"

    if risk_score < 6.0:
        return "Moderate", "Planned"

    if risk_score < 8.0:
        return "High", "Early Review"

    return "Critical", "Immediate Review"


def estimate_task(
    task_name: str,
    complexity: int,
    estimated_hours: float,
    dependencies: int,
    uncertainty: float,
) -> TaskEstimateResult:
    """
    Validates the supplied values and produces a deterministic
    task estimate.
    """

    validated_input = TaskEstimateInput(
        task_name=task_name,
        complexity=complexity,
        estimated_hours=estimated_hours,
        dependencies=dependencies,
        uncertainty=uncertainty,
    )

    estimated_min_hours = round(
        validated_input.estimated_hours,
        2,
    )

    estimated_max_hours = calculate_estimated_max_hours(
        base_hours=validated_input.estimated_hours,
        complexity=validated_input.complexity,
        dependencies=validated_input.dependencies,
        uncertainty=validated_input.uncertainty,
    )

    risk_score = calculate_risk_score(
        complexity=validated_input.complexity,
        dependencies=validated_input.dependencies,
        uncertainty=validated_input.uncertainty,
    )

    risk_level, review_priority = classify_risk(risk_score)

    explanation = (
        f"{validated_input.task_name} has a {risk_level.lower()} "
        f"planning risk. The base estimate is "
        f"{estimated_min_hours:.2f} hours, with an upper estimate "
        f"of {estimated_max_hours:.2f} hours after accounting for "
        f"complexity, dependencies, and uncertainty."
    )

    return TaskEstimateResult(
        task_name=validated_input.task_name,
        complexity=validated_input.complexity,
        base_hours=estimated_min_hours,
        dependency_count=validated_input.dependencies,
        uncertainty=validated_input.uncertainty,
        estimated_min_hours=estimated_min_hours,
        estimated_max_hours=estimated_max_hours,
        risk_score=risk_score,
        risk_level=risk_level,
        review_priority=review_priority,
        explanation=explanation,
    )