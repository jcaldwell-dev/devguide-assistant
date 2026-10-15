from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
)


def test_default_registry_contains_expected_tools() -> None:
    registry = create_default_registry()

    assert registry.list_tool_names() == [
        "estimate_task",
        "search_project_kb",
    ]


def test_registry_reports_registered_tool() -> None:
    registry = create_default_registry()

    assert registry.has_tool("search_project_kb") is True
    assert registry.has_tool("missing_tool") is False


def test_registry_returns_tool_definition() -> None:
    registry = create_default_registry()

    definition = registry.get_tool("estimate_task")

    assert definition.name == "estimate_task"
    assert callable(definition.handler)


def test_get_unknown_tool_raises_key_error() -> None:
    registry = create_default_registry()

    with pytest.raises(
        KeyError,
        match="Unknown tool",
    ):
        registry.get_tool("missing_tool")


def test_tool_schemas_include_required_metadata() -> None:
    registry = create_default_registry()

    schemas = registry.get_tool_schemas()

    assert len(schemas) == 2

    estimator_schema = next(
        schema
        for schema in schemas
        if schema["name"] == "estimate_task"
    )

    assert estimator_schema["description"]
    assert estimator_schema["parameters"]["type"] == "object"

    properties = estimator_schema["parameters"]["properties"]

    assert "task_name" in properties
    assert "complexity" in properties
    assert "estimated_hours" in properties
    assert "dependencies" in properties
    assert "uncertainty" in properties


def test_registry_executes_kb_search() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="search_project_kb",
        arguments={
            "query": "Actor Component Blueprint prefix",
            "category": "naming",
            "max_results": 3,
        },
    )

    assert result.success is True
    assert result.error_type is None
    assert result.output is not None

    matches = result.output["matches"]

    assert len(matches) > 0
    assert matches[0]["id"] == "naming-002"
    assert "BPC_" in matches[0]["content"]


def test_registry_executes_task_estimator() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="estimate_task",
        arguments={
            "task_name": "Add three vehicle types",
            "complexity": 3,
            "estimated_hours": 6.0,
            "dependencies": 2,
            "uncertainty": 0.25,
        },
    )

    assert result.success is True
    assert result.error_type is None
    assert result.output is not None
    assert result.output["risk_score"] == 4.85
    assert result.output["risk_level"] == "Moderate"
    assert result.output["estimated_max_hours"] == 9.25


def test_unknown_tool_returns_structured_error() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="delete_the_internet",
        arguments={},
    )

    assert result.success is False
    assert result.output is None
    assert result.error_type == "unknown_tool"
    assert "Unknown tool" in result.error_message


def test_invalid_tool_arguments_return_validation_error() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="estimate_task",
        arguments={
            "task_name": "Invalid task",
            "complexity": 99,
            "estimated_hours": 5.0,
            "dependencies": 1,
            "uncertainty": 0.2,
        },
    )

    assert result.success is False
    assert result.output is None
    assert result.error_type == "validation_error"
    assert result.error_message is not None


def test_missing_required_argument_returns_validation_error() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="estimate_task",
        arguments={
            "task_name": "Missing complexity",
            "estimated_hours": 5.0,
            "dependencies": 1,
            "uncertainty": 0.2,
        },
    )

    assert result.success is False
    assert result.error_type == "validation_error"


def test_empty_kb_query_returns_validation_error() -> None:
    registry = create_default_registry()

    result = registry.execute(
        tool_name="search_project_kb",
        arguments={
            "query": "",
        },
    )

    assert result.success is False
    assert result.error_type == "validation_error"


def test_duplicate_tool_registration_is_rejected() -> None:
    registry = create_default_registry()
    existing_definition = registry.get_tool("estimate_task")

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(existing_definition)


class DemoInput(BaseModel):
    value: int = Field(ge=0)


def demo_handler(value: int) -> dict[str, Any]:
    return {
        "doubled": value * 2,
    }


def test_custom_tool_can_be_registered_and_executed() -> None:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="double_value",
            description="Doubles a nonnegative integer.",
            input_model=DemoInput,
            handler=demo_handler,
        )
    )

    result = registry.execute(
        tool_name="double_value",
        arguments={
            "value": 7,
        },
    )

    assert result.success is True
    assert result.output == {
        "doubled": 14,
    }


def exploding_handler(value: int) -> dict[str, Any]:
    raise RuntimeError("Controlled test failure")


def test_handler_exception_returns_execution_error() -> None:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="exploding_tool",
            description="Raises an error for testing.",
            input_model=DemoInput,
            handler=exploding_handler,
        )
    )

    result = registry.execute(
        tool_name="exploding_tool",
        arguments={
            "value": 1,
        },
    )

    assert result.success is False
    assert result.output is None
    assert result.error_type == "execution_error"
    assert result.error_message == "Controlled test failure"
