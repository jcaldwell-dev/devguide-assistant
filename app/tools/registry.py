from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.tools.kb_search import (
    KnowledgeSearchInput,
    search_project_kb,
)
from app.tools.task_estimator import (
    TaskEstimateInput,
    estimate_task,
)


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """
    Internal definition of one callable tool.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler


class ToolExecutionResult(BaseModel):
    """
    Structured result produced whenever the registry executes a tool.
    """

    success: bool
    tool_name: str
    output: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


class ToolRegistry:
    """
    Stores tool definitions, exposes their input schemas, validates
    arguments, and dispatches tool executions.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """
        Registers a tool definition.

        Raises:
            ValueError:
                If the tool name is blank or already registered.
        """

        cleaned_name = definition.name.strip()

        if not cleaned_name:
            raise ValueError("Tool name cannot be blank.")

        if cleaned_name in self._tools:
            raise ValueError(
                f"Tool '{cleaned_name}' is already registered."
            )

        self._tools[cleaned_name] = definition

    def has_tool(self, tool_name: str) -> bool:
        """
        Returns True when a tool is registered.
        """

        return tool_name in self._tools

    def get_tool(self, tool_name: str) -> ToolDefinition:
        """
        Returns one registered tool definition.

        Raises:
            KeyError:
                If no matching tool is registered.
        """

        try:
            return self._tools[tool_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown tool: '{tool_name}'."
            ) from error

    def list_tool_names(self) -> list[str]:
        """
        Returns registered tool names in deterministic order.
        """

        return sorted(self._tools)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Returns neutral function schemas for all registered tools.

        The future model service will adapt these schemas to the
        exact API request format being used.
        """

        schemas: list[dict[str, Any]] = []

        for tool_name in self.list_tool_names():
            definition = self._tools[tool_name]

            schemas.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": (
                        definition.input_model.model_json_schema()
                    ),
                }
            )

        return schemas

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        Validates arguments and executes a registered tool.

        Tool errors are returned as structured data rather than
        escaping into the future API layer.
        """

        if not self.has_tool(tool_name):
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error_type="unknown_tool",
                error_message=f"Unknown tool: '{tool_name}'.",
            )

        definition = self.get_tool(tool_name)

        try:
            validated_input = definition.input_model.model_validate(
                arguments
            )
        except ValidationError as error:
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error_type="validation_error",
                error_message=str(error),
            )

        try:
            result = definition.handler(
                **validated_input.model_dump()
            )
        except Exception as error:
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error_type="execution_error",
                error_message=str(error),
            )

        if isinstance(result, BaseModel):
            output = result.model_dump()
        elif isinstance(result, dict):
            output = result
        else:
            output = {"result": result}

        return ToolExecutionResult(
            success=True,
            tool_name=tool_name,
            output=output,
        )


def create_default_registry() -> ToolRegistry:
    """
    Creates the application's standard tool registry.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_project_kb",
            description=(
                "Search the local game-development project knowledge base. "
                "Use this tool for project naming conventions, Blueprint "
                "practices, testing standards, organization rules, and "
                "project-specific guidance. For naming-prefix questions, "
                "search with category 'naming' or omit the category. "
                "Do not answer from general Unreal conventions when the "
                "knowledge base contains project-specific guidance."
            ),
            input_model=KnowledgeSearchInput,
            handler=search_project_kb,
        )
    )

    registry.register(
        ToolDefinition(
            name="estimate_task",
            description=(
                "Calculate a deterministic development-task estimate, "
                "including an effort range, planning-risk score, risk "
                "level, and review priority."
            ),
            input_model=TaskEstimateInput,
            handler=estimate_task,
        )
    )

    return registry


DEFAULT_TOOL_REGISTRY = create_default_registry()