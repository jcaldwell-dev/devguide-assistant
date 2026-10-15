from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.memory_service import MemoryService
from app.services.openai_service import OpenAIService
from app.tools.registry import create_default_registry


SYSTEM_PROMPT = (
    "You are a safe game-development production assistant."
)


class FakeOutputItem:
    def __init__(
        self,
        *,
        item_type: str,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self.type = item_type
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(
        self,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        data = {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }

        if exclude_none:
            return {
                key: value
                for key, value in data.items()
                if value is not None
            }

        return data


class FakeResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output_text: str = "",
        output: list[FakeOutputItem] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.id = response_id
        self.output_text = output_text
        self.output = output or []
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class FakeResponsesAPI:
    def __init__(
        self,
        responses: list[FakeResponse],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append(kwargs)

        if not self._responses:
            raise AssertionError(
                "Fake client received too many calls."
            )

        return self._responses.pop(0)


class FakeOpenAIClient:
    def __init__(
        self,
        responses: list[FakeResponse],
    ) -> None:
        self.responses = FakeResponsesAPI(responses)


def create_service(
    responses: list[FakeResponse],
    *,
    max_turns: int = 8,
    max_tool_rounds: int = 4,
) -> tuple[
    OpenAIService,
    MemoryService,
    FakeOpenAIClient,
]:
    memory = MemoryService(
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )

    client = FakeOpenAIClient(responses)

    service = OpenAIService(
        client=client,
        memory=memory,
        registry=create_default_registry(),
        model="test-model",
        max_tool_rounds=max_tool_rounds,
    )

    return service, memory, client


def test_chat_without_tool_call_returns_text() -> None:
    service, memory, client = create_service(
        [
            FakeResponse(
                response_id="response-1",
                output_text="Hello from the assistant.",
                input_tokens=10,
                output_tokens=5,
            )
        ]
    )

    result = service.chat(
        conversation_id="chat-1",
        user_message="Hello",
    )

    assert result.text == "Hello from the assistant."
    assert result.response_id == "response-1"
    assert result.model_rounds == 1
    assert result.tool_calls == []
    assert result.total_tokens == 15
    assert memory.get_turn_count("chat-1") == 1
    assert len(client.responses.calls) == 1


def test_chat_executes_kb_tool_and_continues() -> None:
    tool_arguments = json.dumps(
        {
            "query": "Actor Component prefix",
            "category": "naming",
            "max_results": 3,
        }
    )

    service, memory, client = create_service(
        [
            FakeResponse(
                response_id="response-tool",
                output=[
                    FakeOutputItem(
                        item_type="function_call",
                        call_id="call-kb-1",
                        name="search_project_kb",
                        arguments=tool_arguments,
                    )
                ],
                input_tokens=20,
                output_tokens=8,
            ),
            FakeResponse(
                response_id="response-final",
                output_text=(
                    "Actor Component Blueprints use "
                    "the BPC_ prefix."
                ),
                input_tokens=30,
                output_tokens=10,
            ),
        ]
    )

    result = service.chat(
        conversation_id="chat-1",
        user_message=(
            "What prefix should an Actor Component use?"
        ),
    )

    assert result.model_rounds == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == (
        "search_project_kb"
    )
    assert result.tool_calls[0].success is True
    assert "BPC_" in result.text
    assert result.total_tokens == 68
    assert memory.get_turn_count("chat-1") == 1
    assert len(client.responses.calls) == 2

    second_input = client.responses.calls[1]["input"]

    assert any(
        item.get("type") == "function_call_output"
        for item in second_input
        if isinstance(item, dict)
    )


def test_task_estimator_executes_through_registry() -> None:
    service, _, _ = create_service(
        [
            FakeResponse(
                response_id="response-tool",
                output=[
                    FakeOutputItem(
                        item_type="function_call",
                        call_id="call-estimate-1",
                        name="estimate_task",
                        arguments=json.dumps(
                            {
                                "task_name": "Add vehicles",
                                "complexity": 3,
                                "estimated_hours": 6.0,
                                "dependencies": 2,
                                "uncertainty": 0.25,
                            }
                        ),
                    )
                ],
            ),
            FakeResponse(
                response_id="response-final",
                output_text=(
                    "The task has moderate planning risk."
                ),
            ),
        ]
    )

    result = service.chat(
        conversation_id="estimate-chat",
        user_message="Estimate the vehicle task.",
    )

    assert result.tool_calls[0].success is True
    assert result.tool_calls[0].tool_name == (
        "estimate_task"
    )
    assert "moderate" in result.text.lower()


def test_prior_conversation_text_is_sent_to_model() -> None:
    service, memory, client = create_service(
        [
            FakeResponse(
                response_id="response-1",
                output_text="Recorded.",
            ),
            FakeResponse(
                response_id="response-2",
                output_text=(
                    "The approved vehicles are Sedan, "
                    "Hybrid, and Truck."
                ),
            ),
        ]
    )

    service.chat(
        conversation_id="memory-chat",
        user_message=(
            "The approved vehicles are Sedan, "
            "Hybrid, and Truck."
        ),
    )

    service.chat(
        conversation_id="memory-chat",
        user_message="Which vehicles were approved?",
    )

    second_request_input = client.responses.calls[1][
        "input"
    ]

    assert {
        "role": "user",
        "content": (
            "The approved vehicles are Sedan, "
            "Hybrid, and Truck."
        ),
    } in second_request_input

    assert {
        "role": "assistant",
        "content": "Recorded.",
    } in second_request_input

    assert memory.get_turn_count("memory-chat") == 2


def test_invalid_json_tool_arguments_are_controlled() -> None:
    service, _, client = create_service(
        [
            FakeResponse(
                response_id="response-tool",
                output=[
                    FakeOutputItem(
                        item_type="function_call",
                        call_id="bad-call",
                        name="estimate_task",
                        arguments="{not valid json",
                    )
                ],
            ),
            FakeResponse(
                response_id="response-final",
                output_text=(
                    "The tool arguments were invalid."
                ),
            ),
        ]
    )

    result = service.chat(
        conversation_id="bad-json",
        user_message="Estimate this task.",
    )

    assert result.tool_calls[0].success is False
    assert result.tool_calls[0].error_type == (
        "argument_parse_error"
    )

    second_input = client.responses.calls[1]["input"]

    function_output = next(
        item
        for item in second_input
        if isinstance(item, dict)
        and item.get("type")
        == "function_call_output"
    )

    assert "argument_parse_error" in (
        function_output["output"]
    )


def test_unknown_tool_is_returned_as_tool_error() -> None:
    service, _, _ = create_service(
        [
            FakeResponse(
                response_id="response-tool",
                output=[
                    FakeOutputItem(
                        item_type="function_call",
                        call_id="unknown-call",
                        name="missing_tool",
                        arguments="{}",
                    )
                ],
            ),
            FakeResponse(
                response_id="response-final",
                output_text="The requested tool is unavailable.",
            ),
        ]
    )

    result = service.chat(
        conversation_id="unknown-tool",
        user_message="Use the missing tool.",
    )

    assert result.tool_calls[0].success is False
    assert result.tool_calls[0].error_type == (
        "unknown_tool"
    )


def test_blank_user_message_is_rejected() -> None:
    service, _, _ = create_service([])

    with pytest.raises(
        ValueError,
        match="user_message cannot be blank",
    ):
        service.chat(
            conversation_id="chat-1",
            user_message="   ",
        )


def test_blank_model_is_rejected() -> None:
    memory = MemoryService(
        system_prompt=SYSTEM_PROMPT,
    )

    with pytest.raises(
        ValueError,
        match="model cannot be blank",
    ):
        OpenAIService(
            client=FakeOpenAIClient([]),
            memory=memory,
            registry=create_default_registry(),
            model=" ",
        )


def test_missing_final_text_raises_runtime_error() -> None:
    service, memory, _ = create_service(
        [
            FakeResponse(
                response_id="empty-response",
                output_text="",
                output=[],
            )
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="no final text",
    ):
        service.chat(
            conversation_id="empty-chat",
            user_message="Hello",
        )

    assert memory.get_turn_count("empty-chat") == 0


def test_tool_round_limit_is_enforced() -> None:
    repeated_call = FakeOutputItem(
        item_type="function_call",
        call_id="repeated-call",
        name="search_project_kb",
        arguments=json.dumps(
            {
                "query": "Blueprint",
            }
        ),
    )

    service, memory, _ = create_service(
        [
            FakeResponse(
                response_id="round-1",
                output=[repeated_call],
            ),
            FakeResponse(
                response_id="round-2",
                output=[repeated_call],
            ),
        ],
        max_tool_rounds=1,
    )

    with pytest.raises(
        RuntimeError,
        match="Maximum tool-call rounds exceeded",
    ):
        service.chat(
            conversation_id="loop-chat",
            user_message="Keep searching.",
        )

    assert memory.get_turn_count("loop-chat") == 0