from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.memory_service import MemoryService
from app.services.streaming_service import (
    OpenAIStreamingService,
)
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
        values = {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }

        if exclude_none:
            return {
                key: value
                for key, value in values.items()
                if value is not None
            }

        return values


class FakeCompletedResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output: list[FakeOutputItem] | None = None,
        output_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.id = response_id
        self.output = output or []
        self.output_text = output_text
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class FakeAPIEvent:
    def __init__(
        self,
        *,
        event_type: str,
        delta: str | None = None,
        response: FakeCompletedResponse | None = None,
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.response = response


class FakeResponsesAPI:
    def __init__(
        self,
        streams: list[list[FakeAPIEvent]],
    ) -> None:
        self._streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        **kwargs: Any,
    ) -> list[FakeAPIEvent]:
        self.calls.append(kwargs)

        if not self._streams:
            raise AssertionError(
                "Fake streaming client received too many calls."
            )

        return self._streams.pop(0)


class FakeOpenAIClient:
    def __init__(
        self,
        streams: list[list[FakeAPIEvent]],
    ) -> None:
        self.responses = FakeResponsesAPI(
            streams
        )


def create_service(
    streams: list[list[FakeAPIEvent]],
) -> tuple[
    OpenAIStreamingService,
    MemoryService,
    FakeOpenAIClient,
]:
    memory = MemoryService(
        system_prompt=SYSTEM_PROMPT,
        max_turns=8,
    )

    client = FakeOpenAIClient(streams)

    service = OpenAIStreamingService(
        client=client,
        memory=memory,
        registry=create_default_registry(),
        model="test-model",
        max_tool_rounds=4,
    )

    return service, memory, client


def test_stream_without_tool_emits_text_deltas() -> None:
    completed = FakeCompletedResponse(
        response_id="response-1",
        output_text="Hello there.",
        input_tokens=10,
        output_tokens=3,
    )

    service, memory, client = create_service(
        [
            [
                FakeAPIEvent(
                    event_type="response.output_text.delta",
                    delta="Hello ",
                ),
                FakeAPIEvent(
                    event_type="response.output_text.delta",
                    delta="there.",
                ),
                FakeAPIEvent(
                    event_type="response.completed",
                    response=completed,
                ),
            ]
        ]
    )

    events = list(
        service.stream_chat(
            conversation_id="chat-1",
            user_message="Hello",
        )
    )

    assert events[0].event == "start"

    deltas = [
        event.delta
        for event in events
        if event.event == "text_delta"
    ]

    assert deltas == [
        "Hello ",
        "there.",
    ]

    complete = events[-1]

    assert complete.event == "complete"
    assert complete.text == "Hello there."
    assert complete.total_tokens == 13
    assert memory.get_turn_count("chat-1") == 1

    assert client.responses.calls[0]["stream"] is True


def test_stream_executes_tool_then_streams_answer() -> None:
    function_call = FakeOutputItem(
        item_type="function_call",
        call_id="call-1",
        name="search_project_kb",
        arguments=json.dumps(
            {
                "query": "Actor Component prefix",
                "category": "naming",
                "max_results": 3,
            }
        ),
    )

    tool_response = FakeCompletedResponse(
        response_id="response-tool",
        output=[function_call],
        input_tokens=20,
        output_tokens=5,
    )

    final_response = FakeCompletedResponse(
        response_id="response-final",
        output_text="Use BPC_.",
        input_tokens=30,
        output_tokens=4,
    )

    service, memory, client = create_service(
        [
            [
                FakeAPIEvent(
                    event_type="response.completed",
                    response=tool_response,
                ),
            ],
            [
                FakeAPIEvent(
                    event_type="response.output_text.delta",
                    delta="Use ",
                ),
                FakeAPIEvent(
                    event_type="response.output_text.delta",
                    delta="BPC_.",
                ),
                FakeAPIEvent(
                    event_type="response.completed",
                    response=final_response,
                ),
            ],
        ]
    )

    events = list(
        service.stream_chat(
            conversation_id="tool-chat",
            user_message=(
                "What prefix should an Actor "
                "Component use?"
            ),
        )
    )

    tool_events = [
        event
        for event in events
        if event.event == "tool_call"
    ]

    assert len(tool_events) == 1
    assert (
        tool_events[0].tool_call.tool_name
        == "search_project_kb"
    )
    assert tool_events[0].tool_call.success is True

    complete = events[-1]

    assert complete.text == "Use BPC_."
    assert complete.model_rounds == 2
    assert complete.total_tokens == 59
    assert memory.get_turn_count("tool-chat") == 1
    assert len(client.responses.calls) == 2

    second_input = client.responses.calls[1]["input"]

    assert any(
        isinstance(item, dict)
        and item.get("type")
        == "function_call_output"
        for item in second_input
    )


def test_stream_rejects_blank_message() -> None:
    service, _, _ = create_service([])

    with pytest.raises(
        ValueError,
        match="user_message cannot be blank",
    ):
        list(
            service.stream_chat(
                conversation_id="chat-1",
                user_message=" ",
            )
        )


def test_stream_requires_completed_event() -> None:
    service, memory, _ = create_service(
        [
            [
                FakeAPIEvent(
                    event_type="response.output_text.delta",
                    delta="Partial",
                )
            ]
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="response.completed",
    ):
        list(
            service.stream_chat(
                conversation_id="chat-1",
                user_message="Hello",
            )
        )

    assert memory.get_turn_count("chat-1") == 0