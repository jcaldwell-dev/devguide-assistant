from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.memory_service import (
    ChatMessage,
    MemoryService,
)
from app.services.openai_service import ToolCallRecord
from app.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
)


StreamEventType = Literal[
    "start",
    "tool_call",
    "text_delta",
    "complete",
    "error",
]


class ChatStreamEvent(BaseModel):
    """
    One structured event emitted during a streamed chatbot turn.
    """

    event: StreamEventType

    conversation_id: str
    model: str

    delta: str | None = None
    text: str | None = None

    response_id: str | None = None

    tool_call: ToolCallRecord | None = None
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list
    )

    model_rounds: int | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    error_type: str | None = None
    error_message: str | None = None


class OpenAIStreamingService:
    """
    Coordinates Responses API streaming, tool execution, and memory.

    Tool-selection rounds may complete without visible text. The
    assistant's final natural-language response is emitted through
    response.output_text.delta events as the model generates it.
    """

    def __init__(
        self,
        client: Any,
        memory: MemoryService,
        registry: ToolRegistry,
        model: str,
        max_tool_rounds: int = 4,
    ) -> None:
        cleaned_model = model.strip()

        if not cleaned_model:
            raise ValueError("model cannot be blank.")

        if max_tool_rounds < 1:
            raise ValueError(
                "max_tool_rounds must be at least 1."
            )

        self._client = client
        self._memory = memory
        self._registry = registry
        self._model = cleaned_model
        self._max_tool_rounds = max_tool_rounds

    @property
    def model(self) -> str:
        return self._model

    def stream_chat(
        self,
        conversation_id: str,
        user_message: str,
    ) -> Iterator[ChatStreamEvent]:
        """
        Processes one complete turn and emits structured stream events.

        Memory is updated only after a complete final assistant answer
        has been received.
        """

        cleaned_id = conversation_id.strip()
        cleaned_message = user_message.strip()

        if not cleaned_id:
            raise ValueError(
                "conversation_id cannot be blank."
            )

        if not cleaned_message:
            raise ValueError(
                "user_message cannot be blank."
            )

        running_input: list[Any] = [
            *self._build_prior_input(cleaned_id),
            {
                "role": "user",
                "content": cleaned_message,
            },
        ]

        memory_turn: list[ChatMessage] = [
            ChatMessage(
                role="user",
                content=cleaned_message,
            )
        ]

        tools = self._build_response_tools()

        tool_records: list[ToolCallRecord] = []

        total_input_tokens = 0
        total_output_tokens = 0

        latest_response_id: str | None = None

        yield ChatStreamEvent(
            event="start",
            conversation_id=cleaned_id,
            model=self._model,
        )

        for round_number in range(
            1,
            self._max_tool_rounds + 2,
        ):
            completed_response: Any | None = None
            visible_text_parts: list[str] = []

            stream = self._client.responses.create(
                model=self._model,
                instructions=self._memory.system_prompt,
                input=running_input,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                stream=True,
            )

            for api_event in stream:
                event_type = getattr(
                    api_event,
                    "type",
                    None,
                )

                if event_type == "response.output_text.delta":
                    delta = str(
                        getattr(api_event, "delta", "")
                        or ""
                    )

                    if not delta:
                        continue

                    visible_text_parts.append(delta)

                    yield ChatStreamEvent(
                        event="text_delta",
                        conversation_id=cleaned_id,
                        model=self._model,
                        delta=delta,
                    )

                elif event_type == "response.completed":
                    completed_response = getattr(
                        api_event,
                        "response",
                        None,
                    )

            if completed_response is None:
                raise RuntimeError(
                    "The streamed response ended without "
                    "a response.completed event."
                )

            latest_response_id = getattr(
                completed_response,
                "id",
                None,
            )

            input_tokens, output_tokens = (
                self._extract_usage(
                    completed_response
                )
            )

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            response_output = list(
                getattr(
                    completed_response,
                    "output",
                    [],
                )
                or []
            )

            function_calls = [
                item
                for item in response_output
                if getattr(item, "type", None)
                == "function_call"
            ]

            if not function_calls:
                final_text = "".join(
                    visible_text_parts
                ).strip()

                if not final_text:
                    final_text = self._extract_output_text(
                        completed_response
                    )

                if not final_text:
                    raise RuntimeError(
                        "The model returned no final text."
                    )

                memory_turn.append(
                    ChatMessage(
                        role="assistant",
                        content=final_text,
                    )
                )

                self._memory.add_turn(
                    conversation_id=cleaned_id,
                    messages=memory_turn,
                )

                yield ChatStreamEvent(
                    event="complete",
                    conversation_id=cleaned_id,
                    model=self._model,
                    response_id=latest_response_id,
                    text=final_text,
                    tool_calls=tool_records,
                    model_rounds=round_number,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=(
                        total_input_tokens
                        + total_output_tokens
                    ),
                )

                return

            if round_number > self._max_tool_rounds:
                raise RuntimeError(
                    "Maximum tool-call rounds exceeded."
                )

            running_input.extend(
                self._serialize_output_items(
                    response_output
                )
            )

            memory_turn.append(
                self._create_assistant_tool_message(
                    function_calls
                )
            )

            for function_call in function_calls:
                (
                    tool_record,
                    tool_message,
                    function_output,
                ) = self._execute_function_call(
                    function_call
                )

                tool_records.append(tool_record)
                memory_turn.append(tool_message)
                running_input.append(function_output)

                yield ChatStreamEvent(
                    event="tool_call",
                    conversation_id=cleaned_id,
                    model=self._model,
                    tool_call=tool_record,
                )

        raise RuntimeError(
            "The model did not produce a completed response."
        )

    def _build_prior_input(
        self,
        conversation_id: str,
    ) -> list[dict[str, str]]:
        """
        Converts retained text messages into Responses API input.
        """

        prior_messages = self._memory.get_messages(
            conversation_id
        )

        input_items: list[dict[str, str]] = []

        for message in prior_messages:
            if message.role not in {
                "user",
                "assistant",
            }:
                continue

            if not message.content:
                continue

            input_items.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        return input_items

    def _build_response_tools(
        self,
    ) -> list[dict[str, Any]]:
        """
        Converts neutral registry schemas into Responses tools.
        """

        return [
            {
                "type": "function",
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            }
            for schema in self._registry.get_tool_schemas()
        ]

    def _execute_function_call(
        self,
        function_call: Any,
    ) -> tuple[
        ToolCallRecord,
        ChatMessage,
        dict[str, Any],
    ]:
        call_id = str(function_call.call_id)
        tool_name = str(function_call.name)

        try:
            arguments = json.loads(
                function_call.arguments or "{}"
            )

            if not isinstance(arguments, dict):
                raise ValueError(
                    "Tool arguments must decode to an object."
                )

            execution_result = self._registry.execute(
                tool_name=tool_name,
                arguments=arguments,
            )

        except (
            json.JSONDecodeError,
            ValueError,
        ) as error:
            arguments = {}

            execution_result = ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                error_type="argument_parse_error",
                error_message=str(error),
            )

        output_text = (
            execution_result.model_dump_json()
        )

        tool_record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            success=execution_result.success,
            error_type=execution_result.error_type,
        )

        memory_message = ChatMessage(
            role="tool",
            name=tool_name,
            tool_call_id=call_id,
            content=output_text,
        )

        function_output = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_text,
        }

        return (
            tool_record,
            memory_message,
            function_output,
        )

    @staticmethod
    def _create_assistant_tool_message(
        function_calls: list[Any],
    ) -> ChatMessage:
        tool_calls = []

        for function_call in function_calls:
            tool_calls.append(
                {
                    "id": str(
                        function_call.call_id
                    ),
                    "type": "function",
                    "function": {
                        "name": str(
                            function_call.name
                        ),
                        "arguments": str(
                            function_call.arguments
                        ),
                    },
                }
            )

        return ChatMessage(
            role="assistant",
            tool_calls=tool_calls,
        )

    @staticmethod
    def _serialize_output_items(
        output_items: list[Any],
    ) -> list[dict[str, Any]]:
        serialized_items: list[dict[str, Any]] = []

        for item in output_items:
            if hasattr(item, "model_dump"):
                serialized_items.append(
                    item.model_dump(
                        exclude_none=True
                    )
                )

            elif isinstance(item, dict):
                serialized_items.append(
                    dict(item)
                )

            else:
                raise TypeError(
                    "Unsupported response output item."
                )

        return serialized_items

    @staticmethod
    def _extract_output_text(
        response: Any,
    ) -> str:
        output_text = getattr(
            response,
            "output_text",
            "",
        )

        return str(
            output_text or ""
        ).strip()

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int, int]:
        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None:
            return 0, 0

        input_tokens = int(
            getattr(
                usage,
                "input_tokens",
                0,
            )
            or 0
        )

        output_tokens = int(
            getattr(
                usage,
                "output_tokens",
                0,
            )
            or 0
        )

        return input_tokens, output_tokens