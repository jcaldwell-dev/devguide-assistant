from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.api.server import (
    create_api,
    get_application_services,
)
from app.services.openai_service import (
    ChatResult,
    ToolCallRecord,
)
from app.services.streaming_service import (
    ChatStreamEvent,
)


class FakeMemoryService:
    """
    Minimal memory replacement used by API tests.
    """

    def __init__(self) -> None:
        self.cleared_ids: list[str] = []
        self.existing_ids = {
            "existing-conversation",
        }
        self.turn_counts: dict[str, int] = {}

    def clear_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        if not conversation_id:
            raise ValueError(
                "conversation_id cannot be blank."
            )

        self.cleared_ids.append(conversation_id)
        self.turn_counts.pop(conversation_id, None)

        if conversation_id in self.existing_ids:
            self.existing_ids.remove(conversation_id)
            return True

        return False

    def get_turn_count(
        self,
        conversation_id: str,
    ) -> int:
        return self.turn_counts.get(
            conversation_id,
            1,
        )


class FakeMetricsService:
    """
    Minimal metrics replacement used by API tests.
    """

    def __init__(self) -> None:
        self.created_metrics: list[dict[str, Any]] = []
        self.recorded_metrics: list[dict[str, Any]] = []

    def create_turn_metric(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.created_metrics.append(kwargs)
        return kwargs

    def record_turn(
        self,
        metric: dict[str, Any],
    ) -> None:
        self.recorded_metrics.append(metric)


class FakeOpenAIService:
    """
    Predictable OpenAIService replacement used by endpoint tests.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def chat(
        self,
        conversation_id: str,
        user_message: str,
    ) -> ChatResult:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "user_message": user_message,
            }
        )

        if user_message == "raise value error":
            raise ValueError(
                "Controlled validation failure."
            )

        if user_message == "raise runtime error":
            raise RuntimeError(
                "Controlled model failure."
            )

        return ChatResult(
            conversation_id=conversation_id,
            model="test-model",
            response_id="response-test-001",
            text="Test assistant response.",
            tool_calls=[
                ToolCallRecord(
                    call_id="call-test-001",
                    tool_name="search_project_kb",
                    arguments={
                        "query": "Actor Component prefix",
                    },
                    success=True,
                )
            ],
            model_rounds=2,
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
        )


class FakeStreamingService:
    """
    Predictable streaming-service replacement used by endpoint tests.
    """

    def stream_chat(
        self,
        conversation_id: str,
        user_message: str,
    ):
        yield ChatStreamEvent(
            event="start",
            conversation_id=conversation_id,
            model="test-model",
        )

        yield ChatStreamEvent(
            event="text_delta",
            conversation_id=conversation_id,
            model="test-model",
            delta="Hello ",
        )

        yield ChatStreamEvent(
            event="text_delta",
            conversation_id=conversation_id,
            model="test-model",
            delta="stream.",
        )

        yield ChatStreamEvent(
            event="complete",
            conversation_id=conversation_id,
            model="test-model",
            text="Hello stream.",
            response_id="response-stream-1",
            model_rounds=1,
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
        )


class FakeApplicationServices:
    """
    Dependency graph compatible with the API route requirements.
    """

    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            openai_model="test-model"
        )

        self.openai_service = FakeOpenAIService()
        self.streaming_service = FakeStreamingService()
        self.memory = FakeMemoryService()
        self.metrics = FakeMetricsService()


def create_test_client() -> tuple[
    TestClient,
    FakeApplicationServices,
]:
    services = FakeApplicationServices()
    app = create_api(services=services)
    client = TestClient(app)

    return client, services


def test_health_endpoint() -> None:
    client, _ = create_test_client()

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "application": "DevGuide Assistant",
        "model": "test-model",
    }


def test_chat_endpoint_returns_structured_result() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "chat-001",
            "user_message": (
                "What prefix should an Actor Component use?"
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["conversation_id"] == "chat-001"
    assert body["model"] == "test-model"
    assert body["text"] == "Test assistant response."
    assert body["model_rounds"] == 2
    assert body["input_tokens"] == 100
    assert body["output_tokens"] == 25
    assert body["total_tokens"] == 125

    assert body["tool_calls"][0]["tool_name"] == (
        "search_project_kb"
    )

    assert services.openai_service.calls == [
        {
            "conversation_id": "chat-001",
            "user_message": (
                "What prefix should an Actor Component use?"
            ),
        }
    ]

    assert len(
        services.metrics.recorded_metrics
    ) == 1

    metric = services.metrics.recorded_metrics[0]

    assert metric["conversation_id"] == "chat-001"
    assert metric["model"] == "test-model"
    assert metric["input_tokens"] == 100
    assert metric["output_tokens"] == 25
    assert metric["total_tokens"] == 125
    assert metric["model_rounds"] == 2
    assert metric["memory_turns"] == 1
    assert metric["tool_calls"][0]["tool_name"] == (
        "search_project_kb"
    )


def test_chat_request_values_are_trimmed() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "  chat-001  ",
            "user_message": "  Hello  ",
        },
    )

    assert response.status_code == 200

    assert services.openai_service.calls[0] == {
        "conversation_id": "chat-001",
        "user_message": "Hello",
    }

    assert (
        services.metrics.recorded_metrics[0][
            "conversation_id"
        ]
        == "chat-001"
    )

    assert (
        services.metrics.recorded_metrics[0][
            "user_message"
        ]
        == "Hello"
    )


def test_blank_user_message_is_rejected() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "chat-001",
            "user_message": "   ",
        },
    )

    assert response.status_code == 422
    assert services.openai_service.calls == []
    assert services.metrics.recorded_metrics == []


def test_blank_conversation_id_is_rejected() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "   ",
            "user_message": "Hello",
        },
    )

    assert response.status_code == 422
    assert services.openai_service.calls == []
    assert services.metrics.recorded_metrics == []


def test_missing_user_message_is_rejected() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "chat-001",
        },
    )

    assert response.status_code == 422
    assert services.metrics.recorded_metrics == []


def test_invalid_json_body_is_rejected() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        content="{invalid-json",
        headers={
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 422
    assert services.metrics.recorded_metrics == []


def test_value_error_returns_bad_request() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "chat-001",
            "user_message": "raise value error",
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Controlled validation failure."
    )

    assert services.metrics.recorded_metrics == []


def test_runtime_error_returns_bad_gateway() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat/json",
        json={
            "conversation_id": "chat-001",
            "user_message": "raise runtime error",
        },
    )

    assert response.status_code == 502

    assert response.json()["detail"] == (
        "Controlled model failure."
    )

    assert services.metrics.recorded_metrics == []


def test_existing_conversation_can_be_cleared() -> None:
    client, services = create_test_client()

    response = client.delete(
        "/conversations/existing-conversation"
    )

    assert response.status_code == 200

    assert response.json() == {
        "conversation_id": "existing-conversation",
        "removed": True,
    }

    assert services.memory.cleared_ids == [
        "existing-conversation"
    ]


def test_unknown_conversation_clear_returns_false() -> None:
    client, _ = create_test_client()

    response = client.delete(
        "/conversations/missing-conversation"
    )

    assert response.status_code == 200

    assert response.json() == {
        "conversation_id": "missing-conversation",
        "removed": False,
    }


def test_openapi_schema_contains_chat_endpoint() -> None:
    client, _ = create_test_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "/chat" in schema["paths"]
    assert "post" in schema["paths"]["/chat"]

    assert "/chat/json" in schema["paths"]
    assert "post" in schema["paths"]["/chat/json"]

    assert "/chat/stream" not in schema["paths"]

    assert (
        schema["info"]["title"]
        == "DevGuide Assistant API"
    )


def test_stream_chat_endpoint_returns_sse() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat",
        json={
            "conversation_id": "stream-chat",
            "user_message": "Hello",
        },
    )

    assert response.status_code == 200

    assert response.headers[
        "content-type"
    ].startswith("text/event-stream")

    body = response.text

    assert "event: start" in body
    assert "event: text_delta" in body
    assert "Hello " in body
    assert "stream." in body
    assert "event: complete" in body

    assert len(
        services.metrics.recorded_metrics
    ) == 1

    metric = services.metrics.recorded_metrics[0]

    assert metric["conversation_id"] == "stream-chat"
    assert metric["assistant_text"] == "Hello stream."
    assert metric["input_tokens"] == 10
    assert metric["output_tokens"] == 2
    assert metric["total_tokens"] == 12
    assert metric["model_rounds"] == 1


def test_stream_endpoint_rejects_blank_message() -> None:
    client, services = create_test_client()

    response = client.post(
        "/chat",
        json={
            "conversation_id": "stream-chat",
            "user_message": " ",
        },
    )

    assert response.status_code == 422
    assert services.metrics.recorded_metrics == []


def test_root_serves_chat_interface() -> None:
    client, _ = create_test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DevGuide Assistant" in response.text
    assert 'id="chatForm"' in response.text


def test_static_javascript_is_available() -> None:
    client, _ = create_test_client()

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert 'fetch("/chat"' in response.text


def test_static_stylesheet_is_available() -> None:
    client, _ = create_test_client()

    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ".chat-messages" in response.text


def test_application_services_dependency_is_cached() -> None:
    get_application_services.cache_clear()

    first_services = get_application_services()
    second_services = get_application_services()

    assert first_services is second_services
    assert first_services.memory is second_services.memory
    assert first_services.metrics is second_services.metrics

    get_application_services.cache_clear()