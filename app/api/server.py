from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from app.api.models import (
    ChatRequest,
    ClearConversationResponse,
    HealthResponse,
)
from app.application import (
    ApplicationServices,
    create_application_services,
)
from app.services.openai_service import ChatResult
from app.services.streaming_service import (
    ChatStreamEvent,
)


STATIC_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "static"
)

INDEX_FILE = STATIC_DIRECTORY / "index.html"


@lru_cache
def get_application_services() -> ApplicationServices:
    """
    Returns one shared dependency graph.
    """

    return create_application_services()


ApplicationDependency = Annotated[
    ApplicationServices,
    Depends(get_application_services),
]


def format_sse_event(
    event: ChatStreamEvent,
) -> str:
    payload = event.model_dump_json(
        exclude_none=True
    )

    return (
        f"event: {event.event}\n"
        f"data: {payload}\n\n"
    )


def get_client_key(
    http_request: Request,
    conversation_id: str,
) -> str:
    """
    Creates a rate-limit key from client host and conversation.
    """

    client_host = (
        http_request.client.host
        if http_request.client is not None
        else "unknown-client"
    )

    return (
        f"{client_host}:"
        f"{conversation_id.strip()}"
    )


def enforce_rate_limit(
    http_request: Request,
    conversation_id: str,
    app_services: ApplicationServices,
) -> None:
    """
    Raises HTTP 429 when the request limit is exceeded.
    """

    rate_limiter = getattr(
        app_services,
        "rate_limiter",
        None,
    )

    if rate_limiter is None:
        return

    decision = rate_limiter.check(
        get_client_key(
            http_request,
            conversation_id,
        )
    )

    if decision.allowed:
        return

    retry_after = max(
        1,
        ceil(decision.retry_after_seconds),
    )

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            "Too many chatbot requests. "
            f"Try again in approximately "
            f"{retry_after} seconds."
        ),
        headers={
            "Retry-After": str(retry_after),
        },
    )


def redact_text(
    app_services: ApplicationServices,
    text: str,
) -> str:
    safety = getattr(
        app_services,
        "safety",
        None,
    )

    if safety is None:
        return text

    return safety.redact_text(text)


def record_metric(
    *,
    app_services: ApplicationServices,
    conversation_id: str,
    model: str,
    response_id: str | None,
    user_message: str,
    assistant_text: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    model_rounds: int,
    tool_calls: list[dict],
) -> None:
    """
    Records one redacted turn metric.
    """

    metric = (
        app_services.metrics.create_turn_metric(
            conversation_id=conversation_id,
            model=model,
            response_id=response_id,
            user_message=redact_text(
                app_services,
                user_message,
            ),
            assistant_text=redact_text(
                app_services,
                assistant_text,
            ),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_rounds=model_rounds,
            memory_turns=(
                app_services.memory.get_turn_count(
                    conversation_id
                )
            ),
            tool_calls=tool_calls,
        )
    )

    app_services.metrics.record_turn(metric)


def create_api(
    services: ApplicationServices | None = None,
) -> FastAPI:
    app = FastAPI(
        title="DevGuide Assistant API",
        description=(
            "Task-oriented game-development assistant with "
            "memory, tools, streaming, safety checks, rate "
            "limiting, and persistent metrics."
        ),
        version="1.4.0",
    )

    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIRECTORY),
        name="static",
    )

    if services is not None:
        app.dependency_overrides[
            get_application_services
        ] = lambda: services

    @app.get(
        "/",
        include_in_schema=False,
    )
    async def user_interface() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
    )
    async def health(
        app_services: ApplicationDependency,
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            application="DevGuide Assistant",
            model=app_services.settings.openai_model,
        )

    @app.post(
        "/chat/json",
        response_model=ChatResult,
        tags=["Chat"],
    )
    async def chat(
        http_request: Request,
        chat_request: ChatRequest,
        app_services: ApplicationDependency,
    ) -> ChatResult:
        enforce_rate_limit(
            http_request,
            chat_request.conversation_id,
            app_services,
        )

        started_at = perf_counter()

        safety = getattr(
            app_services,
            "safety",
            None,
        )

        safety_decision = (
            safety.evaluate(
                chat_request.user_message
            )
            if safety is not None
            else None
        )

        if (
            safety_decision is not None
            and not safety_decision.allowed
        ):
            refusal_text = (
                safety_decision.refusal_message
                or "The request cannot be completed."
            )

            result = ChatResult(
                conversation_id=(
                    chat_request.conversation_id
                ),
                model=(
                    app_services.settings.openai_model
                ),
                text=refusal_text,
                tool_calls=[],
                model_rounds=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )

            record_metric(
                app_services=app_services,
                conversation_id=(
                    result.conversation_id
                ),
                model=result.model,
                response_id=None,
                user_message=(
                    chat_request.user_message
                ),
                assistant_text=refusal_text,
                latency_ms=(
                    perf_counter() - started_at
                )
                * 1000,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model_rounds=0,
                tool_calls=[],
            )

            return result

        try:
            result = await run_in_threadpool(
                app_services.openai_service.chat,
                chat_request.conversation_id,
                chat_request.user_message,
            )

            record_metric(
                app_services=app_services,
                conversation_id=(
                    result.conversation_id
                ),
                model=result.model,
                response_id=result.response_id,
                user_message=(
                    chat_request.user_message
                ),
                assistant_text=result.text,
                latency_ms=(
                    perf_counter() - started_at
                )
                * 1000,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                model_rounds=result.model_rounds,
                tool_calls=[
                    tool_call.model_dump()
                    for tool_call
                    in result.tool_calls
                ],
            )

            return result

        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "OpenAI authentication failed."
                ),
            ) from error

        except RateLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "The model provider rejected the request "
                    "because of a rate limit or unavailable quota."
                ),
            ) from error

        except APIConnectionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The application could not connect to "
                    "the model provider."
                ),
            ) from error

        except APIStatusError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "The model provider returned an error "
                    f"with status {error.status_code}."
                ),
            ) from error

        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(error),
            ) from error

    @app.post(
        "/chat",
        tags=["Chat"],
        response_class=StreamingResponse,
    )
    @app.post(
        "/chat/stream",
        tags=["Chat"],
        response_class=StreamingResponse,
        include_in_schema=False,
    )
    async def stream_chat(
        http_request: Request,
        chat_request: ChatRequest,
        app_services: ApplicationDependency,
    ) -> StreamingResponse:
        enforce_rate_limit(
            http_request,
            chat_request.conversation_id,
            app_services,
        )

        safety = getattr(
            app_services,
            "safety",
            None,
        )

        safety_decision = (
            safety.evaluate(
                chat_request.user_message
            )
            if safety is not None
            else None
        )

        if (
            safety_decision is not None
            and not safety_decision.allowed
        ):
            refusal_text = (
                safety_decision.refusal_message
                or "The request cannot be completed."
            )

            def refusal_generator() -> Iterator[str]:
                started_at = perf_counter()

                start_event = ChatStreamEvent(
                    event="start",
                    conversation_id=(
                        chat_request.conversation_id
                    ),
                    model=(
                        app_services.settings
                        .openai_model
                    ),
                )

                delta_event = ChatStreamEvent(
                    event="text_delta",
                    conversation_id=(
                        chat_request.conversation_id
                    ),
                    model=(
                        app_services.settings
                        .openai_model
                    ),
                    delta=refusal_text,
                )

                complete_event = ChatStreamEvent(
                    event="complete",
                    conversation_id=(
                        chat_request.conversation_id
                    ),
                    model=(
                        app_services.settings
                        .openai_model
                    ),
                    text=refusal_text,
                    model_rounds=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                )

                yield format_sse_event(start_event)
                yield format_sse_event(delta_event)

                record_metric(
                    app_services=app_services,
                    conversation_id=(
                        chat_request.conversation_id
                    ),
                    model=(
                        app_services.settings
                        .openai_model
                    ),
                    response_id=None,
                    user_message=(
                        chat_request.user_message
                    ),
                    assistant_text=refusal_text,
                    latency_ms=(
                        perf_counter() - started_at
                    )
                    * 1000,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    model_rounds=0,
                    tool_calls=[],
                )

                yield format_sse_event(
                    complete_event
                )

            return StreamingResponse(
                refusal_generator(),
                media_type="text/event-stream",
            )

        def event_generator() -> Iterator[str]:
            started_at = perf_counter()

            try:
                for stream_event in (
                    app_services
                    .streaming_service
                    .stream_chat(
                        conversation_id=(
                            chat_request.conversation_id
                        ),
                        user_message=(
                            chat_request.user_message
                        ),
                    )
                ):
                    if stream_event.event == "complete":
                        record_metric(
                            app_services=app_services,
                            conversation_id=(
                                stream_event
                                .conversation_id
                            ),
                            model=stream_event.model,
                            response_id=(
                                stream_event.response_id
                            ),
                            user_message=(
                                chat_request.user_message
                            ),
                            assistant_text=(
                                stream_event.text or ""
                            ),
                            latency_ms=(
                                perf_counter() - started_at
                            )
                            * 1000,
                            input_tokens=(
                                stream_event.input_tokens
                            ),
                            output_tokens=(
                                stream_event.output_tokens
                            ),
                            total_tokens=(
                                stream_event.total_tokens
                            ),
                            model_rounds=(
                                stream_event.model_rounds
                                or 0
                            ),
                            tool_calls=[
                                tool_call.model_dump()
                                for tool_call
                                in stream_event.tool_calls
                            ],
                        )

                    yield format_sse_event(
                        stream_event
                    )

            except Exception as error:
                yield format_sse_event(
                    ChatStreamEvent(
                        event="error",
                        conversation_id=(
                            chat_request.conversation_id
                        ),
                        model=(
                            app_services.settings
                            .openai_model
                        ),
                        error_type=(
                            type(error).__name__
                        ),
                        error_message=str(error),
                    )
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete(
        "/conversations/{conversation_id}",
        response_model=ClearConversationResponse,
        tags=["Chat"],
    )
    async def clear_conversation(
        conversation_id: str,
        app_services: ApplicationDependency,
    ) -> ClearConversationResponse:
        try:
            cleaned_id = conversation_id.strip()

            removed = (
                app_services.memory.clear_conversation(
                    cleaned_id
                )
            )

            return ClearConversationResponse(
                conversation_id=cleaned_id,
                removed=removed,
            )

        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    return app


app = create_api()