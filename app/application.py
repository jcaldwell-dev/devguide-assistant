from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from app.config import Settings, get_settings
from app.services.memory_service import MemoryService
from app.services.metrics_service import MetricsService
from app.services.openai_service import OpenAIService
from app.services.rate_limit_service import (
    RateLimitService,
)
from app.services.safety_service import SafetyService
from app.services.streaming_service import (
    OpenAIStreamingService,
)
from app.tools.registry import (
    ToolRegistry,
    create_default_registry,
)


@dataclass(frozen=True)
class ApplicationServices:
    """
    Fully constructed application dependencies.
    """

    settings: Settings
    client: OpenAI

    memory: MemoryService
    metrics: MetricsService
    safety: SafetyService
    rate_limiter: RateLimitService

    registry: ToolRegistry
    openai_service: OpenAIService
    streaming_service: OpenAIStreamingService


def create_application_services(
    settings: Settings | None = None,
) -> ApplicationServices:
    """
    Constructs the complete dependency graph.
    """

    resolved_settings = settings or get_settings()

    client = OpenAI(
        api_key=(
            resolved_settings
            .openai_api_key
            .get_secret_value()
        )
    )

    memory = MemoryService(
        system_prompt=resolved_settings.system_prompt,
        max_turns=resolved_settings.max_memory_turns,
    )

    metrics = MetricsService(
        results_directory=(
            resolved_settings.results_directory
        ),
        input_cost_per_million=(
            resolved_settings
            .openai_input_cost_per_million
        ),
        output_cost_per_million=(
            resolved_settings
            .openai_output_cost_per_million
        ),
    )

    safety = SafetyService()

    rate_limiter = RateLimitService(
        max_requests=(
            resolved_settings.rate_limit_requests
        ),
        window_seconds=(
            resolved_settings
            .rate_limit_window_seconds
        ),
    )

    registry = create_default_registry()

    openai_service = OpenAIService(
        client=client,
        memory=memory,
        registry=registry,
        model=resolved_settings.openai_model,
        max_tool_rounds=(
            resolved_settings.max_tool_rounds
        ),
    )

    streaming_service = OpenAIStreamingService(
        client=client,
        memory=memory,
        registry=registry,
        model=resolved_settings.openai_model,
        max_tool_rounds=(
            resolved_settings.max_tool_rounds
        ),
    )

    return ApplicationServices(
        settings=resolved_settings,
        client=client,
        memory=memory,
        metrics=metrics,
        safety=safety,
        rate_limiter=rate_limiter,
        registry=registry,
        openai_service=openai_service,
        streaming_service=streaming_service,
    )