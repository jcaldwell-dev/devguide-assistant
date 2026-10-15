from app.application import (
    create_application_services,
)
from app.config import Settings


def create_test_settings() -> Settings:
    return Settings(
        openai_api_key="fake-key-for-testing",
        openai_model="test-model",
        openai_input_cost_per_million=0.75,
        openai_output_cost_per_million=4.50,
        max_tool_rounds=2,
        max_memory_turns=5,
        results_directory="test-results",
        system_prompt="Test assistant prompt.",
        _env_file=None,
    )


def test_application_services_use_settings() -> None:
    settings = create_test_settings()

    services = create_application_services(settings)

    assert services.settings is settings
    assert services.openai_service.model == "test-model"
    assert services.streaming_service.model == "test-model"
    assert services.memory.max_turns == 5
    assert (
        services.memory.system_prompt
        == "Test assistant prompt."
    )


def test_application_contains_default_tools() -> None:
    services = create_application_services(
        create_test_settings()
    )

    assert services.registry.list_tool_names() == [
        "estimate_task",
        "search_project_kb",
    ]


def test_application_components_share_dependencies() -> None:
    services = create_application_services(
        create_test_settings()
    )

    assert services.client is not None
    assert services.memory is not None
    assert services.metrics is not None
    assert services.registry is not None
    assert services.openai_service is not None
    assert services.streaming_service is not None


def test_factory_does_not_make_api_request() -> None:
    services = create_application_services(
        create_test_settings()
    )

    assert services.memory.conversation_count() == 0
    assert services.metrics.load_turn_metrics() == []