from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


TEST_KEY = "test-api-key-not-real"


def create_settings(
    **overrides: object,
) -> Settings:
    values = {
        "openai_api_key": TEST_KEY,
        "_env_file": None,
    }

    values.update(overrides)

    return Settings(**values)


def test_default_settings_values() -> None:
    settings = create_settings()

    assert settings.openai_model == "gpt-5.4-mini"
    assert settings.max_tool_rounds == 4
    assert settings.max_memory_turns == 8
    assert settings.system_prompt


def test_settings_accept_direct_overrides() -> None:
    settings = create_settings(
        openai_model="test-model",
        max_tool_rounds=2,
        max_memory_turns=5,
        system_prompt="Test system prompt.",
    )

    assert settings.openai_model == "test-model"
    assert settings.max_tool_rounds == 2
    assert settings.max_memory_turns == 5
    assert settings.system_prompt == "Test system prompt."


def test_settings_load_from_env_file(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"

    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-test-key",
                "OPENAI_MODEL=file-model",
                "MAX_TOOL_ROUNDS=3",
                "MAX_MEMORY_TURNS=6",
                "SYSTEM_PROMPT=Loaded from env file.",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=env_path,
    )

    assert (
        settings.openai_api_key.get_secret_value()
        == "file-test-key"
    )
    assert settings.openai_model == "file-model"
    assert settings.max_tool_rounds == 3
    assert settings.max_memory_turns == 6
    assert settings.system_prompt == "Loaded from env file."


def test_api_key_is_masked_when_converted_to_string() -> None:
    settings = create_settings()

    assert str(settings.openai_api_key) == "**********"
    assert TEST_KEY not in repr(settings)


def test_blank_api_key_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="OPENAI_API_KEY cannot be blank",
    ):
        create_settings(
            openai_api_key="   ",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        11,
    ],
)
def test_invalid_tool_round_limits_are_rejected(
    invalid_value: int,
) -> None:
    with pytest.raises(ValidationError):
        create_settings(
            max_tool_rounds=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        51,
    ],
)
def test_invalid_memory_limits_are_rejected(
    invalid_value: int,
) -> None:
    with pytest.raises(ValidationError):
        create_settings(
            max_memory_turns=invalid_value,
        )


def test_text_values_are_trimmed() -> None:
    settings = create_settings(
        openai_model="  test-model  ",
        system_prompt="  Test prompt.  ",
    )

    assert settings.openai_model == "test-model"
    assert settings.system_prompt == "Test prompt."