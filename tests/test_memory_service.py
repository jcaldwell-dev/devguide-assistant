import pytest
from pydantic import ValidationError

from app.services.memory_service import (
    ChatMessage,
    MemoryService,
)


SYSTEM_PROMPT = (
    "You are a safe game-development production assistant."
)


def create_memory(max_turns: int = 8) -> MemoryService:
    return MemoryService(
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )


def test_unknown_conversation_returns_system_prompt_only() -> None:
    memory = create_memory()

    messages = memory.get_messages("conversation-1")

    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert memory.has_conversation("conversation-1") is False


def test_simple_turn_is_stored() -> None:
    memory = create_memory()

    snapshot = memory.add_turn(
        conversation_id="conversation-1",
        messages=[
            {
                "role": "user",
                "content": "Remember Sedan, Hybrid, and Truck.",
            },
            {
                "role": "assistant",
                "content": "The approved vehicles are recorded.",
            },
        ],
    )

    assert snapshot.turn_count == 1
    assert len(snapshot.messages) == 3
    assert snapshot.messages[1].role == "user"
    assert snapshot.messages[2].role == "assistant"


def test_conversation_id_whitespace_is_normalized() -> None:
    memory = create_memory()

    memory.add_turn(
        conversation_id="  project-chat  ",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            },
            {
                "role": "assistant",
                "content": "Hello.",
            },
        ],
    )

    assert memory.has_conversation("project-chat") is True
    assert memory.get_turn_count(" project-chat ") == 1


def test_separate_conversations_remain_isolated() -> None:
    memory = create_memory()

    memory.add_turn(
        "conversation-a",
        [
            {
                "role": "user",
                "content": "Milestone Alpha",
            },
            {
                "role": "assistant",
                "content": "Recorded Alpha.",
            },
        ],
    )

    memory.add_turn(
        "conversation-b",
        [
            {
                "role": "user",
                "content": "Milestone Beta",
            },
            {
                "role": "assistant",
                "content": "Recorded Beta.",
            },
        ],
    )

    messages_a = memory.get_messages("conversation-a")
    messages_b = memory.get_messages("conversation-b")

    contents_a = [message.content for message in messages_a]
    contents_b = [message.content for message in messages_b]

    assert "Milestone Alpha" in contents_a
    assert "Milestone Beta" not in contents_a

    assert "Milestone Beta" in contents_b
    assert "Milestone Alpha" not in contents_b


def test_oldest_complete_turn_is_removed() -> None:
    memory = create_memory(max_turns=2)

    for number in range(1, 4):
        memory.add_turn(
            "rolling-chat",
            [
                {
                    "role": "user",
                    "content": f"User turn {number}",
                },
                {
                    "role": "assistant",
                    "content": f"Assistant turn {number}",
                },
            ],
        )

    messages = memory.get_messages("rolling-chat")
    contents = [message.content for message in messages]

    assert memory.get_turn_count("rolling-chat") == 2
    assert "User turn 1" not in contents
    assert "Assistant turn 1" not in contents
    assert "User turn 2" in contents
    assert "Assistant turn 2" in contents
    assert "User turn 3" in contents
    assert "Assistant turn 3" in contents


def test_tool_call_sequence_is_preserved_as_one_turn() -> None:
    memory = create_memory(max_turns=1)

    memory.add_turn(
        "tool-chat",
        [
            {
                "role": "user",
                "content": "Estimate the inventory feature.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "estimate_task",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"risk_level": "Moderate"}',
                "tool_call_id": "call-1",
                "name": "estimate_task",
            },
            {
                "role": "assistant",
                "content": "The task has moderate planning risk.",
            },
        ],
    )

    messages = memory.get_messages("tool-chat")

    assert len(messages) == 5
    assert messages[1].role == "user"
    assert messages[2].tool_calls is not None
    assert messages[3].role == "tool"
    assert messages[3].tool_call_id == "call-1"
    assert messages[4].role == "assistant"


def test_tool_turn_is_removed_as_a_complete_unit() -> None:
    memory = create_memory(max_turns=1)

    memory.add_turn(
        "tool-chat",
        [
            {
                "role": "user",
                "content": "Use a tool.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "old-call",
                        "type": "function",
                        "function": {
                            "name": "search_project_kb",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"matches": []}',
                "tool_call_id": "old-call",
            },
            {
                "role": "assistant",
                "content": "No matches were found.",
            },
        ],
    )

    memory.add_turn(
        "tool-chat",
        [
            {
                "role": "user",
                "content": "New question",
            },
            {
                "role": "assistant",
                "content": "New answer",
            },
        ],
    )

    messages = memory.get_messages("tool-chat")
    serialized = [
        message.model_dump(exclude_none=True)
        for message in messages
    ]

    assert len(messages) == 3
    assert all(
        item.get("tool_call_id") != "old-call"
        for item in serialized
    )
    assert all(
        item.get("tool_calls") is None
        for item in serialized
    )


def test_message_dicts_exclude_none_fields() -> None:
    memory = create_memory()

    memory.add_turn(
        "serialized-chat",
        [
            {
                "role": "user",
                "content": "Hello",
            },
            {
                "role": "assistant",
                "content": "Hello.",
            },
        ],
    )

    messages = memory.get_message_dicts("serialized-chat")

    assert messages[1] == {
        "role": "user",
        "content": "Hello",
    }

    assert "tool_call_id" not in messages[1]
    assert "tool_calls" not in messages[1]


def test_clear_conversation_removes_only_requested_chat() -> None:
    memory = create_memory()

    for conversation_id in ["chat-a", "chat-b"]:
        memory.add_turn(
            conversation_id,
            [
                {
                    "role": "user",
                    "content": "Hello",
                },
                {
                    "role": "assistant",
                    "content": "Hello.",
                },
            ],
        )

    removed = memory.clear_conversation("chat-a")

    assert removed is True
    assert memory.has_conversation("chat-a") is False
    assert memory.has_conversation("chat-b") is True
    assert memory.conversation_count() == 1


def test_clearing_unknown_conversation_returns_false() -> None:
    memory = create_memory()

    assert memory.clear_conversation("missing") is False


def test_clear_all_returns_removed_count() -> None:
    memory = create_memory()

    for conversation_id in ["chat-a", "chat-b", "chat-c"]:
        memory.add_turn(
            conversation_id,
            [
                {
                    "role": "user",
                    "content": "Hello",
                },
                {
                    "role": "assistant",
                    "content": "Hello.",
                },
            ],
        )

    removed_count = memory.clear_all()

    assert removed_count == 3
    assert memory.conversation_count() == 0


def test_blank_conversation_id_is_rejected() -> None:
    memory = create_memory()

    with pytest.raises(
        ValueError,
        match="conversation_id cannot be blank",
    ):
        memory.get_messages("   ")


def test_blank_system_prompt_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt cannot be blank",
    ):
        MemoryService(
            system_prompt=" ",
            max_turns=8,
        )


def test_invalid_max_turns_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_turns must be at least 1",
    ):
        MemoryService(
            system_prompt=SYSTEM_PROMPT,
            max_turns=0,
        )


def test_blank_user_message_is_rejected() -> None:
    memory = create_memory()

    with pytest.raises(ValidationError):
        memory.add_turn(
            "invalid-chat",
            [
                {
                    "role": "user",
                    "content": " ",
                },
                {
                    "role": "assistant",
                    "content": "Response",
                },
            ],
        )


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(
            role="tool",
            content='{"result": true}',
        )


def test_assistant_may_contain_tool_calls_without_text() -> None:
    message = ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "estimate_task",
                    "arguments": "{}",
                },
            }
        ],
    )

    assert message.content == ""
    assert message.tool_calls is not None