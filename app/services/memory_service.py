from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]

class ChatMessage(BaseModel):
    """
    Represents one message stored in conversation memory.

    Assistant tool-request messages may contain tool_calls instead
    of normal text. Tool-result messages must include a tool_call_id.
    """

    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_message(self) -> "ChatMessage":
        cleaned_content = self.content.strip()

        if self.role in {"system", "user"}:
            if not cleaned_content:
                raise ValueError(
                    f"{self.role} messages must contain text."
                )

        elif self.role == "assistant":
            has_text = bool(cleaned_content)
            has_tool_calls = bool(self.tool_calls)

            if not has_text and not has_tool_calls:
                raise ValueError(
                    "Assistant messages must contain text or tool calls."
                )

        elif self.role == "tool":
            if not cleaned_content:
                raise ValueError(
                    "Tool messages must contain a result."
                )

            if not self.tool_call_id:
                raise ValueError(
                    "Tool messages must include tool_call_id."
                )

        self.content = cleaned_content
        return self

class ConversationTurn(BaseModel):
    """
    Represents one complete conversation turn.

    A turn may contain a simple user/assistant exchange or a full
    user/tool-call/tool-result/assistant sequence.
    """

    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_system_messages(self) -> "ConversationTurn":
        if any(
            message.role == "system"
            for message in self.messages
        ):
            raise ValueError(
                "System messages are managed by MemoryService"
                "and cannot be added inside a conversation turn."
            )

        return self

class ConversationSnapshot(BaseModel):
    """
    Read-only representation of one conversation's current state.
    """

    conversation_id: str
    max_turns: int
    turn_count: int
    messages: list[ChatMessage]

@dataclass
class _ConversationState:
    """
    Private mutable storage for one conversation.
    """

    turns: list[ConversationTurn] = field(default_factory=list)

class MemoryService:
    """
    Thread-safe in-memory conversation storage keyed by conversation_id.

    The store preserves complete turns and removes the oldest whole
    turn when the configured rolling-window limit is exceeded.
    """

    def __init__(
            self,
            system_prompt: str,
            max_turns: int = 8,
    ) -> None:
        cleaned_prompt = system_prompt.strip()

        if not cleaned_prompt:
            raise ValueError("system_prompt cannot be blank.")

        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self._system_prompt = cleaned_prompt
        self._max_turns = max_turns
        self._conversations: dict[str, _ConversationState] = {}
        self._lock = RLock()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @staticmethod
    def _normalize_conversation_id(
            conversation_id: str,

    ) -> str:
        cleaned_id = conversation_id.strip()

        if not cleaned_id:
            raise ValueError(
                "conversation_id cannot be blank."
            )

        if len(cleaned_id) > 100:
            raise ValueError(
                "conversation_id cannot exceed 100 characters."
            )

        return cleaned_id

    def has_conversation(
            self,
            conversation_id: str,
    ) -> bool:
        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        with self._lock:
            return cleaned_id in self._conversations


    def add_turn(
            self,
            conversation_id: str,
            messages: Sequence[
                ChatMessage | dict[str, Any]
            ],
    ) -> ConversationSnapshot:
        """
        Validates and appends one complete turn.

        If the conversation exceeds max_turns, the oldest complete
        turns are discarded.
        """

        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        validated_turn = ConversationTurn.model_validate(
            {
                "messages": list(messages),
            }
        )

        with self._lock:
            state = self._conversations.setdefault(
                cleaned_id,
                _ConversationState(),
            )

            state.turns.append(
                validated_turn.model_copy(deep=True)
            )

            if len(state.turns) > self._max_turns:
                state.turns = state.turns[-self._max_turns:]

            return self._create_snapshot(
                cleaned_id,
                state,
            )

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[ChatMessage]:
        """
        Returns the system prompt followed by all retained messages.

        Reading an unknown conversation does not create it.
        """

        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        system_message = ChatMessage(
            role="system",
            content=self._system_prompt,
        )

        with self._lock:
            state = self._conversations.get(cleaned_id)

            if state is None:
                return [system_message]

            messages = [system_message]

            for turn in state.turns:
                messages.extend(
                    message.model_copy(deep=True)
                    for message in turn.messages
                )

            return messages

    def get_message_dicts(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        Returns API-ready message dictionaries.
        """

        return [
            message.model_dump(exclude_none=True)
            for message in self.get_messages(conversation_id)
        ]

    def get_turn_count(
        self,
        conversation_id: str,
    ) -> int:
        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        with self._lock:
            state = self._conversations.get(cleaned_id)

            if state is None:
                return 0

            return len(state.turns)

    def get_snapshot(
        self,
        conversation_id: str,
    ) -> ConversationSnapshot:
        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        with self._lock:
            state = self._conversations.get(
                cleaned_id,
                _ConversationState(),
            )

            return self._create_snapshot(
                cleaned_id,
                state,
            )

    def clear_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """
        Removes one conversation.

        Returns True when a stored conversation was removed.
        """

        cleaned_id = self._normalize_conversation_id(
            conversation_id
        )

        with self._lock:
            return (
                self._conversations.pop(cleaned_id, None)
                is not None
            )

    def clear_all(self) -> int:
        """
        Removes every stored conversation.

        Returns the number of conversations removed.
        """

        with self._lock:
            removed_count = len(self._conversations)
            self._conversations.clear()
            return removed_count

    def conversation_count(self) -> int:
        with self._lock:
            return len(self._conversations)

    def _create_snapshot(
        self,
        conversation_id: str,
        state: _ConversationState,
    ) -> ConversationSnapshot:
        messages = [
            ChatMessage(
                role="system",
                content=self._system_prompt,
            )
        ]

        for turn in state.turns:
            messages.extend(
                message.model_copy(deep=True)
                for message in turn.messages
            )

        return ConversationSnapshot(
            conversation_id=conversation_id,
            max_turns=self._max_turns,
            turn_count=len(state.turns),
            messages=messages,
        )