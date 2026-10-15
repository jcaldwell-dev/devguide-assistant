from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request body accepted by POST /chat.
    """

    conversation_id: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Client-provided identifier used to retain "
            "conversation memory."
        ),
        examples=["demo-conversation-001"],
    )

    user_message: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's message to the assistant.",
        examples=[
            "What prefix should an Actor Component Blueprint use?"
        ],
    )

    @field_validator(
        "conversation_id",
        "user_message",
    )
    @classmethod
    def strip_and_reject_blank_text(
        cls,
        value: str,
    ) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Value cannot be blank."
            )

        return cleaned_value


class HealthResponse(BaseModel):
    """
    Response returned by GET /health.
    """

    status: str
    application: str
    model: str


class ClearConversationResponse(BaseModel):
    """
    Response returned when a conversation is cleared.
    """

    conversation_id: str
    removed: bool