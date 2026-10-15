from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel


SafetyCategory = Literal[
    "allowed",
    "secret_extraction",
    "system_prompt_extraction",
    "prompt_injection",
]


class SafetyDecision(BaseModel):
    """
    Result of evaluating one user message.
    """

    allowed: bool
    category: SafetyCategory
    refusal_message: str | None = None


class SafetyService:
    """
    Applies basic server-side safety checks and secret redaction.

    The service blocks attempts to retrieve application secrets,
    hidden instructions, or override the assistant's safety rules.
    """

    _SECRET_REQUEST_PATTERN = re.compile(
        r"\b("
        r"show|reveal|display|print|give|send|return|expose|tell"
        r")\b"
        r".{0,100}"
        r"\b("
        r"api[\s_-]*keys?|"
        r"openai_api_key|"
        r"secrets?|"
        r"tokens?|"
        r"\.env|"
        r"environment variables?"
        r")\b",
        re.IGNORECASE | re.DOTALL,
    )

    _SYSTEM_PROMPT_PATTERN = re.compile(
        r"\b("
        r"show|reveal|display|print|give|repeat|return|expose"
        r")\b"
        r".{0,100}"
        r"\b("
        r"system prompts?|"
        r"hidden prompts?|"
        r"hidden instructions?|"
        r"developer messages?|"
        r"developer instructions?"
        r")\b",
        re.IGNORECASE | re.DOTALL,
    )

    _PROMPT_INJECTION_PATTERN = re.compile(
        r"\b("
        r"ignore (all |any )?(previous|prior|system|developer) "
        r"instructions|"
        r"override (the )?(system|developer|safety) "
        r"(prompt|instructions|rules)|"
        r"bypass (the )?(safety|security|content) "
        r"(rules|policy|filter)|"
        r"jailbreak (the )?(assistant|model|system)"
        r")\b",
        re.IGNORECASE,
    )

    _OPENAI_KEY_PATTERN = re.compile(
        r"\bsk-[A-Za-z0-9_-]{12,}\b"
    )

    _ENV_SECRET_PATTERN = re.compile(
        r"(?i)\b("
        r"OPENAI_API_KEY|"
        r"API_KEY|"
        r"SECRET_KEY|"
        r"ACCESS_TOKEN"
        r")\s*=\s*[^\s\"']+"
    )

    _BEARER_TOKEN_PATTERN = re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"
    )

    def evaluate(
        self,
        user_message: str,
    ) -> SafetyDecision:
        """
        Evaluates a user message before it reaches the model.
        """

        cleaned_message = user_message.strip()

        if self._SECRET_REQUEST_PATTERN.search(
            cleaned_message
        ):
            return SafetyDecision(
                allowed=False,
                category="secret_extraction",
                refusal_message=(
                    "I cannot reveal API keys, environment "
                    "variables, access tokens, or other "
                    "application secrets."
                ),
            )

        if self._SYSTEM_PROMPT_PATTERN.search(
            cleaned_message
        ):
            return SafetyDecision(
                allowed=False,
                category="system_prompt_extraction",
                refusal_message=(
                    "I cannot provide hidden system or developer "
                    "instructions. I can explain the assistant's "
                    "public capabilities and safety behavior."
                ),
            )

        if self._PROMPT_INJECTION_PATTERN.search(
            cleaned_message
        ):
            return SafetyDecision(
                allowed=False,
                category="prompt_injection",
                refusal_message=(
                    "I cannot override or bypass the assistant's "
                    "safety and application instructions. Please "
                    "ask a project-related question instead."
                ),
            )

        return SafetyDecision(
            allowed=True,
            category="allowed",
        )

    def redact_text(
        self,
        text: str,
    ) -> str:
        """
        Redacts common API-key and token formats before logging.
        """

        redacted = self._OPENAI_KEY_PATTERN.sub(
            "[REDACTED_API_KEY]",
            text,
        )

        redacted = self._ENV_SECRET_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}=[REDACTED_SECRET]"
            ),
            redacted,
        )

        redacted = self._BEARER_TOKEN_PATTERN.sub(
            "Bearer [REDACTED_TOKEN]",
            redacted,
        )

        return redacted