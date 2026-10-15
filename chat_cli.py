from __future__ import annotations

import json

from uuid import uuid4

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from app.application import (
    ApplicationServices,
    create_application_services,
)
from app.services.openai_service import ChatResult


def print_result(result: ChatResult) -> None:
    """
    Displays the completed assistant response and basic API metadata.
    """

    print("\nAssistant:")
    print(result.text)

    print("\n--- Turn Details ---")
    print(f"Model: {result.model}")
    print(f"Model rounds: {result.model_rounds}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Total tokens: {result.total_tokens}")

    if result.tool_calls:
        print("Tool calls:")

        for tool_call in result.tool_calls:
            status = (
                "success"
                if tool_call.success
                else f"failed: {tool_call.error_type}"
            )

            print(
                f"  - {tool_call.tool_name} "
                f"({status})"
            )

            print(
                "    Arguments: "
                f"{json.dumps(tool_call.arguments, indent=2)}"
            )
    else:
        print("Tool calls: none")


def print_help() -> None:
    """
    Displays the commands supported by the local CLI.
    """

    print(
        "\nCommands:\n"
        "  /help   Show available commands\n"
        "  /new    Start a new conversation\n"
        "  /clear  Clear the current conversation\n"
        "  /quit   Exit the chatbot\n"
    )


def run_chat(
    services: ApplicationServices,
) -> None:
    """
    Runs an interactive command-line chatbot session.
    """

    conversation_id = str(uuid4())

    print("\nDevGuide Assistant")
    print("==================")
    print(f"Model: {services.settings.openai_model}")
    print(
        "Type /help for commands or /quit to exit."
    )
    print(
        "API requests may consume available API credits.\n"
    )

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nClosing DevGuide Assistant.")
            break

        if not user_message:
            continue

        command = user_message.lower()

        if command in {"/quit", "/exit"}:
            print("Closing DevGuide Assistant.")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/new":
            conversation_id = str(uuid4())

            print(
                "\nStarted a new conversation.\n"
            )
            continue

        if command == "/clear":
            removed = (
                services.memory.clear_conversation(
                    conversation_id
                )
            )

            if removed:
                print(
                    "\nCurrent conversation cleared.\n"
                )
            else:
                print(
                    "\nThe current conversation was "
                    "already empty.\n"
                )

            continue

        try:
            result = services.openai_service.chat(
                conversation_id=conversation_id,
                user_message=user_message,
            )

            print_result(result)

        except AuthenticationError:
            print(
                "\nAuthentication failed. Check that "
                "OPENAI_API_KEY contains a valid project "
                "API key.\n"
            )

        except RateLimitError as error:
            print(
                "\nThe API rejected the request because of "
                "a rate limit or unavailable quota."
            )
            print(
                "Check the API project's billing, credits, "
                "and model access."
            )
            print(f"Details: {error}\n")

        except APIConnectionError as error:
            print(
                "\nThe application could not connect to "
                "the OpenAI API."
            )
            print(
                "Check the internet connection and try again."
            )
            print(f"Details: {error}\n")

        except APIStatusError as error:
            print(
                "\nThe OpenAI API returned an error."
            )
            print(f"HTTP status: {error.status_code}")
            print(f"Details: {error}\n")

        except Exception as error:
            print(
                "\nThe chatbot encountered an unexpected "
                "error."
            )
            print(
                f"{type(error).__name__}: {error}\n"
            )


def main() -> None:
    services = create_application_services()
    run_chat(services)


if __name__ == "__main__":
    main()