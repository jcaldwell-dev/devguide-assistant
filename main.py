from app.application import create_application_services


def main() -> None:
    services = create_application_services()

    print("APPLICATION CONFIGURATION")
    print(
        f"Model: {services.settings.openai_model}"
    )
    print(
        "Memory turns: "
        f"{services.settings.max_memory_turns}"
    )
    print(
        "Tool rounds: "
        f"{services.settings.max_tool_rounds}"
    )
    print(
        "API key loaded: "
        f"{bool(services.settings.openai_api_key)}"
    )

    print("\nREGISTERED TOOLS")

    for tool_name in services.registry.list_tool_names():
        print(f"- {tool_name}")


if __name__ == "__main__":
    main()