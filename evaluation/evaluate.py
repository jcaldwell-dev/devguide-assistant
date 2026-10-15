from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient

from app.api.server import create_api
from app.application import create_application_services
from app.config import PROJECT_ROOT, get_settings


PROMPTS_PATH = PROJECT_ROOT / "evaluation" / "prompts.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

EVALUATION_RESULTS_PATH = (
    RESULTS_DIRECTORY / "evaluation_results.json"
)

ABLATION_RESULTS_PATH = (
    RESULTS_DIRECTORY / "ablation_results.json"
)

METRICS_PATH = RESULTS_DIRECTORY / "metrics.json"
MANUAL_REVIEW_PATH = RESULTS_DIRECTORY / "manual_review.csv"
TRANSCRIPTS_DIRECTORY = RESULTS_DIRECTORY / "transcripts"


@dataclass(frozen=True)
class SuiteConfiguration:
    """
    Configuration for one evaluation run.
    """

    name: str
    memory_turns: int
    output_directory: Path


def load_prompts() -> list[dict[str, Any]]:
    """
    Loads and validates the evaluation prompt definitions.
    """

    data = json.loads(
        PROMPTS_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "evaluation/prompts.json must contain a list."
        )

    if len(data) != 20:
        raise ValueError(
            "The required evaluation suite must contain "
            "exactly 20 prompts."
        )

    return data


def contains_all_terms(
    text: str,
    terms: list[str],
) -> bool:
    normalized_text = text.lower()

    return all(
        term.lower() in normalized_text
        for term in terms
    )


def contains_any_term(
    text: str,
    terms: list[str],
) -> bool:
    if not terms:
        return True

    normalized_text = text.lower()

    return any(
        term.lower() in normalized_text
        for term in terms
    )


def automatically_score(
    prompt_definition: dict[str, Any],
    status_code: int,
    response_body: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Applies deterministic checks to one evaluation response.
    """

    failures: list[str] = []

    expected_status = int(
        prompt_definition.get("expected_status", 200)
    )

    if status_code != expected_status:
        failures.append(
            f"Expected HTTP {expected_status}, received "
            f"{status_code}."
        )

        return False, failures

    if status_code != 200:
        return True, failures

    assistant_text = str(
        response_body.get("text", "")
    )

    tool_calls = response_body.get("tool_calls", [])

    actual_tool_names = {
        str(tool_call.get("tool_name", ""))
        for tool_call in tool_calls
    }

    expected_tools = set(
        prompt_definition.get("expected_tools", [])
    )

    missing_tools = (
        expected_tools - actual_tool_names
    )

    if missing_tools:
        failures.append(
            "Missing expected tools: "
            + ", ".join(sorted(missing_tools))
        )

    required_terms = prompt_definition.get(
        "required_terms",
        [],
    )

    if not contains_all_terms(
        assistant_text,
        required_terms,
    ):
        failures.append(
            "One or more required response terms were missing."
        )

    required_any_terms = prompt_definition.get(
        "required_any_terms",
        [],
    )

    if not contains_any_term(
        assistant_text,
        required_any_terms,
    ):
        failures.append(
            "None of the acceptable response terms were found."
        )

    forbidden_terms = prompt_definition.get(
        "forbidden_terms",
        [],
    )

    if forbidden_terms and contains_any_term(
        assistant_text,
        forbidden_terms,
    ):
        failures.append(
            "A forbidden response term was present."
        )

    if prompt_definition.get(
        "expected_refusal",
        False,
    ):
        model_rounds = int(
            response_body.get("model_rounds", -1)
        )

        total_tokens = int(
            response_body.get("total_tokens", -1)
        )

        if model_rounds != 0:
            failures.append(
                "Safety refusal reached the model."
            )

        if total_tokens != 0:
            failures.append(
                "Safety refusal consumed model tokens."
            )

    return len(failures) == 0, failures


def parse_response_json(
    response,
) -> dict[str, Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return {
            "raw_text": response.text,
        }

    if isinstance(parsed, dict):
        return parsed

    return {
        "response": parsed,
    }


def write_prompt_transcript(
    result: dict[str, Any],
    directory: Path,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    transcript_path = (
        directory / f"{result['prompt_id']}.json"
    )

    transcript_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_suite(
    prompts: list[dict[str, Any]],
    configuration: SuiteConfiguration,
) -> list[dict[str, Any]]:
    """
    Runs prompts through the real FastAPI application and OpenAI API.
    """

    output_directory = (
        configuration.output_directory.resolve()
    )

    runtime_results_directory = (
        output_directory / "runtime_metrics"
    )

    base_settings = get_settings()

    settings = base_settings.model_copy(
        update={
            "max_memory_turns": (
                configuration.memory_turns
            ),
            "results_directory": (
                runtime_results_directory
            ),
        }
    )

    services = create_application_services(settings)
    api = create_api(services)

    results: list[dict[str, Any]] = []

    with TestClient(api) as client:
        for index, prompt_definition in enumerate(
            prompts,
            start=1,
        ):
            prompt_id = str(prompt_definition["id"])
            prompt_text = str(
                prompt_definition["prompt"]
            )

            print(
                f"[{index:02d}/{len(prompts):02d}] "
                f"{configuration.name} {prompt_id}"
            )

            started_at = perf_counter()

            response = client.post(
                "/chat/json",
                json={
                    "conversation_id": (
                        prompt_definition[
                            "conversation_id"
                        ]
                    ),
                    "user_message": prompt_text,
                },
            )

            latency_ms = (
                perf_counter() - started_at
            ) * 1000

            response_body = parse_response_json(
                response
            )

            automatic_success, failures = (
                automatically_score(
                    prompt_definition,
                    response.status_code,
                    response_body,
                )
            )

            input_tokens = int(
                response_body.get("input_tokens", 0)
            )

            output_tokens = int(
                response_body.get("output_tokens", 0)
            )

            estimated_cost = (
                services.metrics.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

            result = {
                "suite": configuration.name,
                "prompt_id": prompt_id,
                "category": prompt_definition[
                    "category"
                ],
                "description": prompt_definition[
                    "description"
                ],
                "conversation_id": prompt_definition[
                    "conversation_id"
                ],
                "prompt": prompt_text,
                "status_code": response.status_code,
                "response": response_body,
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(
                    response_body.get(
                        "total_tokens",
                        0,
                    )
                ),
                "estimated_cost_usd": (
                    estimated_cost
                ),
                "automatic_success": (
                    automatic_success
                ),
                "automatic_failures": failures,
                "manual_task_success": None,
                "manual_hallucination": None,
                "manual_notes": "",
            }

            results.append(result)

            write_prompt_transcript(
                result,
                output_directory / "transcripts",
            )

    return results


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Produces aggregate evaluation metrics.
    """

    if not results:
        raise ValueError(
            "Cannot summarize an empty result set."
        )

    successful_results = sum(
        1
        for result in results
        if result["automatic_success"]
    )

    total_latency = sum(
        float(result["latency_ms"])
        for result in results
    )

    total_input_tokens = sum(
        int(result["input_tokens"])
        for result in results
    )

    total_output_tokens = sum(
        int(result["output_tokens"])
        for result in results
    )

    total_tokens = sum(
        int(result["total_tokens"])
        for result in results
    )

    total_cost = sum(
        float(result["estimated_cost_usd"])
        for result in results
    )

    category_summaries: dict[str, Any] = {}

    categories = sorted(
        {
            str(result["category"])
            for result in results
        }
    )

    for category in categories:
        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        category_successes = sum(
            1
            for result in category_results
            if result["automatic_success"]
        )

        category_summaries[category] = {
            "prompt_count": len(
                category_results
            ),
            "automatic_success_count": (
                category_successes
            ),
            "automatic_task_success_percent": round(
                category_successes
                / len(category_results)
                * 100,
                2,
            ),
            "average_latency_ms": round(
                sum(
                    float(result["latency_ms"])
                    for result in category_results
                )
                / len(category_results),
                3,
            ),
            "average_total_tokens": round(
                sum(
                    int(result["total_tokens"])
                    for result in category_results
                )
                / len(category_results),
                3,
            ),
        }

    count = len(results)

    return {
        "prompt_count": count,
        "automatic_success_count": successful_results,
        "automatic_failure_count": (
            count - successful_results
        ),
        "task_success_percent": round(
            successful_results / count * 100,
            2,
        ),
        "average_latency_ms": round(
            total_latency / count,
            3,
        ),
        "average_input_tokens": round(
            total_input_tokens / count,
            3,
        ),
        "average_output_tokens": round(
            total_output_tokens / count,
            3,
        ),
        "average_total_tokens": round(
            total_tokens / count,
            3,
        ),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_estimated_cost_usd": round(
            total_cost,
            8,
        ),
        "hallucination_rate_percent": None,
        "categories": category_summaries,
    }


def write_manual_review_csv(
    results: list[dict[str, Any]],
) -> None:
    """
    Creates a worksheet for manual success and hallucination review.
    """

    with MANUAL_REVIEW_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "prompt_id",
                "category",
                "automatic_success",
                "manual_task_success",
                "manual_hallucination",
                "manual_notes",
                "prompt",
                "assistant_response",
            ],
        )

        writer.writeheader()

        for result in results:
            response = result.get(
                "response",
                {},
            )

            writer.writerow(
                {
                    "prompt_id": result["prompt_id"],
                    "category": result["category"],
                    "automatic_success": (
                        result["automatic_success"]
                    ),
                    "manual_task_success": "",
                    "manual_hallucination": "",
                    "manual_notes": "",
                    "prompt": result["prompt"],
                    "assistant_response": (
                        response.get("text", "")
                        if isinstance(response, dict)
                        else ""
                    ),
                }
            )


def run_ablation(
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compares 2-turn and 8-turn memory using the five memory prompts.
    """

    memory_prompts = [
        prompt
        for prompt in prompts
        if prompt["category"] == "memory"
    ]

    short_results = run_suite(
        memory_prompts,
        SuiteConfiguration(
            name="memory_2_turns",
            memory_turns=2,
            output_directory=(
                RESULTS_DIRECTORY
                / "ablation"
                / "memory_2_turns"
            ),
        ),
    )

    long_results = run_suite(
        memory_prompts,
        SuiteConfiguration(
            name="memory_8_turns",
            memory_turns=8,
            output_directory=(
                RESULTS_DIRECTORY
                / "ablation"
                / "memory_8_turns"
            ),
        ),
    )

    return {
        "experiment": "short_vs_long_memory",
        "short_memory_turns": 2,
        "long_memory_turns": 8,
        "short_memory": {
            "summary": summarize_results(
                short_results
            ),
            "results": short_results,
        },
        "long_memory": {
            "summary": summarize_results(
                long_results
            ),
            "results": long_results,
        },
    }


def main() -> None:
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSCRIPTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompts = load_prompts()

    baseline_results = run_suite(
        prompts,
        SuiteConfiguration(
            name="baseline_8_turns",
            memory_turns=8,
            output_directory=RESULTS_DIRECTORY,
        ),
    )

    baseline_summary = summarize_results(
        baseline_results
    )

    evaluation_payload = {
        "configuration": {
            "model": get_settings().openai_model,
            "memory_turns": 8,
            "prompt_count": len(prompts),
        },
        "summary": baseline_summary,
        "results": baseline_results,
    }

    EVALUATION_RESULTS_PATH.write_text(
        json.dumps(
            evaluation_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    METRICS_PATH.write_text(
        json.dumps(
            baseline_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_manual_review_csv(
        baseline_results
    )

    ablation_payload = run_ablation(
        prompts
    )

    ABLATION_RESULTS_PATH.write_text(
        json.dumps(
            ablation_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEvaluation complete.")
    print(
        "Automatic Task Success: "
        f"{baseline_summary['task_success_percent']}%"
    )
    print(
        "Average Latency: "
        f"{baseline_summary['average_latency_ms']} ms"
    )
    print(
        "Total Tokens: "
        f"{baseline_summary['total_tokens']}"
    )
    print(
        "Estimated Cost: $"
        f"{baseline_summary['total_estimated_cost_usd']:.8f}"
    )
    print(
        f"Results: {EVALUATION_RESULTS_PATH}"
    )
    print(
        f"Manual Review: {MANUAL_REVIEW_PATH}"
    )
    print(
        f"Ablation: {ABLATION_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()