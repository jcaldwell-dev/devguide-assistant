from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT


RESULTS_DIRECTORY = PROJECT_ROOT / "results"

EVALUATION_RESULTS_PATH = (
    RESULTS_DIRECTORY / "evaluation_results.json"
)

MANUAL_REVIEW_PATH = (
    RESULTS_DIRECTORY / "manual_review.csv"
)

METRICS_PATH = RESULTS_DIRECTORY / "metrics.json"

MANUAL_SUMMARY_PATH = (
    RESULTS_DIRECTORY / "manual_scoring_summary.json"
)


def parse_binary_score(
    value: str,
    field_name: str,
    prompt_id: str,
) -> int:
    cleaned_value = value.strip()

    if cleaned_value not in {"0", "1"}:
        raise ValueError(
            f"{field_name} for {prompt_id} must be 0 or 1."
        )

    return int(cleaned_value)


def load_manual_reviews() -> list[dict[str, Any]]:
    with MANUAL_REVIEW_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != 20:
        raise ValueError(
            "manual_review.csv must contain exactly 20 rows."
        )

    return rows


def main() -> None:
    evaluation_payload = json.loads(
        EVALUATION_RESULTS_PATH.read_text(
            encoding="utf-8"
        )
    )

    reviews = load_manual_reviews()

    review_by_id = {
        row["prompt_id"]: row
        for row in reviews
    }

    manual_success_count = 0
    hallucination_count = 0

    for result in evaluation_payload["results"]:
        prompt_id = result["prompt_id"]
        review = review_by_id[prompt_id]

        manual_success = parse_binary_score(
            review["manual_task_success"],
            "manual_task_success",
            prompt_id,
        )

        manual_hallucination = parse_binary_score(
            review["manual_hallucination"],
            "manual_hallucination",
            prompt_id,
        )

        result["manual_task_success"] = bool(
            manual_success
        )

        result["manual_hallucination"] = bool(
            manual_hallucination
        )

        result["manual_notes"] = review.get(
            "manual_notes",
            "",
        )

        manual_success_count += manual_success
        hallucination_count += manual_hallucination

    prompt_count = len(
        evaluation_payload["results"]
    )

    manual_task_success_percent = round(
        manual_success_count
        / prompt_count
        * 100,
        2,
    )

    hallucination_rate_percent = round(
        hallucination_count
        / prompt_count
        * 100,
        2,
    )

    automatic_success_count = sum(
        1
        for result in evaluation_payload["results"]
        if result["automatic_success"]
    )

    automatic_task_success_percent = round(
        automatic_success_count / prompt_count * 100,
        2,
    )

    evaluation_payload["summary"][
        "task_success_percent"
    ] = automatic_task_success_percent

    evaluation_payload["summary"][
        "automatic_task_success_percent"
    ] = automatic_task_success_percent

    evaluation_payload["summary"][
        "manual_task_success_percent"
    ] = manual_task_success_percent

    evaluation_payload["summary"][
        "hallucination_rate_percent"
    ] = hallucination_rate_percent

    EVALUATION_RESULTS_PATH.write_text(
        json.dumps(
            evaluation_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics = dict(evaluation_payload["summary"])

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manual_summary = {
        "prompt_count": prompt_count,
        "manual_success_count": manual_success_count,
        "manual_failure_count": (
            prompt_count - manual_success_count
        ),
        "manual_task_success_percent": (
            manual_task_success_percent
        ),
        "hallucination_count": hallucination_count,
        "hallucination_rate_percent": (
            hallucination_rate_percent
        ),
    }

    MANUAL_SUMMARY_PATH.write_text(
        json.dumps(
            manual_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Manual scoring complete.")
    print(
        "Task Success: "
        f"{manual_task_success_percent}%"
    )
    print(
        "Hallucination Rate: "
        f"{hallucination_rate_percent}%"
    )


if __name__ == "__main__":
    main()