from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from app.config import PROJECT_ROOT


RESULTS_DIRECTORY = PROJECT_ROOT / "results"

EVALUATION_RESULTS_PATH = (
    RESULTS_DIRECTORY / "evaluation_results.json"
)

ABLATION_RESULTS_PATH = (
    RESULTS_DIRECTORY / "ablation_results.json"
)

FIGURES_DIRECTORY = (
    RESULTS_DIRECTORY / "figures"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_figure(
    filename: str,
) -> None:
    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIRECTORY / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_latency_chart(
    results: list[dict[str, Any]],
) -> None:
    prompt_ids = [
        result["prompt_id"]
        for result in results
    ]

    latencies = [
        result["latency_ms"]
        for result in results
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(prompt_ids, latencies)

    plt.title(
        "Latency by Evaluation Prompt",
        fontweight="bold",
    )

    plt.xlabel(
        "Prompt ID",
        fontstyle="italic",
    )

    plt.ylabel(
        "Latency (ms)",
        fontstyle="italic",
    )

    plt.xticks(rotation=45)

    save_figure(
        "latency_by_prompt.png"
    )


def create_token_chart(
    results: list[dict[str, Any]],
) -> None:
    category_totals: dict[
        str,
        dict[str, float],
    ] = defaultdict(
        lambda: {
            "input": 0.0,
            "output": 0.0,
            "count": 0.0,
        }
    )

    for result in results:
        category = result["category"]

        category_totals[category]["input"] += (
            result["input_tokens"]
        )

        category_totals[category]["output"] += (
            result["output_tokens"]
        )

        category_totals[category]["count"] += 1

    categories = sorted(category_totals)

    average_input = [
        category_totals[category]["input"]
        / category_totals[category]["count"]
        for category in categories
    ]

    average_output = [
        category_totals[category]["output"]
        / category_totals[category]["count"]
        for category in categories
    ]

    positions = list(range(len(categories)))
    width = 0.38

    plt.figure(figsize=(10, 6))

    plt.bar(
        [
            position - width / 2
            for position in positions
        ],
        average_input,
        width=width,
        label="Input Tokens",
    )

    plt.bar(
        [
            position + width / 2
            for position in positions
        ],
        average_output,
        width=width,
        label="Output Tokens",
    )

    plt.title(
        "Average Token Usage by Category",
        fontweight="bold",
    )

    plt.xlabel(
        "Evaluation Category",
        fontstyle="italic",
    )

    plt.ylabel(
        "Average Tokens",
        fontstyle="italic",
    )

    plt.xticks(
        positions,
        categories,
        rotation=20,
    )

    plt.legend()

    save_figure(
        "token_usage_by_category.png"
    )


def create_success_chart(
    summary: dict[str, Any],
) -> None:
    categories = sorted(
        summary["categories"]
    )

    success_rates = [
        summary["categories"][category][
            "automatic_task_success_percent"
        ]
        for category in categories
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        categories,
        success_rates,
    )

    plt.ylim(0, 105)

    plt.title(
        "Automatic Task Success by Category",
        fontweight="bold",
    )

    plt.xlabel(
        "Evaluation Category",
        fontstyle="italic",
    )

    plt.ylabel(
        "Task Success (%)",
        fontstyle="italic",
    )

    plt.xticks(rotation=20)

    save_figure(
        "task_success_by_category.png"
    )


def create_tool_usage_chart(
    results: list[dict[str, Any]],
) -> None:
    tool_counter: Counter[str] = Counter()

    for result in results:
        response = result.get(
            "response",
            {},
        )

        for tool_call in response.get(
            "tool_calls",
            [],
        ):
            tool_counter[
                tool_call["tool_name"]
            ] += 1

    tool_names = list(tool_counter.keys())
    tool_counts = [
        tool_counter[tool_name]
        for tool_name in tool_names
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        tool_names,
        tool_counts,
    )

    plt.title(
        "Tool Calls During Evaluation",
        fontweight="bold",
    )

    plt.xlabel(
        "Tool",
        fontstyle="italic",
    )

    plt.ylabel(
        "Call Count",
        fontstyle="italic",
    )

    plt.xticks(rotation=15)

    save_figure(
        "tool_usage.png"
    )


def create_outcome_heatmap(
    results: list[dict[str, Any]],
) -> None:
    category_order = [
        "tool_use",
        "memory",
        "safety",
        "edge_case",
    ]

    matrix: list[list[int]] = []

    labels: list[list[str]] = []

    for category in category_order:
        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        matrix.append(
            [
                int(result["automatic_success"])
                for result in category_results
            ]
        )

        labels.append(
            [
                result["prompt_id"]
                for result in category_results
            ]
        )

    plt.figure(figsize=(9, 5))

    image = plt.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        ticks=[0, 1],
        label="Automatic Outcome",
    )

    plt.title(
        "Evaluation Outcome Heatmap",
        fontweight="bold",
    )

    plt.xlabel(
        "Prompt Position Within Category",
        fontstyle="italic",
    )

    plt.ylabel(
        "Evaluation Category",
        fontstyle="italic",
    )

    plt.yticks(
        range(len(category_order)),
        category_order,
    )

    plt.xticks(
        range(5),
        ["1", "2", "3", "4", "5"],
    )

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            plt.text(
                column_index,
                row_index,
                labels[row_index][column_index],
                ha="center",
                va="center",
                color="white" if value == 0 else "black",
            )

    save_figure(
        "evaluation_outcomes_heatmap.png"
    )


def create_ablation_chart(
    ablation: dict[str, Any],
) -> None:
    short_summary = ablation["short_memory"]["summary"]
    long_summary = ablation["long_memory"]["summary"]

    panels = [
        (
            "Task Success",
            "Task Success (%)",
            "task_success_percent",
            1.0,
        ),
        (
            "Average Latency",
            "Latency (s)",
            "average_latency_ms",
            1000.0,
        ),
        (
            "Average Token Usage",
            "Tokens per Prompt",
            "average_total_tokens",
            1.0,
        ),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    for axis, panel in zip(axes, panels):
        title, ylabel, metric, divisor = panel

        values = [
            short_summary[metric] / divisor,
            long_summary[metric] / divisor,
        ]

        axis.bar(
            ["2 turns", "8 turns"],
            values,
            color=["tab:blue", "tab:orange"],
        )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(
            0,
            max(values) * 1.15,
        )

        for position, value in enumerate(values):
            axis.annotate(
                f"{value:.2f}",
                xy=(position, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    figure.suptitle(
        "Memory Window Ablation",
        fontweight="bold",
    )

    save_figure(
        "memory_ablation.png"
    )


def main() -> None:
    evaluation = load_json(
        EVALUATION_RESULTS_PATH
    )

    ablation = load_json(
        ABLATION_RESULTS_PATH
    )

    results = evaluation["results"]
    summary = evaluation["summary"]

    create_latency_chart(results)
    create_token_chart(results)
    create_success_chart(summary)
    create_tool_usage_chart(results)
    create_outcome_heatmap(results)
    create_ablation_chart(ablation)

    print(
        f"Charts created in {FIGURES_DIRECTORY}"
    )


if __name__ == "__main__":
    main()