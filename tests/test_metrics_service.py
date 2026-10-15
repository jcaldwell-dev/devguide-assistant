from pathlib import Path

import pytest

from app.services.metrics_service import MetricsService


def create_service(
    tmp_path: Path,
) -> MetricsService:
    return MetricsService(
        results_directory=tmp_path / "results",
        input_cost_per_million=0.75,
        output_cost_per_million=4.50,
    )


def test_cost_calculation(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    cost = service.calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == 5.25


def test_turn_is_written_to_jsonl(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    metric = service.create_turn_metric(
        conversation_id="chat-1",
        model="test-model",
        response_id="response-1",
        user_message="Hello",
        assistant_text="Hello.",
        latency_ms=125.4567,
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        model_rounds=1,
        memory_turns=1,
        tool_calls=[],
    )

    service.record_turn(metric)

    assert service.turn_metrics_path.exists()

    loaded = service.load_turn_metrics()

    assert len(loaded) == 1
    assert loaded[0].conversation_id == "chat-1"
    assert loaded[0].latency_ms == 125.457


def test_transcript_file_is_created(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    metric = service.create_turn_metric(
        conversation_id="chat/unsafe:name",
        model="test-model",
        response_id=None,
        user_message="Question",
        assistant_text="Answer",
        latency_ms=10,
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
        model_rounds=1,
        memory_turns=1,
        tool_calls=[],
    )

    service.record_turn(metric)

    transcript = (
        tmp_path
        / "results"
        / "transcripts"
        / "chat_unsafe_name.jsonl"
    )

    assert transcript.exists()


def test_summary_aggregates_metrics(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    for number in range(2):
        metric = service.create_turn_metric(
            conversation_id="chat-1",
            model="test-model",
            response_id=f"response-{number}",
            user_message="Question",
            assistant_text="Answer",
            latency_ms=100 + number * 100,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model_rounds=1,
            memory_turns=number + 1,
            tool_calls=[],
        )

        service.record_turn(metric)

    summary = service.generate_summary()

    assert summary.total_turns == 2
    assert summary.successful_turns == 2
    assert summary.average_latency_ms == 150.0
    assert summary.total_tokens == 240


def test_tool_counts_are_recorded(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    metric = service.create_turn_metric(
        conversation_id="chat-1",
        model="test-model",
        response_id="response-1",
        user_message="Search",
        assistant_text="Found",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        model_rounds=2,
        memory_turns=1,
        tool_calls=[
            {
                "tool_name": "search_project_kb",
                "success": True,
            },
            {
                "tool_name": "missing_tool",
                "success": False,
            },
        ],
    )

    assert metric.tool_call_count == 2
    assert metric.tool_success_count == 1
    assert metric.tool_failure_count == 1


def test_manual_scores_update_summary(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    summary = service.update_manual_scores(
        task_success_percent=90.0,
        hallucination_rate_percent=5.0,
    )

    assert summary.task_success_percent == 90.0
    assert summary.hallucination_rate_percent == 5.0
    assert service.summary_path.exists()


def test_invalid_manual_score_is_rejected(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    with pytest.raises(ValueError):
        service.update_manual_scores(
            task_success_percent=101,
            hallucination_rate_percent=0,
        )