from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field


class TurnMetric(BaseModel):
    """
    Persistent measurements for one completed chatbot turn.
    """

    timestamp_utc: str

    conversation_id: str
    model: str
    response_id: str | None = None

    user_message: str
    assistant_text: str

    latency_ms: float

    input_tokens: int
    output_tokens: int
    total_tokens: int

    estimated_cost_usd: float

    model_rounds: int
    memory_turns: int

    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tool_call_count: int
    tool_success_count: int
    tool_failure_count: int

    completed: bool = True
    error_type: str | None = None
    error_message: str | None = None


class MetricsSummary(BaseModel):
    """
    Aggregate metrics written to results/metrics.json.
    """

    generated_at_utc: str

    total_turns: int
    successful_turns: int
    failed_turns: int

    task_success_percent: float | None = None
    hallucination_rate_percent: float | None = None

    average_latency_ms: float
    average_input_tokens: float
    average_output_tokens: float
    average_total_tokens: float

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    total_estimated_cost_usd: float

    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int


class MetricsService:
    """
    Thread-safe JSON/JSONL persistence for turn measurements and
    conversation transcripts.
    """

    def __init__(
        self,
        results_directory: Path,
        input_cost_per_million: float,
        output_cost_per_million: float,
    ) -> None:
        if input_cost_per_million < 0:
            raise ValueError(
                "input_cost_per_million cannot be negative."
            )

        if output_cost_per_million < 0:
            raise ValueError(
                "output_cost_per_million cannot be negative."
            )

        self._results_directory = results_directory
        self._transcripts_directory = (
            results_directory / "transcripts"
        )

        self._turn_metrics_path = (
            results_directory / "turn_metrics.jsonl"
        )

        self._summary_path = (
            results_directory / "metrics.json"
        )

        self._input_cost_per_million = (
            input_cost_per_million
        )

        self._output_cost_per_million = (
            output_cost_per_million
        )

        self._lock = RLock()

        self._results_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._transcripts_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    @property
    def turn_metrics_path(self) -> Path:
        return self._turn_metrics_path

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        input_cost = (
            input_tokens
            / 1_000_000
            * self._input_cost_per_million
        )

        output_cost = (
            output_tokens
            / 1_000_000
            * self._output_cost_per_million
        )

        return round(input_cost + output_cost, 8)

    def create_turn_metric(
        self,
        *,
        conversation_id: str,
        model: str,
        response_id: str | None,
        user_message: str,
        assistant_text: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model_rounds: int,
        memory_turns: int,
        tool_calls: list[dict[str, Any]],
        completed: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> TurnMetric:
        successful_tools = sum(
            1
            for tool_call in tool_calls
            if tool_call.get("success") is True
        )

        failed_tools = (
            len(tool_calls) - successful_tools
        )

        return TurnMetric(
            timestamp_utc=self._utc_now(),
            conversation_id=conversation_id,
            model=model,
            response_id=response_id,
            user_message=user_message,
            assistant_text=assistant_text,
            latency_ms=round(latency_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self.calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            model_rounds=model_rounds,
            memory_turns=memory_turns,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            tool_success_count=successful_tools,
            tool_failure_count=failed_tools,
            completed=completed,
            error_type=error_type,
            error_message=error_message,
        )

    def record_turn(
        self,
        metric: TurnMetric,
    ) -> None:
        with self._lock:
            self._append_json_line(
                self._turn_metrics_path,
                metric.model_dump(),
            )

            transcript_path = (
                self._transcripts_directory
                / f"{self._safe_filename(metric.conversation_id)}.jsonl"
            )

            self._append_json_line(
                transcript_path,
                metric.model_dump(),
            )

            self._write_summary()

    def load_turn_metrics(self) -> list[TurnMetric]:
        if not self._turn_metrics_path.exists():
            return []

        metrics: list[TurnMetric] = []

        with self._turn_metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                cleaned_line = line.strip()

                if not cleaned_line:
                    continue

                metrics.append(
                    TurnMetric.model_validate_json(
                        cleaned_line
                    )
                )

        return metrics

    def generate_summary(self) -> MetricsSummary:
        metrics = self.load_turn_metrics()

        if not metrics:
            return MetricsSummary(
                generated_at_utc=self._utc_now(),
                total_turns=0,
                successful_turns=0,
                failed_turns=0,
                average_latency_ms=0.0,
                average_input_tokens=0.0,
                average_output_tokens=0.0,
                average_total_tokens=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_estimated_cost_usd=0.0,
                total_tool_calls=0,
                successful_tool_calls=0,
                failed_tool_calls=0,
            )

        successful_turns = sum(
            1 for metric in metrics if metric.completed
        )

        total_input_tokens = sum(
            metric.input_tokens
            for metric in metrics
        )

        total_output_tokens = sum(
            metric.output_tokens
            for metric in metrics
        )

        total_tokens = sum(
            metric.total_tokens
            for metric in metrics
        )

        total_cost = sum(
            metric.estimated_cost_usd
            for metric in metrics
        )

        total_tool_calls = sum(
            metric.tool_call_count
            for metric in metrics
        )

        successful_tool_calls = sum(
            metric.tool_success_count
            for metric in metrics
        )

        failed_tool_calls = sum(
            metric.tool_failure_count
            for metric in metrics
        )

        count = len(metrics)

        return MetricsSummary(
            generated_at_utc=self._utc_now(),
            total_turns=count,
            successful_turns=successful_turns,
            failed_turns=count - successful_turns,
            average_latency_ms=round(
                sum(
                    metric.latency_ms
                    for metric in metrics
                )
                / count,
                3,
            ),
            average_input_tokens=round(
                total_input_tokens / count,
                3,
            ),
            average_output_tokens=round(
                total_output_tokens / count,
                3,
            ),
            average_total_tokens=round(
                total_tokens / count,
                3,
            ),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            total_estimated_cost_usd=round(
                total_cost,
                8,
            ),
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_tool_calls,
            failed_tool_calls=failed_tool_calls,
        )

    def update_manual_scores(
        self,
        task_success_percent: float,
        hallucination_rate_percent: float,
    ) -> MetricsSummary:
        if not 0 <= task_success_percent <= 100:
            raise ValueError(
                "task_success_percent must be from 0 to 100."
            )

        if not 0 <= hallucination_rate_percent <= 100:
            raise ValueError(
                "hallucination_rate_percent must be from 0 to 100."
            )

        summary = self.generate_summary()

        summary.task_success_percent = round(
            task_success_percent,
            2,
        )

        summary.hallucination_rate_percent = round(
            hallucination_rate_percent,
            2,
        )

        self._summary_path.write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return summary

    def _write_summary(self) -> None:
        existing_manual_scores = self._load_manual_scores()

        summary = self.generate_summary()

        summary.task_success_percent = (
            existing_manual_scores[0]
        )

        summary.hallucination_rate_percent = (
            existing_manual_scores[1]
        )

        self._summary_path.write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _load_manual_scores(
        self,
    ) -> tuple[float | None, float | None]:
        if not self._summary_path.exists():
            return None, None

        try:
            data = json.loads(
                self._summary_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None, None

        return (
            data.get("task_success_percent"),
            data.get("hallucination_rate_percent"),
        )

    @staticmethod
    def _append_json_line(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        with path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe_characters = []

        for character in value:
            if character.isalnum() or character in {
                "-",
                "_",
            }:
                safe_characters.append(character)
            else:
                safe_characters.append("_")

        cleaned_value = "".join(safe_characters)

        return cleaned_value[:100] or "conversation"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()