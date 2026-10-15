from evaluation.evaluate import (
    automatically_score,
    contains_all_terms,
    contains_any_term,
    load_prompts,
    summarize_results,
)


def test_required_prompt_count_is_twenty() -> None:
    prompts = load_prompts()

    assert len(prompts) == 20


def test_prompt_categories_have_five_entries() -> None:
    prompts = load_prompts()

    categories = {
        category: sum(
            1
            for prompt in prompts
            if prompt["category"] == category
        )
        for category in {
            "tool_use",
            "memory",
            "safety",
            "edge_case",
        }
    }

    assert categories == {
        "tool_use": 5,
        "memory": 5,
        "safety": 5,
        "edge_case": 5,
    }


def test_term_matching_is_case_insensitive() -> None:
    assert contains_all_terms(
        "Use the BPC_ prefix.",
        ["bpc_"],
    )

    assert contains_any_term(
        "The task has Moderate risk.",
        ["moderate", "critical"],
    )


def test_automatic_tool_score_passes() -> None:
    prompt = {
        "expected_status": 200,
        "expected_tools": [
            "search_project_kb"
        ],
        "required_terms": ["BPC_"],
    }

    response = {
        "text": "Use the BPC_ prefix.",
        "tool_calls": [
            {
                "tool_name": (
                    "search_project_kb"
                )
            }
        ],
        "model_rounds": 2,
        "total_tokens": 100,
    }

    success, failures = automatically_score(
        prompt,
        200,
        response,
    )

    assert success is True
    assert failures == []


def test_local_refusal_score_requires_zero_tokens() -> None:
    prompt = {
        "expected_status": 200,
        "expected_refusal": True,
        "required_any_terms": [
            "cannot reveal"
        ],
    }

    response = {
        "text": (
            "I cannot reveal application secrets."
        ),
        "tool_calls": [],
        "model_rounds": 0,
        "total_tokens": 0,
    }

    success, failures = automatically_score(
        prompt,
        200,
        response,
    )

    assert success is True
    assert failures == []


def test_summary_calculates_success_rate() -> None:
    results = [
        {
            "category": "tool_use",
            "automatic_success": True,
            "latency_ms": 100.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": 0.01,
        },
        {
            "category": "tool_use",
            "automatic_success": False,
            "latency_ms": 200.0,
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "estimated_cost_usd": 0.02,
        },
    ]

    summary = summarize_results(results)

    assert summary["task_success_percent"] == 50.0
    assert summary["average_latency_ms"] == 150.0
    assert summary["total_tokens"] == 45