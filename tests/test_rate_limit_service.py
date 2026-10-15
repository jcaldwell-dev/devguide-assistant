import pytest

from app.services.rate_limit_service import (
    RateLimitService,
)


class FakeClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time

    def advance(
        self,
        seconds: float,
    ) -> None:
        self.current_time += seconds


def test_requests_within_limit_are_allowed() -> None:
    clock = FakeClock()

    service = RateLimitService(
        max_requests=2,
        window_seconds=60,
        clock=clock,
    )

    assert service.check("client").allowed is True
    assert service.check("client").allowed is True


def test_request_over_limit_is_rejected() -> None:
    clock = FakeClock()

    service = RateLimitService(
        max_requests=1,
        window_seconds=60,
        clock=clock,
    )

    service.check("client")
    decision = service.check("client")

    assert decision.allowed is False
    assert decision.retry_after_seconds == 60.0


def test_limit_resets_after_window() -> None:
    clock = FakeClock()

    service = RateLimitService(
        max_requests=1,
        window_seconds=60,
        clock=clock,
    )

    service.check("client")
    clock.advance(61)

    assert service.check("client").allowed is True


def test_clients_are_isolated() -> None:
    service = RateLimitService(
        max_requests=1,
        window_seconds=60,
    )

    assert service.check("client-a").allowed is True
    assert service.check("client-b").allowed is True


def test_blank_client_key_is_rejected() -> None:
    service = RateLimitService()

    with pytest.raises(ValueError):
        service.check(" ")


def test_clear_removes_client_history() -> None:
    service = RateLimitService(
        max_requests=1,
    )

    service.check("client")
    assert service.check("client").allowed is False

    assert service.clear("client") == 1
    assert service.check("client").allowed is True