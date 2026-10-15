from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from threading import RLock
from time import monotonic

from pydantic import BaseModel


class RateLimitDecision(BaseModel):
    """
    Result of checking one client against the rate limit.
    """

    allowed: bool
    remaining_requests: int
    retry_after_seconds: float


class RateLimitService:
    """
    Thread-safe in-memory fixed-window rate limiter.

    Requests are tracked independently by caller-provided keys.
    """

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError(
                "max_requests must be at least 1."
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be greater than 0."
            )

        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock

        self._requests: dict[
            str,
            deque[float],
        ] = defaultdict(deque)

        self._lock = RLock()

    @property
    def max_requests(self) -> int:
        return self._max_requests

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def check(
        self,
        client_key: str,
    ) -> RateLimitDecision:
        """
        Records one request if capacity remains.
        """

        cleaned_key = client_key.strip()

        if not cleaned_key:
            raise ValueError(
                "client_key cannot be blank."
            )

        current_time = self._clock()
        cutoff_time = (
            current_time - self._window_seconds
        )

        with self._lock:
            timestamps = self._requests[cleaned_key]

            while (
                timestamps
                and timestamps[0] <= cutoff_time
            ):
                timestamps.popleft()

            if len(timestamps) >= self._max_requests:
                retry_after = max(
                    0.0,
                    self._window_seconds
                    - (current_time - timestamps[0]),
                )

                return RateLimitDecision(
                    allowed=False,
                    remaining_requests=0,
                    retry_after_seconds=round(
                        retry_after,
                        3,
                    ),
                )

            timestamps.append(current_time)

            return RateLimitDecision(
                allowed=True,
                remaining_requests=(
                    self._max_requests
                    - len(timestamps)
                ),
                retry_after_seconds=0.0,
            )

    def clear(
        self,
        client_key: str | None = None,
    ) -> int:
        """
        Clears one client or every tracked client.
        """

        with self._lock:
            if client_key is None:
                removed_count = len(self._requests)
                self._requests.clear()
                return removed_count

            return int(
                self._requests.pop(
                    client_key,
                    None,
                )
                is not None
            )