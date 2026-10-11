"""
Tests for exponential backoff retry mechanism, jitter calculation, and error classification.
"""

import pytest
from src.paper_to_voice.utils.retry import (
    RetryExhaustedError,
    calculate_backoff_delay,
    extract_retry_after,
    is_permanent_error,
    is_transient_error,
    retry_with_backoff,
    sanitize_log_message,
)


def test_retry_transient_failure_then_success():
    """Verify that transient failures are retried and success is returned."""
    attempts = 0
    sleeps = []

    def mock_op():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("503 Service Unavailable - transient overload")
        return "SUCCESS"

    result = retry_with_backoff(
        mock_op,
        operation_name="test_transient_success",
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0,
        sleeper=lambda d: sleeps.append(d),
    )

    assert result == "SUCCESS"
    assert attempts == 3
    assert len(sleeps) == 2
    assert all(d >= 0.0 for d in sleeps)


def test_retry_permanent_failure_no_retries():
    """Verify that permanent errors fail immediately without retry delays."""
    attempts = 0
    sleeps = []

    def mock_op():
        nonlocal attempts
        attempts += 1
        raise ValueError("Invalid argument: bad request payload")

    with pytest.raises(ValueError, match="Invalid argument"):
        retry_with_backoff(
            mock_op,
            operation_name="test_permanent_failure",
            max_retries=3,
            sleeper=lambda d: sleeps.append(d),
        )

    assert attempts == 1
    assert len(sleeps) == 0


def test_retry_exhaustion_raises_explicit_error():
    """Verify that persistent transient errors exhaust the retry limit and raise RetryExhaustedError."""
    attempts = 0
    sleeps = []

    def mock_op():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("429 Too Many Requests - rate limit exceeded")

    with pytest.raises(RetryExhaustedError) as exc_info:
        retry_with_backoff(
            mock_op,
            operation_name="test_exhaustion",
            max_retries=2,
            base_delay=0.1,
            max_delay=0.5,
            sleeper=lambda d: sleeps.append(d),
        )

    assert attempts == 3  # Initial + 2 retries
    assert len(sleeps) == 2
    assert exc_info.value.attempts == 3
    assert exc_info.value.operation == "test_exhaustion"


def test_jitter_backoff_calculation_and_bounds():
    """Verify exponential delay scaling, random jitter variation, and maximum bound."""
    base_delay = 1.0
    max_delay = 10.0

    delays_attempt_0 = [calculate_backoff_delay(0, base_delay, max_delay) for _ in range(50)]
    delays_attempt_2 = [calculate_backoff_delay(2, base_delay, max_delay) for _ in range(50)]
    delays_attempt_10 = [calculate_backoff_delay(10, base_delay, max_delay) for _ in range(50)]

    # Attempt 0 cap is 1.0; full jitter is in [0, 1.0]
    assert all(0.0 <= d <= 1.0 for d in delays_attempt_0)
    # Attempt 2 cap is min(10.0, 4.0) = 4.0; full jitter is in [0, 4.0]
    assert all(0.0 <= d <= 4.0 for d in delays_attempt_2)
    # Attempt 10 cap is min(10.0, 1024.0) = 10.0; full jitter is in [0, 10.0]
    assert all(0.0 <= d <= max_delay for d in delays_attempt_10)

    # Randomized jitter should produce multiple distinct values
    assert len(set(round(d, 4) for d in delays_attempt_2)) > 1


def test_respects_retry_after():
    """Verify that Retry-After header / attribute is respected."""
    class ErrorWithRetryAfter(Exception):
        retry_after = 4.5

    err = ErrorWithRetryAfter("Rate limit")
    extracted = extract_retry_after(err)
    assert extracted == 4.5

    delay = calculate_backoff_delay(0, base_delay=1.0, max_delay=10.0, retry_after=extracted)
    assert delay == 4.5


def test_sensitive_data_redaction():
    """Verify that credentials and API keys are redacted from logs."""
    raw = "Failed with key=AIzaSyA12345678901234567890123456789012 and secret=supersecret123"
    sanitized = sanitize_log_message(raw)
    assert "AIzaSy" not in sanitized
    assert "supersecret123" not in sanitized
    assert "***REDACTED***" in sanitized
