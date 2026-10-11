"""
Exponential backoff retry mechanism with randomized jitter for transient API failures.
"""

import logging
import random
import time
import re
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger("paper_to_voice.retry")

# Common HTTP status codes considered transient
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
PERMANENT_HTTP_STATUS_CODES = {400, 401, 403, 404, 422}

# Sensitive keys to sanitize in log messages
SENSITIVE_PATTERNS = [
    re.compile(r'(key|token|secret|password|authorization)=([^\s&]+)', re.IGNORECASE),
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),  # Google API key pattern
]


def sanitize_log_message(msg: str) -> str:
    """Remove sensitive API keys or credentials from error strings."""
    sanitized = str(msg)
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(r'\1=***REDACTED***' if pattern.groups else '***REDACTED***', sanitized)
    return sanitized


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have failed."""

    def __init__(self, operation: str, attempts: int, last_exception: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_exception = last_exception
        sanitized_msg = sanitize_log_message(str(last_exception))
        super().__init__(
            f"Operation '{operation}' failed after {attempts} attempts. Last error: {sanitized_msg}"
        )


def extract_retry_after(exc: Exception) -> Optional[float]:
    """
    Attempt to extract Retry-After duration in seconds from an exception or its response headers.
    """
    # Direct attribute on custom exceptions
    if hasattr(exc, "retry_after") and isinstance(exc.retry_after, (int, float)):
        return float(exc.retry_after)

    # From response headers (requests / httpx)
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        retry_after_header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if retry_after_header:
            try:
                return max(0.0, float(retry_after_header))
            except (ValueError, TypeError):
                pass
    return None


def is_permanent_error(exc: Exception) -> bool:
    """Determine if an exception represents a known non-transient, permanent error."""
    # Check status code attributes
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int) and status_code in PERMANENT_HTTP_STATUS_CODES:
        return True

    exc_name = exc.__class__.__name__
    permanent_names = {
        "InvalidArgument",
        "PermissionDenied",
        "Unauthenticated",
        "NotFound",
        "BadRequest",
        "Unauthorized",
        "Forbidden",
        "ValidationError",
        "JSONDecodeError",
        "KeyError",
        "TypeError",
        "ValueError",
    }
    if exc_name in permanent_names:
        return True

    # Inspect message for auth / permission errors
    msg_lower = str(exc).lower()
    if any(phrase in msg_lower for phrase in ["invalid api key", "unauthorized", "permission denied", "forbidden", "not found"]):
        return True

    return False


def is_transient_error(exc: Exception) -> bool:
    """Determine if an exception represents a transient failure that is safe to retry."""
    if is_permanent_error(exc):
        return False

    # Check status code
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int) and status_code in TRANSIENT_HTTP_STATUS_CODES:
        return True

    exc_name = exc.__class__.__name__
    transient_names = {
        "ResourceExhausted",
        "InternalServerError",
        "ServiceUnavailable",
        "Unavailable",
        "DeadlineExceeded",
        "Timeout",
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError",
        "NetworkError",
        "AppError",
    }
    if exc_name in transient_names:
        return True

    # Inspect error message for rate limit or transient network markers
    msg_lower = str(exc).lower()
    transient_keywords = [
        "rate limit",
        "quota exceeded",
        "resource exhausted",
        "429",
        "503",
        "502",
        "504",
        "500",
        "temporary failure",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "server error",
        "overloaded",
    ]
    return any(keyword in msg_lower for keyword in transient_keywords)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate backoff delay using exponential backoff with randomized full jitter.
    
    Formula:
        exponential_cap = min(max_delay, base_delay * (2 ** attempt))
        jittered_delay = random.uniform(0, exponential_cap)
    If Retry-After is present, it takes precedence up to max_delay.
    """
    if retry_after is not None and retry_after > 0:
        return min(max_delay, float(retry_after))

    exponential_cap = min(max_delay, base_delay * (2 ** attempt))
    # Full jitter: uniformly distributed between 0 and exponential_cap
    return random.uniform(0.0, max(0.0, exponential_cap))


def retry_with_backoff(
    func: Callable[[], T],
    operation_name: str = "operation",
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    """
    Execute a callable with bounded exponential backoff and randomized jitter for transient errors.
    
    Args:
        func: Zero-argument callable to execute.
        operation_name: Human-readable identifier for structured logging.
        max_retries: Maximum number of retry attempts (bounds total executions to max_retries + 1).
        base_delay: Initial backoff delay in seconds.
        max_delay: Upper bound on delay in seconds.
        retryable_exceptions: Optional explicit tuple of exception types to retry.
        non_retryable_exceptions: Optional explicit tuple of exception types to NEVER retry.
        sleeper: Callable used for sleeping (defaults to time.sleep, customizable for testing).
        
    Returns:
        The return value of func().
        
    Raises:
        RetryExhaustedError: When max_retries are exhausted on transient errors.
        Exception: The immediate exception if it is a non-retryable / permanent error.
    """
    attempt = 0
    total_allowed_attempts = max(1, max_retries + 1)

    while True:
        try:
            result = func()
            if attempt > 0:
                logger.info(
                    "Operation '%s' succeeded on attempt %d/%d",
                    operation_name,
                    attempt + 1,
                    total_allowed_attempts,
                )
            return result
        except Exception as exc:
            attempt += 1
            sanitized_err = sanitize_log_message(str(exc))
            exc_type = exc.__class__.__name__

            # Check explicit non-retryable types first
            if non_retryable_exceptions and isinstance(exc, non_retryable_exceptions):
                logger.warning(
                    "Operation '%s' encountered non-retryable exception %s: %s",
                    operation_name,
                    exc_type,
                    sanitized_err,
                )
                raise exc

            # Check explicit retryable types, or fall back to classification
            can_retry = False
            if retryable_exceptions and isinstance(exc, retryable_exceptions):
                can_retry = True
            elif not is_permanent_error(exc) and is_transient_error(exc):
                can_retry = True

            if not can_retry:
                logger.warning(
                    "Operation '%s' failed with permanent error %s (not retrying): %s",
                    operation_name,
                    exc_type,
                    sanitized_err,
                )
                raise exc

            if attempt >= total_allowed_attempts:
                logger.error(
                    "Operation '%s' exhausted all %d attempts. Final error: %s",
                    operation_name,
                    total_allowed_attempts,
                    sanitized_err,
                )
                raise RetryExhaustedError(operation_name, total_allowed_attempts, exc) from exc

            retry_after = extract_retry_after(exc)
            delay = calculate_backoff_delay(attempt - 1, base_delay, max_delay, retry_after)
            
            logger.info(
                "Operation '%s' transient failure on attempt %d/%d (%s: %s). Retrying in %.2fs (retry_after=%s)...",
                operation_name,
                attempt,
                total_allowed_attempts,
                exc_type,
                sanitized_err,
                delay,
                retry_after,
            )
            sleeper(delay)


def with_retry(
    operation_name: str = "operation",
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """Decorator version of retry_with_backoff."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                lambda: fn(*args, **kwargs),
                operation_name=operation_name,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions,
            )
        return wrapper
    return decorator
