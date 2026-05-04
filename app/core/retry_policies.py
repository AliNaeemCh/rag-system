from dataclasses import dataclass
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import RateLimitError, APIError, APITimeoutError


# ---------------- CONFIG ----------------
@dataclass
class RetryConfig:
    max_attempts: int = 5
    initial_wait: float = 1
    max_wait: float = 20
    jitter: float = 0.5

# ----------------------------------------

def openai_retry(
    logger: logging.Logger,
    config: RetryConfig | None = None
):
    if config is None:
        config = RetryConfig()

    return retry(
        stop=stop_after_attempt(config.max_attempts),

        wait=wait_exponential_jitter(
            initial=config.initial_wait,
            max=config.max_wait,
            jitter=config.jitter
        ),

        retry=retry_if_exception_type(
            (RateLimitError, APIError, APITimeoutError)
        ),

        reraise=True,

        before_sleep=before_sleep_log(logger, logging.WARNING),
    )