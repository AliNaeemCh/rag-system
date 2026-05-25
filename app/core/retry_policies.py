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
from huggingface_hub.errors import HfHubHTTPError
import requests
from typing import Type
import httpx
import httpcore


# ---------------- CONFIG ----------------
@dataclass
class RetryConfig:
    max_attempts: int = 5
    initial_wait: float = 1
    max_wait: float = 20
    jitter: float = 0.5

# ----------------------------------------

def build_retry(
    logger: logging.Logger,
    retry_exceptions: Type[BaseException] | tuple[Type[BaseException], ...],
    config:  RetryConfig | None = None
):
    if config is None:
        config = RetryConfig()
        
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential_jitter(
            initial=config.initial_wait,
            max=config.max_wait,
            jitter=config.jitter,
        ),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

def openai_retry(logger: logging.Logger, config: RetryConfig | None =None):
    return build_retry(
        logger,
        retry_exceptions=(RateLimitError, APIError, APITimeoutError),
        config=config,
    )


def huggingface_retry(logger: logging.Logger, config: RetryConfig | None = None):
    return build_retry(
        logger,
        retry_exceptions=(
            HfHubHTTPError,
            TimeoutError,
            requests.exceptions.RequestException,
            httpx.ConnectError,
            httpcore.ConnectError,
        ),
        config=config
    )