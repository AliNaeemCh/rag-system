import logging
logger = logging.getLogger("app.infra.usage_tracking.buckets")
logger.info("Loading file...")

from enum import Enum

class Bucket(str, Enum):
    SMALL = "small"
    LARGE = "large"

BUCKET_TO_TOKEN_LIMIT = {
    Bucket.LARGE: 2.5e6,    # 2.5M
    Bucket.SMALL: 250e3     # 250k
}

small_bucket_models = {
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5",
    "gpt-5-codex",
    "gpt-5-chat-latest",
    "gpt-4.1",
    "gpt-4o",
    "o1",
    "o3",
}

large_bucket_models = {
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "o1-mini",
    "o3-mini",
    "o4-mini",
    "codex-mini-latest",
}

def get_bucket(model_name: str) -> Bucket:
    if model_name in large_bucket_models:
        return Bucket.LARGE
    if model_name in small_bucket_models:
        return Bucket.SMALL
    raise ValueError(f"Unknown model: {model_name}")