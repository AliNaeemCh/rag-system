import logging
logger = logging.getLogger("app.rag.config")
logger.info("Loading file...")

from enum import Enum

class ResponseMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ADVANCED = "advanced"