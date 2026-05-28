from enum import Enum

class ResponseMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ADVANCED = "advanced"