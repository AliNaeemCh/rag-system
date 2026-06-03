import logging
logger = logging.getLogger("app.models")
logger.info("Loading file...")

from dataclasses import dataclass
from typing import Any
from enum import Enum

class RetrievalType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"

@dataclass(slots=True)
class ScoreBreakdown:
    retrieval_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

@dataclass(slots=True)
class RetrievedDocument:
    id: int
    content: str
    metadata: dict
    retrieval_type: RetrievalType
    scores: ScoreBreakdown