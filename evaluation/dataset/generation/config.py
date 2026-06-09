from app.core.config import settings

import logging
logger = logging.getLogger("evaluation.dataset.generation.config")
logger.info("Loading file...")

from dataclasses import dataclass
from enum import Enum

class EvalQuestionType(str, Enum):
    FACTUAL = "factual"
    INFERENCE = "inference"
    MULTI_CHUNK = "multi_chunk"     # Factual question based on multiple chunks
    OUT_OF_KNOWLEDGE = "out_of_knowledge"

@dataclass(slots=True)
class EvalDatasetGeneratorConfig:
    eval_set_size: int = 200
    multi_chunk_max_qs: int = settings.FINAL_TOP_K
    num_workers: int = 3
    seed: int = 11
    resume: bool = True
