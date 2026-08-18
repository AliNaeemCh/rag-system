from app.rag.config import ResponseMode

import logging
logger = logging.getLogger("evaluation.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class EvalConfig:
    max_concurrency: int = 3
    resume: bool = True
    response_mode: ResponseMode = ResponseMode.ADVANCED
    generator_temperature: float = 0
    rewriter_temperature: float = 0
    retrieval_eval_temperature: float = 0
    generation_eval_temperature: float = 0
