from app.rag.config import ResponseMode
import logging
logger = logging.getLogger("evaluation.inference.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class EvalInferenceConfig:
    num_workers: int = 3
    resume: bool = True
    response_mode: ResponseMode = ResponseMode.ADVANCED
    generator_temperature: float = 0
    rewriter_temperature: float = 0
