import logging
logger = logging.getLogger("ingestion.chunks_generation.config")
logger.info("Loading file...")

from dataclasses import dataclass
from enum import Enum

class OverlapGranularity(str, Enum):
    SENTENCE_BASED = "sentence_based"
    WORD_BASED = "word_based"

@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 350
    chunk_overlap_pct: int = 15
    separate_h2s: bool = True
    overlap_granularity: OverlapGranularity = OverlapGranularity.SENTENCE_BASED
    cross_section_overlap: bool = False
    silent: bool = True
    resume: bool = True