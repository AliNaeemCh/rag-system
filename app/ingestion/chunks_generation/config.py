from app.core.config import settings, OverlapGranularity

import logging
logger = logging.getLogger("app.ingestion.chunks_generation.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap_pct: int = settings.OVERLAP_TOKENS_PCT
    separate_h2s: bool = settings.SEPARATE_H2s
    overlap_granularity: OverlapGranularity = settings.OVERLAP_GRANULARITY
    cross_section_overlap: bool = settings.CROSS_SECTION_OVERLAP
    silent: bool = True
    resume: bool = True