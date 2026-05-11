from dataclasses import dataclass
from enum import Enum
from app.core.config import settings, OverlapGranularity

@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap_pct: int = settings.OVERLAP_TOKENS_PCT
    separate_h2s: bool = settings.SEPARATE_H2s
    overlap_granularity: OverlapGranularity = settings.OVERLAP_GRANULARITY
    silent: bool = True