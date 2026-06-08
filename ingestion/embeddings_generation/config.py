from app.core.config import settings

import logging
logger = logging.getLogger("ingestion.embeddings_generation.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class EmbeddingPipelineConfig:
    batch_size: int = settings.CHUNKS_EMBEDDING_BATCH_SIZE
    embedding_dim: int = settings.EMBEDDING_DIMENSIONS
    m: int = settings.HNSW_M
    ef_construction: int = settings.HNSW_EF_CONSTRUCTION
    num_workers: int = 2
    resume: bool = True
