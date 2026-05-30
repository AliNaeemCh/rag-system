from app.core.config import settings

import logging
logger = logging.getLogger("app.ingestion.embeddings_generation.config")
logger.info("Loading file...")

from pathlib import Path
from dataclasses import dataclass

@dataclass(slots=True)
class EmbeddingPipelineConfig:
    chunks_jsonl_path: Path
    batch_size: int = settings.CHUNKS_EMBEDDING_BATCH_SIZE
    embedding_model: str = settings.EMBEDDING_MODEL
    embedding_dim: int = settings.EMBEDDING_DIMENSIONS
    m: int = settings.PGVECTOR_HNSW_M
    ef_construction: int = settings.PGVECTOR_HNSW_EF_CONSTRUCTION
    num_workers: int = 4
    resume: bool = True
