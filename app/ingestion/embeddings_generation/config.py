from app.core.config import settings

class EmbeddingPipelineConfig:
    chunks_jsonl_path: str
    batch_size: int = settings.CHUNKS_EMBEDDING_BATCH_SIZE
    embedding_model: str = settings.EMBEDDING_MODEL
    embedding_dim: int = settings.EMBEDDING_DIMENSIONS
    m: int = settings.PGVECTOR_HNSW_M
    ef_construction: int = settings.PGVECTOR_HNSW_EF_CONSTRUCTION