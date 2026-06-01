from app.infra.vector_stores.base import BaseVectorStore
from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.infra.db.pool import db_pool

import logging
logger = logging.getLogger("rag.retriever")
logger.info("Loading file...")

from psycopg2.extensions import connection

class Retriever:
    def __init__(self, embedding_model: BaseEmbeddingProvider, vector_store: BaseVectorStore, top_k: int):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, filters: dict | None = None, normalize_query: bool = True, ef_search: int | None = None):
        """
        1. Embed query
        2. Search vector DB
        3. Return top-k docs
        """
        ef_search = ef_search or settings.PGVECTOR_HNSW_EF_SEARCH

        logger.info(f"Retrieval started")

        # 1. Get embedding
        query_embedding = self.embedding_model.embed_query(query, normalize=normalize_query)

        # 2. Search vector store
        docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filters=filters,
            ef_search=ef_search
        )

        logger.info(f"Retrieval completed")

        return docs