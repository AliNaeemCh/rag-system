import logging
from app.infra.vector_stores.pgvector_store import PgVectorStore
from app.infra.embeddings.hugging_face import HuggingFaceEmbeddingProvider
from app.core.config import settings

logger = logging.getLogger("rag.retriever")


class Retriever:
    def __init__(self, embedding_model: HuggingFaceEmbeddingProvider, vector_store: PgVectorStore, top_k: int):
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

        logger.debug(f"Retrieval started | Query = {query}")

        # 1. Get embedding
        query_embedding = self.embedding_model.embed_query(query, normalize=normalize_query)

        logger.debug(f"Query embedded | Embedding = {query_embedding}")

        # 2. Search vector store
        docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filters=filters,
            ef_search=ef_search
        )

        logger.info(f"Document retrieval completed | Count = {len(docs)}")
        logger.debug(f"Retrieved docs: {docs}")

        return docs