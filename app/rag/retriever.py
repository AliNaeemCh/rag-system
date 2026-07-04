from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.infra.retrieval.base import BaseRetrievalStore
from app.models import RetrievedDocument

import logging
logger = logging.getLogger("rag.retriever")
logger.info("Loading file...")

class Retriever:
    def __init__(self, embedding_provider: BaseEmbeddingProvider, retrieval_store: BaseRetrievalStore, dense_top_k: int, sparse_top_k: int):
        self.embedding_provider = embedding_provider
        self.retrieval_store = retrieval_store
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k

    def retrieve(self, query: str,
                 filters: dict | None = None,
                 normalize_query: bool = True,
                 ef_search: int | None = None,
                 ) -> dict[str, RetrievedDocument]:
        """
        1. Embed query
        2. Vector search
        3. Keyword search
        4. Return top-k docs
        """
        ef_search = ef_search or settings.HNSW_EF_SEARCH

        logger.info(f"Retrieval started")

        # 1. Get embedding
        query_embedding = self.embedding_provider.embed_query(query, normalize=normalize_query)

        # 2. Similarity search
        dense_docs = self.retrieval_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.dense_top_k,
            ef_search=ef_search,
            filters=filters
        )

        # 3. Keyword search
        sparse_docs = self.retrieval_store.keyword_search(
            query=query,
            top_k=self.sparse_top_k,
            filters=filters
        )

        logger.info(f"Retrieval completed")

        # return docs
        return {
            "dense_docs": dense_docs,
            "sparse_docs": sparse_docs
        }