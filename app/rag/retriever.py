import logging
from app.infra.vectorstores.pgvector_store import PgVectorStore
from app.infra.embeddings.hugging_face import HuggingFaceEmbeddingProvider

logger = logging.getLogger("rag.retriever")


class Retriever:
    def __init__(self, embedding_model: HuggingFaceEmbeddingProvider, vector_store: PgVectorStore, top_k: int):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, filters: dict | None = None):
        """
        1. Embed query
        2. Search vector DB
        3. Return top-k docs
        """

        logger.debug(f"Retrieval started | Query = {query}")

        # 1. Get embedding
        query_embedding = self.embedding_model.embed_query(query)

        logger.debug(f"Query embedded | Embedding = {query_embedding}")

        # 2. Search vector store
        docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filters=filters
        )

        logger.info(f"Document retrieval completed | Count = {len(docs)}")
        logger.debug(f"Retrieved docs: {docs}")

        return docs