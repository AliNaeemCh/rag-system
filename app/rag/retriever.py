import logging
from app.infra.vectorstore import FAISSVectorStore

logger = logging.getLogger("rag.retriever")


class Retriever:
    def __init__(self, vectorstore: FAISSVectorStore, top_k: int):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str):
        """
        1. Embed query
        2. Search vector DB
        3. Return top-k docs
        """

        logger.debug(f"Retrieval started | Query = {query}")

        # 1. Get embedding
        query_embedding = self.vectorstore.embed_query(query)

        logger.debug("Query embedded", extra={"embedding": query_embedding})

        # 2. Search vector store
        docs = self.vectorstore.search(
            embedding=query_embedding,
            top_k=self.top_k
        )

        logger.info(f"Retrieved documents | count={len(docs)}")

        return docs