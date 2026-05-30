from app.infra.vector_stores.pgvector_store.config import RetrievedDocument

import logging
logger = logging.getLogger("rag.reranker")
logger.info("Loading file...")

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, reranker_model: CrossEncoder, top_k: int):
        self.top_k = top_k
        self.reranker_model = reranker_model

    def rerank(self, query: str, docs: list[RetrievedDocument] | list[str]) -> list[RetrievedDocument]:
        """
        1. Create query-document pairs
        2. Score documents using CrossEncoder
        3. Sort by relevance
        4. Return top-k docs
        """

        logger.info(f"Reranking started")

        if not docs:
            logger.warning("No documents received for reranking")
            return []

        # 1. Create query-doc pairs
        pairs = [
            (query, doc.content if isinstance(doc, RetrievedDocument) else doc)
            for doc in docs
        ]

        logger.debug(f"Prepared {len(pairs)} query-document pairs")

        # 2. Predict relevance scores
        scores = self.reranker_model.predict(pairs)

        logger.debug(f"Reranker scores = {scores}")

        # 3. Sorting
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        docs_sorted = [docs[i] for i in sorted_indices]

        # 4. Preparing top-k reranked docs
        reranked_docs = [doc for doc in docs_sorted[: self.top_k]]

        logger.info(f"Reranking completed")

        return reranked_docs