from app.models import RetrievedDocument

import requests
import logging

logger = logging.getLogger("rag.reranker")
logger.info("Loading file...")


class Reranker:

    def __init__(
        self,
        api_url: str,
        top_k: int,
    ):
        self.api_url = api_url
        self.top_k = top_k


    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
    ) -> list:

        logger.info("Calling reranker API")

        payload = {
            "query": query,
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                }
                for doc in docs
            ],
            "top_k": self.top_k,
        }

        response = requests.post(
            self.api_url,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        reranker_scores = result["scores"]

        # Map scores back to documents
        score_map = {
            item["id"]: item["score"]
            for item in reranker_scores
        }

        for doc in docs:
            if doc.id in score_map:
                doc.scores.reranker_score = score_map[doc.id]

        # Sort documents by reranker score
        docs_sorted = sorted(
            docs,
            key=lambda d: d.scores.reranker_score,
            reverse=True,
        )

        logger.info(
            "Reranking completed"
        )
        
        # Return top-k
        return docs_sorted[:self.top_k]