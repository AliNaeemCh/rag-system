from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.retry_policies import huggingface_retry
from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.embeddings.hugging_face")
logger.info("Loading file...")

from huggingface_hub import InferenceClient

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, client: InferenceClient, model: str):
        self.client = client
        self.model = model

    @huggingface_retry(logger)
    def embed_query(self, query: str, retrieval_instruction: str | None = None, normalize: bool = True):
        logger.info(f"Query embedding started")

        retrieval_instruction = retrieval_instruction or settings.RETRIEVAL_INSTRUCTION

        embedding = self.client.feature_extraction(
            retrieval_instruction + query,
            model=self.model
        )

        if normalize:
            embedding = self._normalize_embeddings(embedding)

        logger.info(f"Query embedding completed")

        return embedding
    
    @huggingface_retry(logger)
    def embed_documents(self, documents: list[str] | str, batch_size: int | None = None, normalize: bool = True) -> list[float] | list[list[float]]:

        embeddings = self.client.feature_extraction(
            documents,
            model=self.model
        )

        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        logger.debug(
            f"Documents embedded | count={len(documents)}"
        )

        return embeddings