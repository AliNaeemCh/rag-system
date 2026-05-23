import logging
from huggingface_hub import InferenceClient
from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.retry_policies import huggingface_retry

logger = logging.getLogger("app.infra.embeddings.hugging_face")

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, client: InferenceClient, model: str):
        self.client = client
        self.model = model

    @huggingface_retry(logger)
    def embed_query(self, query: str, retrieval_instruction: str = '', normalize: bool = True):

        embedding = self.client.feature_extraction(
            retrieval_instruction + query,
            model=self.model
        )

        if normalize:
            embedding = self._normalize_embeddings(embedding)

        logger.debug("Query embedded")

        return embedding
    
    @huggingface_retry(logger)
    def embed_documents(self, documents: list[str] | str, normalize: bool = True) -> list[float] | list[list[float]]:

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