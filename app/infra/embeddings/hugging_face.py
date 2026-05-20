import logging
from huggingface_hub import InferenceClient
from app.infra.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger("app.infra.embeddings.hugging_face")


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, client: InferenceClient, model: str):
        self.client = client
        self.model = model

    def embed_query(self, query: str, retrieval_instruction: str = ''):

        embedding = self.client.feature_extraction(
            retrieval_instruction + query,
            model=self.model
        )

        logger.debug("Query embedded")

        return embedding

    def embed_documents(self, documents: list[str] | str) -> list[float] | list[list[float]]:

        embeddings = self.client.feature_extraction(
            documents,
            model=self.model
        )

        logger.debug(
            f"Documents embedded | count={len(documents)}"
        )

        return embeddings