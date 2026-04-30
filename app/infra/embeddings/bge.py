import logging
from huggingface_hub import InferenceClient
from app.infra.embeddings.base import BaseEmbeddingModel

logger = logging.getLogger("app.infra.embeddings.bge")


class BGEEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, client: InferenceClient, model: str):
        self.client = client
        self.model = model

    def embed_query(self, query: str):

        instruction = "Represent this sentence for searching relevant passages:"    # Reference: https://huggingface.co/BAAI/bge-large-en-v1.5#model-list

        embedding = self.client.feature_extraction(
            instruction + query,
            model=self.model
        )

        logger.debug("Query embedded")

        return embedding

    def embed_documents(self, documents: list[str] | str):

        embeddings = self.client.feature_extraction(
            documents,
            model=self.model
        )

        logger.debug(
            f"Documents embedded | count={len(documents)}"
        )

        return embeddings