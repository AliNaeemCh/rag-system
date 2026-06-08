from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.embeddings.sentence_transformer")
logger.info("Loading file...")

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, model):
        self.model = model

    def embed_query(
        self,
        query: str,
        normalize: bool = True
    ) -> list[float]:
        """
        Embed a single query string.
        """
        logger.info(f"Query embedding started")
        
        emb = self.model.encode_query(
            query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        logger.info(f"Query embedding completed")

        return emb.tolist()

    def embed_documents(
        self,
        documents: list[str] | str,
        batch_size: int | None = None,
        normalize: bool = True
    ) -> list[list[float]]:
        """
        Embed one or multiple documents.
        """
        batch_size = batch_size or settings.CHUNKS_EMBEDDING_BATCH_SIZE

        if isinstance(documents, str):
            documents = [documents]

        embeddings = self.model.encode_document(
            documents,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size
        )

        return embeddings.tolist()