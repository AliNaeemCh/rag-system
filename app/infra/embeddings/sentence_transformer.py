from sentence_transformers import SentenceTransformer
from app.infra.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from pathlib import Path
import logging

logger = logging.getLogger("app.infra.embeddings.sentence_transformer")

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        model_path: Path,
        device: str | None = None
    ):
        """
        model_name: any SentenceTransformer-compatible model
        device: 'cpu', 'cuda', or None (auto)
        """
        self.model = SentenceTransformer(str(model_path), device=device)

    def embed_query(
        self,
        query: str,
        retrieval_instruction: str = "",
        normalize: bool = True
    ):
        """
        Embed a single query string.
        """
        logger.info(f"Query embedding started")
        
        emb = self.model.encode(
            retrieval_instruction + query,
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
    ):
        """
        Embed one or multiple documents.
        """
        batch_size = batch_size or settings.CHUNKS_EMBEDDING_BATCH_SIZE

        if isinstance(documents, str):
            documents = [documents]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size
        )

        return embeddings.tolist()