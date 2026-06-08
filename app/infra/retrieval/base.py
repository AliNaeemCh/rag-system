from app.models import RetrievedDocument

import logging
logger = logging.getLogger("app.infra.retrieval.base")
logger.info("Loading file...")

from abc import ABC, abstractmethod

class BaseRetrievalStore(ABC):
    """
    Defines the contract for any retrieval backend
    (OpenSearch, Qdrant, FAISS, etc.)
    """

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        **kwargs
    ) -> list[RetrievedDocument]:
        """
        Dense/vector similarity search (semantic retrieval)
        """
        pass

class BaseDocumentStore(ABC):
    """
    Defines the contract for indexing / ingestion backends
    """

    @abstractmethod
    def add_document(
        self,
        chunk_id: str,
        content: str,
        embedding: list[float],
        metadata: dict | None = None
    ) -> None:
        """
        Index a single chunk into the backend
        """
        pass

    @abstractmethod
    def add_documents_bulk(
        self,
        chunks: list[dict]
    ) -> None:
        """
        Bulk ingestion of multiple chunks
        Expected keys per dict:
            - chunk_id
            - content
            - embedding
            - metadata (optional)
        """
        pass

    @abstractmethod
    def reset_store(self) -> None:
        """
        Reset the store for re-ingestion pipelines
        """
        pass

    @abstractmethod
    def get_max_chunk_id(self) -> int:
        """
        Useful for resuming ingestion safely
        """
        pass