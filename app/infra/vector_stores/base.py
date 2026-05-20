from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """
    Abstract contract for vector stores.
    Keeps retrieval layer decoupled from implementation.
    """

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass