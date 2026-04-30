from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):

    @abstractmethod
    def embed_query(self, query: str):
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str] | str):
        pass