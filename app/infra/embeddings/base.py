from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):

    @abstractmethod
    def embed_query(self, query: str, retrieval_instruction: str = ''):
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str] | str):
        pass