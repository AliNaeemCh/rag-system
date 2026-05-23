from abc import ABC, abstractmethod
import numpy as np

class BaseEmbeddingProvider(ABC):

    @abstractmethod
    def embed_query(self, query: str, retrieval_instruction: str = '', normalize: bool = True):
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str] | str, normalize: bool = True):
        pass

    def _normalize_embeddings(self, embeddings: list[float] | list[list[float]]) -> list[float] | list[list[float]]:

        arr = np.array(embeddings, dtype=np.float32)

        # -------------------------
        # Single vector: (d,)
        # -------------------------
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            if norm == 0:
                return embeddings
            return (arr / norm).tolist()

        # -------------------------
        # Batch vectors: (n, d)
        # -------------------------
        elif arr.ndim == 2:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return (arr / norms).tolist()

        else:
            raise ValueError("Input must be list[float] or list[list[float]]")