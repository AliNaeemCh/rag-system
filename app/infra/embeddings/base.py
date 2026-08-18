import logging
logger = logging.getLogger("app.infra.embeddings.base")
logger.info("Loading file...")

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List

class BaseEmbeddingProvider(ABC):

    @abstractmethod
    async def embed_query(self, query: str, normalize: bool = True):
        pass

    @abstractmethod
    async def embed_documents(self, documents: list[str] | str, batch_size: int | None = None, normalize: bool = True):
        pass

    def _normalize_embeddings(
        self,
        embeddings: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:

        try:
            # -------------------------
            # Basic validation
            # -------------------------
            if embeddings is None:
                raise ValueError("Input embeddings is None")

            arr = np.array(embeddings, dtype=np.float32)

            if arr.size == 0:
                raise ValueError("Empty embedding input")

            # Handle NaN / Inf early
            if not np.isfinite(arr).all():
                raise ValueError("Embedding contains NaN or Inf values")

            # -------------------------
            # Single vector: (d,)
            # -------------------------
            if arr.ndim == 1:

                if arr.shape[0] == 0:
                    raise ValueError("Empty vector provided")

                norm = np.linalg.norm(arr)

                if norm == 0:
                    return arr.tolist()  # safer than returning raw input type

                return (arr / norm).tolist()

            # -------------------------
            # Batch vectors: (n, d)
            # -------------------------
            elif arr.ndim == 2:

                if arr.shape[1] == 0:
                    raise ValueError("Embedding dimension is zero")

                norms = np.linalg.norm(arr, axis=1, keepdims=True)

                # detect bad rows (all-zero vectors)
                zero_mask = norms == 0
                if np.any(zero_mask):
                    norms[zero_mask] = 1.0

                return (arr / norms).tolist()

            else:
                raise ValueError(f"Invalid embedding shape: {arr.shape}")

        except Exception as e:
            # Optional: replace with logging instead of raising
            raise ValueError(f"Normalization failed: {str(e)}") from e