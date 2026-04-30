import faiss
import numpy as np
import logging
from typing import List
import pickle
import os

logger = logging.getLogger("app.infra.vectorstore")


class Document:
    """
    Simple document container
    """
    def __init__(self, text: str, metadata: dict = None):
        self.text = text
        self.metadata = metadata or {}


class FAISSVectorStore:
    def __init__(self, dim: int):
        """
        dim → embedding dimension
        """
        self.dim = dim

        # FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dim)

        # Store documents alongside vectors
        self.documents: List[Document] = []

    # ---------------- ADD DATA ----------------
    def add_documents(self, embeddings: np.ndarray, docs: List[Document]):
        """
        Add documents to FAISS index
        """
        texts = [doc.text for doc in docs]
        embeddings = self.embed_documents(texts)

        self.index.add(embeddings)
        self.documents.extend(docs)

        logger.info(f"Documents added | count={len(docs)}")

    # ---------------- SEARCH ----------------
    def search(self, embedding: np.ndarray, top_k: int):
        """
        Returns top-k most similar documents
        """
        if len(self.documents) == 0:
            logger.warning("Search called on empty index")
            return []

        # FAISS expects shape (1, dim)
        embedding = np.expand_dims(embedding, axis=0)

        distances, indices = self.index.search(embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        logger.debug(f"Search completed | top_k={top_k}")

        return results
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # Save documents
        with open(f"{path}/docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Vectorstore saved at {path}")

    @classmethod
    def load(cls, path: str, embedding_model):
        # Load FAISS index
        index = faiss.read_index(f"{path}/index.faiss")

        # Load documents
        with open(f"{path}/docs.pkl", "rb") as f:
            documents = pickle.load(f)

        obj = cls(embedding_model, dim=index.d)
        obj.index = index
        obj.documents = documents

        logger.info(f"Vectorstore loaded from {path}")

        return obj