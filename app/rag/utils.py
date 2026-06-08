from collections import defaultdict
from app.models import RetrievedDocument, RetrievalType

def reciprocal_rank_fusion(
    dense_docs: list[RetrievedDocument],
    sparse_docs: list[RetrievedDocument],
    top_k: int,
    k: int = 60,
) -> list[RetrievedDocument]:
    """
    Reciprocal Rank Fusion (RRF) for dense + sparse retrieval results.
    """

    def rrf_score(rank: int) -> float:
        return 1.0 / (k + rank)

    def accumulate(docs: list[RetrievedDocument]):
        """
        Adds RRF scores using document identity via object identity fallback.
        """
        for rank, doc in enumerate(docs):
            doc_id = doc.id
            scores[doc_id] += rrf_score(rank)
            if doc_id in store:
                if doc.scores.dense_retrieval_score:
                    store[doc_id].scores.dense_retrieval_score = doc.scores.dense_retrieval_score
                else:
                    store[doc_id].scores.sparse_retrieval_score = doc.scores.sparse_retrieval_score
                store[doc_id].retrieval_type = RetrievalType.DENSE_SPARSE
            else:
                store[doc_id] = doc

    scores = defaultdict(float)
    store: dict[int, RetrievedDocument] = {}

    accumulate(dense_docs)
    accumulate(sparse_docs)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for doc_key, score in ranked[:top_k]:
        doc = store[doc_key]
        doc.scores.rrf_score = score
        results.append(doc)

    return results