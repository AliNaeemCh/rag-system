from app.models import RetrievedDocument, ScoreBreakdown, RetrievalType

import logging
logger = logging.getLogger("app.infra.search_engines.opensearch.store")
logger.info("Loading file...")

from opensearchpy import OpenSearch, helpers

class OpenSearchStore:

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        embedding_dim: int,
        m: int,
        ef_construction: int
    ):
        """
        external_fetch_fn:
            function that takes chunk_id and returns raw text
        """
        self.client = client
        self.index = index_name
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction

        self._ensure_index()

    # -------------------------
    # INDEX SETUP
    # -------------------------
    def _ensure_index(self):
        if self.client.indices.exists(index=self.index):
            return

        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,

                    # Enable k-NN search at index level
                    "knn": True
                }
            },
            "mappings": {
                "_source": {
                    "enabled": True
                },
                "properties": {
                    "chunk_id": {
                        "type": "integer"
                    },
                    "content": {
                        "type": "text"
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,

                        # HNSW configuration
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "m": self.m,
                                "ef_construction": self.ef_construction
                            }
                        }
                    }
                }
            }
        }

        self.client.indices.create(index=self.index, body=mapping)

    # -------------------------
    # ADD SINGLE DOCUMENT
    # -------------------------
    def add_chunk(self, chunk_id: str, content: str, embedding: list[float], metadata: dict | None = None):

        if metadata is None:
            metadata = {}

        doc = {
            "chunk_id": chunk_id,
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }

        self.client.index(
            index=self.index,
            body=doc,
            id=chunk_id
        )

    # -------------------------
    # BULK INGESTION
    # -------------------------
    def add_chunks_bulk(self, chunks: list[dict]):

        actions = []

        for chunk in chunks:
            actions.append({
                "_op_type": "index",
                "_index": self.index,
                "_id": chunk["chunk_id"],
                "_source": {
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "embedding": chunk['embedding'],
                    "metadata": chunk.get("metadata", {})
                }
            })

        helpers.bulk(self.client, actions)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        ef_search: int,
        filters: dict | None = None
    ) -> list[RetrievedDocument]:
        """
        Returns:
        - chunk_id
        - score
        - content
        - metadata
        """

        filter_clauses = []

        if filters:
            for key, value in filters.items():
                filter_clauses.append({
                    "term": {
                        f"metadata.{key}": value
                    }
                })

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                        "method_parameters": {
                            "ef_search": ef_search
                        },
                        "filter": {
                            "bool": {
                                "filter": filter_clauses
                            }
                        }
                    }
                }
            }
        }

        response = self.client.search(
            index=self.index,
            body=body
        )

        results = []

        for hit in response["hits"]["hits"]:
            chunk_id = hit["_source"]["chunk_id"]

            results.append(
                RetrievedDocument(
                    id=chunk_id,
                    content=hit["_source"]["content"],
                    metadata=hit["_source"]["metadata"],
                    retrieval_type=RetrievalType.DENSE,
                    scores=ScoreBreakdown(retrieval_score=hit["_score"])
                )
            )

        return results

    # -------------------------
    # BM25 SEARCH + METADATA FILTER
    # -------------------------
    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None
    ) -> list[RetrievedDocument]:
        """
        Returns:
        - chunk_id
        - score
        - content
        - metadata
        """

        must_clause = {
            "match": {
                "content": query
            }
        }

        filter_clauses = []

        if filters:
            for key, value in filters.items():
                filter_clauses.append({
                    "term": {
                        f"metadata.{key}": value
                    }
                })

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clause,
                    "filter": filter_clauses
                }
            }
        }

        response = self.client.search(
            index=self.index,
            body=body
        )

        results = []

        for hit in response["hits"]["hits"]:
            chunk_id = hit["_source"]["chunk_id"]

            results.append(
                RetrievedDocument(
                    id=chunk_id,
                    content=hit["_source"]["content"],
                    metadata=hit["_source"]["metadata"],
                    retrieval_type=RetrievalType.SPARSE,
                    scores=ScoreBreakdown(retrieval_score=hit["_score"])
                )
            )

        return results

    def get_max_chunk_id(self) -> int:
        """
        Returns the maximum chunk_id present in the index.
        Useful for resuming ingestion.
        """

        body = {
            "size": 0,
            "aggs": {
                "max_chunk_id": {
                    "max": {
                        "field": "chunk_id"
                    }
                }
            }
        }

        response = self.client.search(
            index=self.index,
            body=body
        )

        value = response.get("aggregations", {}) \
                        .get("max_chunk_id", {}) \
                        .get("value")

        if value is None:
            return 0

        return int(value)

    def reset_index(self) -> None:
        """
        Deletes and recreates the index from scratch.
        """

        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)

        # recreate index with same mapping
        self._ensure_index()