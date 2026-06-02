from opensearchpy import OpenSearch, helpers
import copy

class OpenSearchStore:
    """
    OpenSearch used ONLY as:
    - BM25 inverted index
    - metadata filter engine
    - chunk_id retrieval pointer system

    Raw text is fetched from an external store.
    """

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
    ):
        """
        external_fetch_fn:
            function that takes chunk_id and returns raw text
        """
        self.client = client
        self.index = index_name

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
                    "number_of_replicas": 0
                }
            },
            'mappings': {
                "_source": {
                    "enabled": False    # Doesn't store the raw doc data
                },
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "dynamic": True}
                }
            }
        }

        self.client.indices.create(index=self.index, body=mapping)

    # -------------------------
    # ADD SINGLE DOCUMENT
    # -------------------------
    def add_chunk(self, chunk_id: str, text: str, metadata: dict | None = None):

        if metadata is None:
            metadata = {}

        doc = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata
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
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {})
                }
            })

        helpers.bulk(self.client, actions)

    # -------------------------
    # BM25 SEARCH + METADATA FILTER
    # -------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ):
        """
        Returns:
        - chunk_id
        - score
        - raw text (fetched externally)
        - metadata
        """

        must_clause = {
            "match": {
                "text": query
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
            "size": k,
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
            chunk_id = hit["_id"] or hit["_source"].get("chunk_id")

            results.append({
                "chunk_id": chunk_id,
                "score": hit["_score"],
                "metadata": hit["_source"]["metadata"] if "_source" in hit and hit["_source"] else {}
            })

        return results

    # -------------------------
    # GET ONLY IDS
    # -------------------------
    def search_ids(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ):
        """
        Lightweight version: no external fetch
        Useful for reranking pipelines
        """

        must_clause = {
            "match": {
                "text": query
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
            "size": k,
            "_source": False,  # extra safety
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

        return [
            {
                "chunk_id": hit["_id"],
                "score": hit["_score"]
            }
            for hit in response["hits"]["hits"]
        ]

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