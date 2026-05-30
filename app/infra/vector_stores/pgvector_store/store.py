from app.infra.vector_stores.base import BaseVectorStore
from app.infra.vector_stores.pgvector_store.config import RetrievedDocument

import logging
logger = logging.getLogger("app.infra.vector_stores.pg_vector_store.store")
logger.info("Loading file...")

import psycopg2
from psycopg2.extras import execute_values, Json

class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector (cosine similarity + HNSW index)
    - embeddings stored in Postgres
    - cosine similarity as default metric
    - HNSW index for ANN retrieval
    """

    def __init__(self, connection_string: str, embedding_dim: int, m: int, ef_construction: int):
        self.conn = psycopg2.connect(connection_string)
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            with self.conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # -------------------------
                # Documents table
                # -------------------------
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGSERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR({self.embedding_dim}),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                # -------------------------
                # Vector index (HNSW)
                # -------------------------
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
                    ON documents
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (
                        m = {self.m},
                        ef_construction = {self.ef_construction}
                    );
                """)

    def reset_schema(self):
        with self.conn:
            with self.conn.cursor() as cur:
                # 1. Drop everything
                cur.execute("DROP TABLE IF EXISTS documents;")

        # 2. Recreate using existing logic
        self._ensure_schema()

    def add_documents(
        self,
        chunk_ids: list[int],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None
    ) -> None:

        metadatas = metadatas or [{} for _ in texts]

        rows = [
            (
                id,
                text,
                embedding,
                Json(metadata),
            )
            for id, text, embedding, metadata in zip(chunk_ids, texts, embeddings, metadatas)
        ]

        with self.conn:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO documents (id, content, embedding, metadata)
                    VALUES %s
                    """,
                    rows,
                )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        ef_search: int,
        filters: dict | None = None,
    ) -> list[RetrievedDocument]:
        
        logger.info("Similarity search started")

        sql = """
            SELECT id, content, metadata,
                   embedding <=> %s::vector AS distance
            FROM documents
        """

        params = [query_embedding]

        # Optional metadata filtering (JSONB)
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"metadata->>%s = %s")
                params.extend([key, value])

            sql += " WHERE " + " AND ".join(conditions)

        sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params.extend([query_embedding, top_k])

        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                cur.execute(sql, params)
                rows = cur.fetchall()
        
        logger.info("Similarity search completed")

        return [
            RetrievedDocument(**{
                "id": r[0],
                "content": r[1],
                "metadata": r[2],
                "distance": r[3],
            })
            for r in rows
        ]

    def delete(self, ids: int | list[int]) -> None:
        if isinstance(ids, int):
            ids = [ids]

        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id = ANY(%s)",
                    (ids,),
                )

    def get_max_id(self):
        with self.conn.cursor() as cur:
            cur.execute(f'''
                SELECT COALESCE(MAX(id), 0)
                FROM documents;
            ''')
            result = cur.fetchone()[0]
        return result
        
    def close_conn(self):
        if self.conn:
            self.conn.close()