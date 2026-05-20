import uuid
import psycopg2
from psycopg2.extras import execute_values
from app.infra.vector_stores.base import VectorStoreBase
from app.core.config import settings


class PgVectorStore(VectorStoreBase):
    """
    PostgreSQL + pgvector (cosine similarity + HNSW index)

    Production-grade assumptions:
    - embeddings stored in Postgres
    - cosine similarity as default metric
    - HNSW index for ANN retrieval
    """

    def __init__(self, connection_string: str, embedding_dim: int, m: int, ef_construction: int):
        self.conn = psycopg2.connect(connection_string)
        self.conn.autocommit = True
        self._ensure_schema(embedding_dim, m, ef_construction)

    def _ensure_schema(self, embedding_dim: int, m: int, ef_construction: int):
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Main documents table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR({embedding_dim}),
                    metadata JSONB
                );
            """)

            # Production-grade HNSW index for cosine similarity
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
                ON documents
                USING hnsw (embedding vector_cosine_ops)
                WITH (
                    m = {m},
                    ef_construction = {ef_construction}
                );
            """)

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:

        metadatas = metadatas or [{} for _ in texts]

        rows = [
            (
                str(uuid.uuid4()),
                text,
                embedding,
                metadata,
            )
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]

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
        filters: dict | None = None,
    ) -> list[dict]:

        sql = """
            SELECT id, content, metadata,
                   embedding <=> %s AS distance
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
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        params.extend([query_embedding, top_k])

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "metadata": r[2],
                "distance": r[3],
            }
            for r in rows
        ]

    def delete(self, ids: list[str]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE id = ANY(%s)",
                (ids,),
            )
    
    def close(self):
        if self.conn:
            self.conn.close()