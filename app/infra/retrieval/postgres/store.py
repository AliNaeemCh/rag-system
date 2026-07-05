from app.infra.retrieval.base import BaseDocumentStore, BaseRetrievalStore
from app.models import RetrievedDocument, RetrievalType, ScoreBreakdown
from app.infra.db.session import get_connection
from app.infra.db.retry import db_retry
from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.vector_stores.pg_vector_store.store")
logger.info("Loading file...")

from psycopg2.extras import execute_values, Json
from psycopg2.pool import SimpleConnectionPool

class PgStore(BaseRetrievalStore, BaseDocumentStore):
    """
    PostgreSQL + pgvector (cosine similarity + HNSW index) +  Keyword search
    - embeddings stored in Postgres
    - cosine similarity as default metric
    - HNSW index for ANN retrieval
    """

    def __init__(self, db_pool: SimpleConnectionPool, embedding_dim: int, m: int, ef_construction: int, kw_config: str = 'english'):
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self.db_pool = db_pool
        self.kw_config = kw_config
        self._ensure_schema()

    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    def _ensure_schema(self):
        with get_connection(self.db_pool) as conn:
            with conn.cursor() as cur:
                # Enable pgvector + pg_search extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # -------------------------
                # Documents table
                # -------------------------
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGSERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR({self.embedding_dim}),
                        content_tsv tsvector GENERATED ALWAYS AS (
                            to_tsvector('{self.kw_config}', content)
                        ) STORED,
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

                # -------------------------
                # Keyword search index
                # -------------------------
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_content_gin
                    ON documents
                    USING GIN (content_tsv);
                """)
    
    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    def reset_store(self):
        with get_connection(self.db_pool) as conn:
            with conn.cursor() as cur:
                # 1. Drop everything
                cur.execute("DROP TABLE IF EXISTS documents;")

        # 2. Recreate using existing logic
        self._ensure_schema(self.kw_config)

    def add_document(
        self,
        chunk_id: int,
        content: str,
        embedding: list[float],
        metadata: dict | None = None
    ) -> None:

        if metadata is None:
            metadata = {}

        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO documents (id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (chunk_id, content, embedding, Json(metadata)),
                    )

        db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()

    def add_documents_bulk(self, chunks: list[dict]) -> None:

        rows = []

        for chunk in chunks:
            rows.append(
                (
                    chunk['chunk_id'],
                    chunk['content'],
                    chunk['embedding'],
                    Json(chunk.get("metadata", {}))
                )
            )


        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO documents (id, content, embedding, metadata)
                        VALUES %s
                        """,
                        rows,
                    )

        db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()

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

        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                    cur.execute(sql, params)
                    return cur.fetchall()

        rows = db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()
        
        logger.info("Similarity search completed")

        return [
            RetrievedDocument(**{
                "id": r[0],
                "content": r[1],
                "metadata": r[2],
                "retrieval_type": RetrievalType.DENSE,
                "scores": ScoreBreakdown(dense_retrieval_score=1 - r[3])
            })
            for r in rows
        ]
    
    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    def delete(self, ids: int | list[int]) -> None:
        if isinstance(ids, int):
            ids = [ids]

        with get_connection(self.db_pool) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id = ANY(%s)",
                    (ids,),
                )

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedDocument]:

        def build_tsquery(query: str) -> str:
            return " | ".join(query.split())

        logger.info("Keyword search started")

        clean_query = build_tsquery(query)

        sql = """
        WITH q AS (
            SELECT to_tsquery(%s, %s) AS query
        )
        SELECT
            id,
            content,
            metadata,
            ts_rank_cd(content_tsv, q.query) AS score
        FROM documents, q
        WHERE content_tsv @@ q.query
        """

        params = [self.kw_config, clean_query]

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append("metadata->>%s = %s")
                params.extend([key, value])

            sql += " AND " + " AND ".join(conditions)

        sql += """
            ORDER BY score DESC
            LIMIT %s
        """

        params.append(top_k)

        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()

        rows = db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()

        logger.info("Keyword search completed")

        return [
            RetrievedDocument(
                id=r[0],
                content=r[1],
                metadata=r[2],
                retrieval_type=RetrievalType.SPARSE,
                scores=ScoreBreakdown(sparse_retrieval_score=r[3]),
            )
            for r in rows
        ]

    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    def get_max_chunk_id(self):
        with get_connection(self.db_pool) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(id), 0) FROM documents;")
                return cur.fetchone()[0]