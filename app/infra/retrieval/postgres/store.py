from app.infra.retrieval.base import BaseDocumentStore, BaseRetrievalStore
from app.models import RetrievedDocument, RetrievalType, ScoreBreakdown
from app.infra.db.session import get_connection
from app.infra.db.retry import db_retry
from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.vector_stores.pg_vector_store.store")
logger.info("Loading file...")

from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

class PgStore(BaseRetrievalStore, BaseDocumentStore):
    """
    PostgreSQL + pgvector (cosine similarity + HNSW index)
    - embeddings stored in Postgres
    - cosine similarity as default metric
    - HNSW index for ANN retrieval
    """

    def __init__(self, db_pool: AsyncConnectionPool, embedding_dim: int, m: int, ef_construction: int):
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self.db_pool = db_pool

    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    async def ensure_schema(self):
        async with get_connection(self.db_pool) as conn:
            async with conn.cursor() as cur:
                # Enable pgvector + pg_search extensions
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # -------------------------
                # Documents table
                # -------------------------
                await cur.execute(f"""
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
                await cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
                    ON documents
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (
                        m = {self.m},
                        ef_construction = {self.ef_construction}
                    );
                """)
    
    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    async def reset_store(self):
        async with get_connection(self.db_pool) as conn:
            async with conn.cursor() as cur:
                # 1. Drop everything
                await cur.execute("DROP TABLE IF EXISTS documents;")

        # 2. Recreate using existing logic
        await self.ensure_schema()

    async def add_document(
        self,
        chunk_id: int,
        content: str,
        embedding: list[float],
        metadata: dict | None = None
    ) -> None:

        if metadata is None:
            metadata = {}

        async def _db_op():
            async with get_connection(self.db_pool) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO documents (id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (chunk_id, content, embedding, Json(metadata)),
                    )

        await db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()

    async def add_documents_bulk(self, chunks: list[dict]) -> None:

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


        async def _db_op():
            async with get_connection(self.db_pool) as conn:
                async with conn.cursor() as cur:
                    await cur.execute_many(
                        """
                        INSERT INTO documents (id, content, embedding, metadata)
                        VALUES %s
                        """,
                        rows,
                    )

        await db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()

    async def similarity_search(
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

        async def _db_op():
            async with get_connection(self.db_pool) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                    await cur.execute(sql, params)
                    return (await cur.fetchall())

        rows = await db_retry(settings.RAG_DB_POOL_MAX_CONNS)(_db_op)()
        
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

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedDocument]:
        
        # Keyword search not implemented

        return []

    @db_retry(retries=settings.RAG_DB_POOL_MAX_CONNS)
    async def get_max_chunk_id(self):
        async with get_connection(self.db_pool) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COALESCE(MAX(id), 0) FROM documents;")
                chunk_id = await cur.fetchone()[0]
                return chunk_id