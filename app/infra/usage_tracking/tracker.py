from app.infra.usage_tracking.buckets import Bucket
from app.infra.usage_tracking.buckets import get_bucket, BUCKET_TO_TOKEN_LIMIT
from app.infra.db.session import get_connection
from app.infra.db.pool import get_usage_tracker_db_pool
from app.infra.db.retry import db_retry
from app.core.config import settings

import logging
logger = logging.getLogger("app.infra.usage_tracking.tracker")
logger.info("Loading file...")

from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

class UsageTracker:
    def __init__(self, db_pool: SimpleConnectionPool):
        self.db_pool = db_pool
        self._ensure_schema()

    @db_retry(retries=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS)
    def _ensure_schema(self):
        with get_connection(self.db_pool) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS token_usage_daily (
                        day DATE PRIMARY KEY,
                        small_bucket_tokens BIGINT DEFAULT 0,
                        large_bucket_tokens BIGINT DEFAULT 0
                    );
                """)
    
    def get_current_bucket_usage(self, model_name: str) -> int:
        bucket = get_bucket(model_name=model_name)
        column = self._get_bucket_column(bucket=bucket)
        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT {column}
                        FROM token_usage_daily
                        WHERE day = (timezone('UTC', now()))::date
                    """)
                    return cur.fetchone()
        
        row = db_retry(retries=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS)(_db_op)()

        if not row or row[column] is None:
            return 0

        return int(row[column])

    @db_retry(retries=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS)
    def _get_current_usage(self) -> dict:
        with get_connection(self.db_pool) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT small_bucket_tokens, large_bucket_tokens
                    FROM token_usage_daily
                    WHERE day = (timezone('UTC', now()))::date
                """)
                row = cur.fetchone()

        return row or {
            "small_bucket_tokens": 0,
            "large_bucket_tokens": 0
        }

    def increment(self, model_name: str, tokens: int):
        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO token_usage_daily(day, {column})
                        VALUES ((timezone('UTC', now()))::date, %s)
                        ON CONFLICT (day)
                        DO UPDATE SET
                            {column} = token_usage_daily.{column} + EXCLUDED.{column}
                    """, (tokens,))
        try:
            bucket = get_bucket(model_name=model_name)
            column = self._get_bucket_column(bucket=bucket)
            db_retry(retries=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS)(_db_op)()
        except Exception as e:
            logger.exception(f"Usage increment failed.")
    
    def usage_exceeded(self, model_names: list[str], safety_margin_tokens: int = 5000) -> bool:
        usage = self._get_current_usage()

        seen_buckets = set()

        for model_name in model_names:
            bucket = get_bucket(model_name)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)

            column = self._get_bucket_column(bucket)
            limit = BUCKET_TO_TOKEN_LIMIT[bucket]

            if usage[column] > limit - safety_margin_tokens:
                return True

        return False
    
    def delete_older_token_usage(self):
        def _db_op():
            with get_connection(self.db_pool) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM token_usage_daily
                        WHERE day < (timezone('UTC', now()))::date
                    """)
        try:
            db_retry(retries=settings.USAGE_TRACKER_DB_POOL_MAX_CONNS)(_db_op)()
        except Exception as e:
            logger.exception(f"Older token usage deletion failed.")

    def _get_bucket_column(self, bucket: Bucket):
        return (
            "small_bucket_tokens"
            if bucket == Bucket.SMALL
            else "large_bucket_tokens"
        )

usage_tracker: UsageTracker | None = None
if settings.USAGE_TRACKER_DB_URL:
    usage_tracker_db_pool = get_usage_tracker_db_pool()
    usage_tracker = UsageTracker(db_pool=usage_tracker_db_pool)