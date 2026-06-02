import logging
logger = logging.getLogger("app.infra.db.session")
logger.info("Loading file...")

from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import connection
from typing import Iterator

@contextmanager
def get_connection(pool: SimpleConnectionPool) -> Iterator[connection]:
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
        pool.putconn(conn)

    except Exception:
        # always discard broken connections
        pool.putconn(conn, close=True)
        raise