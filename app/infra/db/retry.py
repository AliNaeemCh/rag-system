import logging
logger = logging.getLogger("app.infra.db.retry")
logger.info("Loading file...")

from functools import wraps
from psycopg2 import OperationalError, InterfaceError

def db_retry(retries=1):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)

                except (OperationalError, InterfaceError):
                    if attempt == retries:
                        raise
        return wrapper
    return decorator