import logging
logger = logging.getLogger("app.infra.vector_stores.pgvector_store.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class RetrievedDocument:
    id: int
    content: str
    metadata: dict
    distance: float