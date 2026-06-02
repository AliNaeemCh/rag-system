from app.core.config import settings

import logging
logger = logging.getLogger("app.ingestion.search_index_generation.config")
logger.info("Loading file...")

from pathlib import Path
from dataclasses import dataclass

@dataclass(slots=True)
class SearchIndexPipelineConfig:
    chunks_jsonl_path: Path
    index_name: str = settings.OPENSEARCH_INDEX_NAME
    batch_size: int = settings.CHUNKS_SEARCH_INDEX_BATCH_SIZE
    resume: bool = False