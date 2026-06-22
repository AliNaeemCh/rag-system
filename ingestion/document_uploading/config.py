from app.core.config import settings

import logging
logger = logging.getLogger("ingestion.document_uploading.config")
logger.info("Loading file...")

from dataclasses import dataclass

@dataclass(slots=True)
class DocumentUploadingPipelineConfig:
    batch_size: int = 64
    resume: bool = False