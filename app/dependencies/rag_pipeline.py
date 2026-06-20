import logging
logger = logging.getLogger("app.dependencies.rag_pipeline")
logger.info("Loading file...")

from app.infra.dependencies import build_rag_pipeline

pipeline = build_rag_pipeline()

def rag_pipeline():
    return pipeline