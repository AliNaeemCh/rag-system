from app.core.logger import setup_logging
setup_logging()

from app.core.utils import load_jsonl
from app.core.config import settings
from app.infra.retrieval.postgres.store import PgStore
from app.infra.retrieval.base import BaseDocumentStore
from ingestion.document_uploading.config import DocumentUploadingPipelineConfig
from app.infra.db.pool import get_rag_db_pool

import logging
logger = logging.getLogger("ingestion.document_uploading.run")
logger.info("Loading file...")

from tqdm import tqdm
from pathlib import Path

def run_pipeline(
        config: DocumentUploadingPipelineConfig,
        document_store: BaseDocumentStore,
        chunks_jsonl_path: Path,
        embeddings_jsonl_path: Path):

    try:
        logger.info("Initializing pipeline...")

        batch = []
        last_processed_chunk_id = document_store.get_max_chunk_id()
        total = 0

        if not config.resume:
            if last_processed_chunk_id > 0:
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously processed documents ({last_processed_chunk_id}) will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        document_store.reset_store()
                        break
                    else:
                        print("Invalid input. Try again!")

            logger.info(f"Document Uploading...")
        
        else:
            logger.info(f"Resuming Document Uploading...")

        pbar = tqdm(total=100)

        for (chunk_obj, progress), (emb_obj, _) in zip(
            load_jsonl(chunks_jsonl_path, return_progress=True),
            load_jsonl(embeddings_jsonl_path, return_progress=True)
        ):

            chunk_id = chunk_obj['chunk_id']

            if chunk_id != emb_obj['chunk_id']:
                raise Exception("Chunk and embedding objects are not aligned!")

            if config.resume:
                if chunk_id <= last_processed_chunk_id:
                    total += 1

                    if chunk_id == last_processed_chunk_id:
                        pbar.n = int(progress * 100)
                        pbar.refresh()
                    
                    continue

            content = (chunk_obj.get("content") or "").strip()
            if not content:
                raise Exception(f"Undefined content against chunk_id={chunk_id}")
            
            if not emb_obj.get("embedding"):
                raise Exception(f"Undefined embedding against chunk_id={chunk_id}")

            batch.append({
                "chunk_id": chunk_id,
                "content": content,
                "embedding": emb_obj.get("embedding"),
                "metadata": chunk_obj.get("metadata", {})
            })

            total += 1

            if len(batch) >= config.batch_size:
                document_store.add_documents_bulk(batch)
                batch.clear()
                pbar.n = int(progress * 100)
                pbar.refresh()

        # flush remaining
        if batch:
            document_store.add_documents_bulk(batch)

        pbar.n = 100
        pbar.refresh()

        logger.info(f"Document Uploading Completed | Total Docs = {total}")

    except Exception as e:
        logger.exception(f"Document Uploading Failed!")
        return

config = DocumentUploadingPipelineConfig(resume=False)

rag_db_pool = get_rag_db_pool()


pg_store: BaseDocumentStore = PgStore(
    db_pool=rag_db_pool,
    embedding_dim=settings.EMBEDDING_DIMENSIONS,
    m=settings.HNSW_M,
    ef_construction=settings.HNSW_EF_CONSTRUCTION
)

run_pipeline(
    document_store=pg_store,
    config=config,
    chunks_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    embeddings_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_embeddings.jsonl",
)