from app.core.logger import setup_logging
setup_logging()

from app.core.utils import load_jsonl
from app.core.config import settings
from app.infra.search_engines.opensearch.store import OpenSearchStore
from app.ingestion.search_indexing.opensearch.config import SearchIndexingPipelineConfig

import logging
logger = logging.getLogger("app.ingestion.search_index_generation.run")
logger.info("Loading file...")

from tqdm import tqdm
from opensearchpy import OpenSearch
from pathlib import Path

def run_pipeline(
        config: SearchIndexingPipelineConfig,
        chunks_jsonl_path: Path,
        embeddings_jsonl_path: Path):

    try:
        logger.info("Initializing pipeline...")

        client = OpenSearch(hosts=[settings.OPENSEARCH_HOST],
                            http_auth=(settings.OPENSEARCH_USERNAME,settings.OPENSEARCH_PASSWORD),
                            use_ssl=True,
                            verify_certs=False,
                            ssl_show_warn=False)

        opensearch_store = OpenSearchStore(
            client=client,
            index_name=settings.OPENSEARCH_INDEX_NAME,
            embedding_dim=settings.EMBEDDING_DIMENSIONS,
            m=settings.HNSW_M,
            ef_construction=settings.HNSW_EF_CONSTRUCTION
        )

        batch = []
        total = 0
        last_processed_chunk_id = opensearch_store.get_max_chunk_id()

        if not config.resume:
            if last_processed_chunk_id > 0:
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously processed documents ({last_processed_chunk_id}) will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        opensearch_store.reset_index()
                        break
                    else:
                        print("Invalid input. Try again!")

            logger.info(f"Starting Search Index Generation...")
        
        else:
            logger.info(f"Resuming Search Index Generation...")

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
                opensearch_store.add_chunks_bulk(batch)
                batch.clear()
                pbar.n = int(progress * 100)
                pbar.refresh()

        # flush remaining
        if batch:
            opensearch_store.add_chunks_bulk(batch)

        pbar.n = 100
        pbar.refresh()

        logger.info(f"Search Index Generation Completed | Total Docs = {total}")

    except Exception as e:
        logger.exception(f"Search Index Generation Failed! Error: {e}")
        return

config = SearchIndexingPipelineConfig(resume=False)

run_pipeline(
    config=config,
    chunks_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    embeddings_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_embeddings.jsonl",
)