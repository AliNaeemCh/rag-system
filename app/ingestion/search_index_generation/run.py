from app.core.logger import setup_logging
setup_logging()

from app.core.utils import load_jsonl
from app.core.config import settings
from app.infra.search_engines.opensearch.store import OpenSearchStore
from app.ingestion.search_index_generation.config import SearchIndexPipelineConfig

import logging
logger = logging.getLogger("app.ingestion.search_index_generation.run")
logger.info("Loading file...")

from tqdm import tqdm
from opensearchpy import OpenSearch

def run_pipeline(config: SearchIndexPipelineConfig):

    try:
        logger.info("Initializing pipeline...")

        client = OpenSearch(hosts=[settings.OPENSEARCH_HOST],
                            http_auth=(settings.OPENSEARCH_USERNAME,settings.OPENSEARCH_PASSWORD),
                            use_ssl=True,
                            verify_certs=False,
                            ssl_show_warn=False)

        opensearch_store = OpenSearchStore(
            client=client,
            index_name=settings.OPENSEARCH_INDEX_NAME
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

            logger.info(f"Starting Search Index Generation... | File = {config.chunks_jsonl_path}")
        
        else:
            logger.info(f"Resuming Search Index Generation... | File = {config.chunks_jsonl_path}")

        pbar = tqdm(total=100)

        for obj, progress in load_jsonl(config.chunks_jsonl_path, return_progress=True):

            chunk_id = obj['chunk_id']

            if config.resume:
                if chunk_id <= last_processed_chunk_id:
                    total += 1

                    if chunk_id == last_processed_chunk_id:
                        pbar.n = int(progress * 100)
                        pbar.refresh()
                    
                    continue

            content = (obj.get("content") or "").strip()
            if not content:
                continue

            batch.append({
                "chunk_id": obj["chunk_id"],
                "text": content,
                "metadata": obj.get("metadata", {})
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

    except Exception:
        logger.exception("Search Index Generation Failed!")
        return

config = SearchIndexPipelineConfig(
    chunks_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    resume=False
)

run_pipeline(config)