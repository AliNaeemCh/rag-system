from app.core.logger import setup_logging
setup_logging()

from app.core.utils import load_jsonl, extract_last_jsonl_object, reset_jsonl, write_jsonl
from app.core.config import settings
from app.infra.embeddings.base import BaseEmbeddingProvider
from app.infra.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from ingestion.embeddings_generation.config import EmbeddingPipelineConfig
from app.infra.dependencies import get_embedding_model

import logging
logger = logging.getLogger("ingestion.embeddings_generation.run")
logger.info("Loading file...")

import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def embed_batch(embedding_model: BaseEmbeddingProvider, texts: list[str], normalize: bool = True, batch_size: int | None = None):
    return embedding_model.embed_documents(texts, normalize=normalize, batch_size = batch_size)

def run_pipeline(config: EmbeddingPipelineConfig, input_path: Path, output_path: Path):
    try:
        logger.info("Initializing pipeline...")
        embedding_model = get_embedding_model(settings.LOCAL_EMBEDDING_MODEL_PATH)
        embedding_provider = SentenceTransformerEmbeddingProvider(model=embedding_model)

        # Thread pool for API calls
        executor = ThreadPoolExecutor(max_workers=config.num_workers)
        futures = []

        batch_texts = []
        batch_chunk_ids = []
        total = 0
        total_batches = 0
        completed_batches = 0
        last_processed_chunk_id = extract_last_jsonl_object(output_path).get("chunk_id", 0)

        if not config.resume:
            if last_processed_chunk_id > 0:
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously processed documents ({last_processed_chunk_id}) will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        reset_jsonl(output_path)
                        break
                    else:
                        print("Invalid input. Try again!")

            logger.info(f"Starting Embedding Generation... | File = {input_path}")
        
        else:
            logger.info(f"Resuming Embedding Generation... | File = {input_path}")

        pbar = tqdm(total=100)

        # -------------------------
        # STREAM + BATCH + SUBMIT
        # -------------------------
        for obj in load_jsonl(input_path):

            chunk_id = obj['chunk_id']
            if config.resume:
                if chunk_id <= last_processed_chunk_id:
                    total += 1

                    if chunk_id == last_processed_chunk_id:
                        total_batches = completed_batches = math.ceil(total / config.batch_size)
                    continue

            content = (obj.get("content") or "").strip()

            if not content:
                continue

            batch_texts.append(content)
            batch_chunk_ids.append(chunk_id)
            total += 1

            # submit batch to worker
            if len(batch_texts) >= config.batch_size:
                futures.append(
                    executor.submit(
                        embed_batch,
                        embedding_provider,
                        batch_texts.copy(),
                        batch_size=config.batch_size
                    )
                )
                # store metadata aligned with this batch
                futures[-1]._chunk_ids = batch_chunk_ids.copy()

                total_batches += 1

                batch_texts.clear()
                batch_chunk_ids.clear()

        # -------------------------
        # FLUSH REMAINING BATCH
        # -------------------------
        if batch_texts:
            futures.append(
                executor.submit(
                    embed_batch,
                    embedding_provider,
                    batch_texts.copy(),
                    normalize=True
                )
            )
            futures[-1]._chunk_ids = batch_chunk_ids.copy()
            futures[-1]._texts = batch_texts.copy()
            total_batches += 1

        total_batches = total_batches or completed_batches
        pbar.n = int((completed_batches / total_batches) * 100)
        pbar.refresh()

        # -------------------------
        # COLLECT RESULTS + WRITE
        # -------------------------
        for future in futures:  # Preserves order

            embeddings = future.result()

            if not embeddings:
                raise Exception("Undefined embedding(s)")

            objects = []

            for i in range(len(future._chunk_ids)):
                objects.append(
                    {
                        "chunk_id": future._chunk_ids[i],
                        "embedding": embeddings[i]
                    }
                )
            
            write_jsonl(objects, output_path)

            completed_batches += 1
            pbar.n = int((completed_batches / total_batches) * 100)
            pbar.refresh()

        executor.shutdown(wait=True)

        logger.info(f"Embedding Generation Completed | Total Docs = {total} | Total Batches = {total_batches}")

    except Exception as e:
        logger.exception(f"Embedding Generation Failed!")
        return

config = EmbeddingPipelineConfig(resume = True)

run_pipeline(
    config=config,
    input_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    output_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_embeddings.jsonl",
)