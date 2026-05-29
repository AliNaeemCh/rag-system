import logging
from app.core.logger import setup_logging
setup_logging()
logger = logging.getLogger("app.ingestion.embeddings_generation.run")

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from app.core.utils import load_jsonl
from app.core.config import settings
# from app.infra.dependencies import hf_inference_client
from app.infra.embeddings.base import BaseEmbeddingProvider
from app.infra.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.infra.vector_stores.pgvector_store.store import PgVectorStore
from app.ingestion.embeddings_generation.config import EmbeddingPipelineConfig
import math

def embed_batch(embedding_model: BaseEmbeddingProvider, texts: list[str], normalize: bool = True, batch_size: int | None = None):
    return embedding_model.embed_documents(texts, normalize=normalize, batch_size = batch_size)

def run_pipeline(config: EmbeddingPipelineConfig):
    try:
        logger.info("Initializing pipeline...")


        embedding_model = SentenceTransformerEmbeddingProvider(model_path=settings.LOCAL_EMBEDDING_MODEL_PATH)

        vector_store = PgVectorStore(
            connection_string=settings.POSTGRES_URL,
            embedding_dim=config.embedding_dim,
            m=config.m,
            ef_construction=config.ef_construction
        )

        # Thread pool for API calls
        executor = ThreadPoolExecutor(max_workers=config.num_workers)
        futures = []

        batch_texts = []
        batch_metas = []
        batch_chunk_ids = []
        total = 0
        total_batches = 0
        completed_batches = 0
        last_processed_chunk_id = None

        if not config.resume:
            while True:
                user_in = input("Warning: Previously saved documents (if any) will be deleted. Type 'confirm' to proceed: ")
                if user_in == "confirm":
                    vector_store.reset_schema()
                    break
                else:
                    print("Invalid input. Try again!")
            logger.info(f"Starting Embedding Generation... | File = {config.chunks_jsonl_path}")
        
        else:
            logger.info(f"Resuming Embedding Generation... | File = {config.chunks_jsonl_path}")

        pbar = tqdm(total=100)

        # -------------------------
        # STREAM + BATCH + SUBMIT
        # -------------------------
        for obj in load_jsonl(config.chunks_jsonl_path):

            content = (obj.get("content") or "").strip()
            chunk_id = obj['chunk_id']
            if config.resume:
                if last_processed_chunk_id is None:
                    last_processed_chunk_id = vector_store.get_max_id()

                if chunk_id <= last_processed_chunk_id:
                    total += 1

                    if chunk_id == last_processed_chunk_id:
                        completed_batches = math.ceil(total / config.batch_size)
                    continue

            if not content:
                continue

            batch_texts.append(content)
            batch_metas.append(obj.get("metadata", {}))
            batch_chunk_ids.append(chunk_id)
            total += 1

            # submit batch to worker
            if len(batch_texts) >= config.batch_size:
                futures.append(
                    executor.submit(
                        embed_batch,
                        embedding_model,
                        batch_texts.copy(),
                        batch_size=config.batch_size
                    )
                )
                # store metadata aligned with this batch
                futures[-1]._chunk_ids = batch_chunk_ids.copy()
                futures[-1]._texts = batch_texts.copy()
                futures[-1]._metas = batch_metas.copy()
                total_batches += 1

                batch_texts.clear()
                batch_metas.clear()
                batch_chunk_ids.clear()

        # -------------------------
        # FLUSH REMAINING BATCH
        # -------------------------
        if batch_texts:
            futures.append(
                executor.submit(
                    embed_batch,
                    embedding_model,
                    batch_texts.copy(),
                    normalize=True
                )
            )
            futures[-1]._chunk_ids = batch_chunk_ids.copy()
            futures[-1]._texts = batch_texts.copy()
            futures[-1]._metas = batch_metas.copy()
            total_batches += 1
        total_batches = total_batches or completed_batches
        pbar.n = int((completed_batches / total_batches) * 100)
        pbar.refresh()
        # -------------------------
        # COLLECT RESULTS + WRITE
        # -------------------------
        for future in futures:  # Preserves order
            chunk_ids = future._chunk_ids
            texts = future._texts
            metas = future._metas
            embeddings = future.result()

            if not embeddings:
                continue

            # normalize shape if API returns flat list
            if isinstance(embeddings[0], float):
                embeddings = [embeddings]

            vector_store.add_documents(
                chunk_ids=chunk_ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metas,
            )

            completed_batches += 1
            pbar.n = int((completed_batches / total_batches) * 100)
            pbar.refresh()

        executor.shutdown(wait=True)
        vector_store.close_conn()

        logger.info(f"Embedding Generation Completed | Total Docs = {total} | Total Batches = {total_batches}")

    except Exception:
        logger.exception("Embedding Generation Failed! Please resume the process manually")
        return

config = EmbeddingPipelineConfig(
    chunks_jsonl_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    resume = False
)

run_pipeline(config)