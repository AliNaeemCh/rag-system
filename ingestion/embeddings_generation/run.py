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
from pathlib import Path
from collections import deque
import asyncio
import sys
import selectors

async def embed_batch(embedding_provider: BaseEmbeddingProvider, texts: list[str], normalize: bool = True, batch_size: int | None = None):
    return await embedding_provider.embed_documents(texts, normalize=normalize, batch_size = batch_size)

async def run_pipeline(config: EmbeddingPipelineConfig, input_path: Path, output_path: Path):
    try:
        logger.info("Initializing pipeline...")
        embedding_model = get_embedding_model(settings.EMBEDDING_MODEL_PATH)
        embedding_provider = SentenceTransformerEmbeddingProvider(model=embedding_model)

        tasks = deque()
        task_to_metadata = {}

        batch_texts = []
        batch_chunk_ids = []
        total = 0
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
        initial_progress_refresh = False
        # -------------------------
        # STREAM + BATCH + SUBMIT
        # -------------------------
        for obj, progress in load_jsonl(input_path, return_progress=True):

            chunk_id = obj['chunk_id']
            if config.resume and chunk_id <= last_processed_chunk_id:
                initial_progress_refresh = True
                continue
            
            if initial_progress_refresh:
                pbar.n = int(progress * 100)
                pbar.refresh()
                initial_progress_refresh = False

            content = (obj.get("content") or "").strip()

            if not content:
                continue

            batch_texts.append(content)
            batch_chunk_ids.append(chunk_id)
            total += 1

            # submit batch to worker
            if len(batch_texts) >= config.batch_size:
                if len(tasks) >= config.max_concurrency:
                    task = tasks.popleft()
                    embeddings = await task

                    if not embeddings:
                        raise Exception("Undefined embedding(s)")

                    objects = []

                    for i in range(len(task_to_metadata[task]['chunk_ids'])):
                        objects.append(
                            {
                                "chunk_id": task_to_metadata[task]['chunk_ids'][i],
                                "embedding": embeddings[i]
                            }
                        )
                    
                    write_jsonl(objects, output_path)
                    del task_to_metadata[task]

                    pbar.n = int(progress * 100)
                    pbar.refresh()
                
                task = asyncio.create_task(
                        embed_batch(
                        embedding_provider,
                        batch_texts.copy(),
                        normalize=True,
                        batch_size=config.batch_size
                        ))
                tasks.append(task)
                # store metadata aligned with this batch
                task_to_metadata[task] = {
                    "chunk_ids": batch_chunk_ids.copy()
                }

                batch_texts.clear()
                batch_chunk_ids.clear()

        # -------------------------
        # FLUSH REMAINING BATCH & TASKS
        # -------------------------
        if batch_texts:
            task = asyncio.create_task(
                    embed_batch(
                    embedding_provider,
                    batch_texts.copy(),
                    normalize=True,
                    batch_size=config.batch_size
                    ))
            tasks.append(task)
            task_to_metadata[task] = {
                "chunk_ids": batch_chunk_ids.copy()
            }

            batch_texts.clear()
            batch_chunk_ids.clear()

            while tasks:
                task = tasks.popleft()
                embeddings = await task

                if not embeddings:
                    raise Exception("Undefined embedding(s)")

                objects = []

                for i in range(len(task_to_metadata[task]['chunk_ids'])):
                    objects.append(
                        {
                            "chunk_id": task_to_metadata[task]['chunk_ids'][i],
                            "embedding": embeddings[i]
                        }
                    )
                
                write_jsonl(objects, output_path)
                del task_to_metadata[task]

                pbar.n = 100
                pbar.refresh()

        pbar.n = 100
        pbar.refresh()

        logger.info(f"Embedding Generation Completed | Total Docs = {total} | Total Batches = {math.ceil(total/config.batch_size)}")

    except Exception as e:
        logger.exception(f"Embedding Generation Failed!")
        return

config = EmbeddingPipelineConfig(resume = False)

async def main():

    await run_pipeline(
        config=config,
        input_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
        output_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_embeddings.jsonl",
    )

if sys.platform == "win32":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        ),
    )
else:
    asyncio.run(main())