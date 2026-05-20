import logging
from app.core.logger import setup_logging
from tqdm import tqdm
from app.core.utils import load_jsonl
from app.core.config import settings
from app.infra.dependencies import hf_inference_client
from app.infra.embeddings.hugging_face import HuggingFaceEmbeddingProvider
from app.infra.vector_stores.pgvector_store import PgVectorStore
from app.ingestion.embeddings_generation.config import EmbeddingPipelineConfig

setup_logging()
logger = logging.getLogger("app.ingestion.plain_text_to_chunks.run")

def run_pipeline(config: EmbeddingPipelineConfig):
    try:
        logger.info("Initializing pipeline...")

        pbar = tqdm(total=100)
        embedding_model = HuggingFaceEmbeddingProvider(client=hf_inference_client, model=config.embedding_model)
        # vector store
        vector_store = PgVectorStore(
            connection_string=settings.POSTGRES_URL,
            embedding_dim=config.embedding_dim,
            m=config.m,
            ef_construction=config.ef_construction
        )

        # streaming batch buffers
        batch_texts = []
        batch_metas = []
        total = 0

        logger.info(f"Streaming ingestion started | file={config.chunks_jsonl_path}")

        for output in load_jsonl(config.chunks_jsonl_path, return_progress=True):
            obj, progress = output
            pbar.n = int(progress * 100)
            content = (obj.get("content") or "").strip()
            if not content:
                pbar.refresh()
                continue

            batch_texts.append(content)
            batch_metas.append(obj.get("metadata", {}))
            total += 1

            if len(batch_texts) >= config.batch_size:

                embeddings = embedding_model.embed_documents(batch_texts)
                if isinstance(embeddings[0], float):
                    embeddings = [embeddings]

                vector_store.add_documents(
                    texts=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_metas,
                )

                batch_texts.clear()
                batch_metas.clear()
            
            pbar.refresh()

        # flush remaining
        if batch_texts:
            embeddings = embedding_model.embed(batch_texts)
            if isinstance(embeddings[0], float):
                embeddings = [embeddings]

            vector_store.add_documents(
                texts=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
            )

        logger.info(f"Embedding Generation Completed | total_docs={total}")
    except Exception:
        logger.exception("Embedding Generation Failed!")