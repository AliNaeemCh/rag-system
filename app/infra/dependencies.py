import logging
logger = logging.getLogger("app.infra.dependencies")
logger.info("Loading file...")

from pathlib import Path

def create_openai_client(api_key: str, base_url: str | None = None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    from huggingface_hub import InferenceClient
    return InferenceClient(provider="hf-inference", api_key=hf_token)

def create_opensearch_client(host: str, username: str, password: str, use_ssl: bool = True, verify_certs = False):
    from opensearchpy import OpenSearch
    return OpenSearch(hosts=[host], http_auth=(username, password), use_ssl=use_ssl, verify_certs=verify_certs, pool_maxsize=10)

# ML models
def get_reranker_model(model_path: Path):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_path)

def get_embedding_model(model_path: Path, device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(model_path), device=device)

# RAG pipeline

def build_rag_pipeline():
    from app.infra.usage_tracking.tracker import usage_tracker
    from app.infra.retrieval.opensearch.store import OpenSearchStore
    from app.rag.chat_history import chat_history
    from app.prompts.rag import GENERATOR_SYSTEM_PROMPT, REWRITER_SYSTEM_PROMPT, REWRITER_SCHEMA
    from app.infra.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
    from app.infra.llm_engines.openai.engine import OpenAIEngine
    from app.rag.pipeline import RAGPipeline
    from app.rag.message_rewriter import MessageRewriter
    from app.rag.retriever import Retriever
    from app.rag.reranker import Reranker
    from app.rag.generator import Generator
    from app.core.config import settings

    openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
    llm_generator = OpenAIEngine(model_name=settings.GENERATOR_MODEL, client=openai_client, usage_tracker=usage_tracker, check_usage=False)
    llm_rewriter = OpenAIEngine(model_name=settings.REWRITER_MODEL, client=openai_client, usage_tracker=usage_tracker, check_usage=False)
    opensearch_client = create_opensearch_client(host=settings.OPENSEARCH_HOST,
                                                username=settings.OPENSEARCH_USERNAME,
                                                password=settings.OPENSEARCH_PASSWORD)
    opensearch_store = OpenSearchStore(
        client=opensearch_client,
        index_name=settings.OPENSEARCH_INDEX_NAME,
        embedding_dim=settings.EMBEDDING_DIMENSIONS,
        m=settings.HNSW_M,
        ef_construction=settings.HNSW_EF_CONSTRUCTION
    )
    reranker_model = get_reranker_model(settings.RERANKER_MODEL_PATH)
    embedding_model = get_embedding_model(settings.EMBEDDING_MODEL_PATH)
    embedding_provider = SentenceTransformerEmbeddingProvider(model=embedding_model)
    rewriter = MessageRewriter(llm=llm_rewriter, system_prompt=REWRITER_SYSTEM_PROMPT, output_schema=REWRITER_SCHEMA)
    retriever = Retriever(embedding_provider=embedding_provider, retrieval_store=opensearch_store, dense_top_k=settings.DENSE_TOP_K, sparse_top_k=settings.SPARSE_TOP_K)
    reranker = Reranker(reranker_model=reranker_model, top_k=settings.FINAL_TOP_K)
    generator = Generator(llm_generator, system_prompt=GENERATOR_SYSTEM_PROMPT)

    return RAGPipeline(
        chat_history=chat_history,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        generator=generator
    )
