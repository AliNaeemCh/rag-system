from app.rag.message_rewriter import MessageRewriter
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.infra.llm_engines.openai.engine import OpenAIEngine
from app.infra.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.prompts.rag import GENERATOR_SYSTEM_PROMPT, REWRITER_SYSTEM_PROMPT, REWRITER_SCHEMA
from app.rag.chat_history import chat_history
from app.core.config import settings
from app.infra.vector_stores.pgvector_store.store import PgVectorStore
from app.infra.db.pool import db_pool
from app.infra.usage_tracking.tracker import usage_tracker
from app.infra.search_engines.opensearch.store import OpenSearchStore

import logging
logger = logging.getLogger("app.infra.dependencies")
logger.info("Loading file...")

from openai import OpenAI
from sentence_transformers import CrossEncoder
from huggingface_hub import InferenceClient
from opensearchpy import OpenSearch

def create_openai_client(api_key: str, base_url: str | None = None):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)

def create_opensearch_client(host: str, username: str, password: str, use_ssl: bool = True, verify_certs = False):
    return OpenSearch(hosts=[host], http_auth=(username, password), use_ssl=use_ssl, verify_certs=verify_certs)

# API clients
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
gemini_openai_client = create_openai_client(api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_OPENAI_BASE_URL)

# ML models
reranker_model = CrossEncoder(settings.RERANKER_MODEL_PATH)

# DBs
vector_store = PgVectorStore(db_pool=db_pool, embedding_dim=settings.EMBEDDING_DIMENSIONS,
                             m=settings.PGVECTOR_HNSW_M, ef_construction=settings.PGVECTOR_HNSW_EF_CONSTRUCTION)

# OpenSearch
opensearch_client = create_opensearch_client(host=settings.OPENSEARCH_HOST,
                                             username=settings.OPENSEARCH_USERNAME,
                                             password=settings.OPENSEARCH_PASSWORD)

# RAG pipeline
logger.info("Loading RAG pipeline")
llm_generator = OpenAIEngine(model_name=settings.GENERATOR_MODEL, client=openai_client, usage_tracker=usage_tracker)
llm_rewriter = OpenAIEngine(model_name=settings.REWRITER_MODEL, client=openai_client, usage_tracker=usage_tracker)
embedding_model = SentenceTransformerEmbeddingProvider(model_path=settings.LOCAL_EMBEDDING_MODEL_PATH)
opensearch_store = OpenSearchStore(client=opensearch_client, index_name=settings.OPENSEARCH_INDEX_NAME)

def build_rag_pipeline():
    rewriter = MessageRewriter(llm=llm_rewriter, system_prompt=REWRITER_SYSTEM_PROMPT, output_schema=REWRITER_SCHEMA)
    retriever = Retriever(embedding_model=embedding_model, vector_store=vector_store, top_k=settings.RETRIEVER_INITIAL_K)
    reranker = Reranker(reranker_model=reranker_model, top_k=settings.TOP_K)
    generator = Generator(llm_generator, system_prompt=GENERATOR_SYSTEM_PROMPT)

    return RAGPipeline(
        chat_history=chat_history,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        generator=generator
    )

rag_pipeline = build_rag_pipeline()

def get_rag_pipeline():
    return rag_pipeline
