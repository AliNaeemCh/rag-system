from huggingface_hub import InferenceClient
from openai import OpenAI
from app.core.config import settings
from sentence_transformers import CrossEncoder
from app.infra.vector_stores.pgvector_store.store import PgVectorStore
import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.infra.llm_engines.openai.engine import OpenAIEngine
from app.infra.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.prompts.rag import GENERATOR_SYSTEM_PROMPT, REWRITER_SYSTEM_PROMPT, REWRITER_SCHEMA
from app.rag.chat_history import ChatHistory

logger = logging.getLogger("app.infra.dependencies")

def create_openai_client(api_key: str, base_url: str | None = None):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)

def create_vector_store(connection_string: str, embedding_dim: int, m: int, ef_construction: int):
    return PgVectorStore(connection_string=connection_string, embedding_dim=embedding_dim, m=m, ef_construction=ef_construction)

# API clients
logger.info("Loading OpenAI")
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)

logger.info("Loading Gemini OpenAI client")
gemini_openai_client = create_openai_client(api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_OPENAI_BASE_URL)

# ML models
logger.info("Loading reranker model")
reranker_model = CrossEncoder(settings.RERANKER_MODEL_PATH)

# DBs
logger.info("Loading vector store")
vector_store = create_vector_store(connection_string=settings.POSTGRES_URL, embedding_dim=settings.EMBEDDING_DIMENSIONS,
                                   m=settings.PGVECTOR_HNSW_M, ef_construction=settings.PGVECTOR_HNSW_EF_CONSTRUCTION)

# RAG pipeline
logger.info("Loading RAG pipeline")
llm_generator = OpenAIEngine(openai_client, settings.GENERATOR_MODEL)
llm_rewriter = OpenAIEngine(openai_client, settings.REWRITER_MODEL)
embedding_model = SentenceTransformerEmbeddingProvider(model_path=settings.LOCAL_EMBEDDING_MODEL_PATH)
chat_history = ChatHistory()

def build_rag_pipeline():
    rewriter = MessageRewriter(llm=llm_rewriter, system_prompt=REWRITER_SYSTEM_PROMPT, output_schema=REWRITER_SCHEMA)
    retriever = Retriever(embedding_model=embedding_model, vector_store=vector_store,
                          top_k=settings.RETRIEVER_INITIAL_K, retrieval_instruction=settings.RETRIEVAL_INSTRUCTION)
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
