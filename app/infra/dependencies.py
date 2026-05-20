from huggingface_hub import InferenceClient
from openai import OpenAI
from app.core.config import settings
from sentence_transformers import CrossEncoder
from app.infra.vector_stores.pgvector_store import PgVectorStore

def create_openai_client(api_key: str, base_url: str | None = None):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)

def create_vector_store(connection_string: str, embedding_dim: int, m: int, ef_construction: int):
    return PgVectorStore(connection_string=connection_string, embedding_dim=embedding_dim, m=m, ef_construction=ef_construction)

# API clients
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
gemini_openai_client = create_openai_client(api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_OPENAI_BASE_URL)
hf_inference_client = create_hf_inference_client(hf_token=settings.HF_TOKEN)

# ML models
# reranker_model = CrossEncoder(settings.RERANKER_MODEL_PATH)

# DBs
vector_store = create_vector_store(connection_string=settings.POSTGRES_URL, embedding_dim=settings.EMBEDDING_DIMENSIONS,
                                   m=settings.PGVECTOR_HNSW_M, ef_construction=settings.PGVECTOR_HNSW_EF_CONSTRUCTION)