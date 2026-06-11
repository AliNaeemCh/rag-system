import logging
logger = logging.getLogger("app.core.config")
logger.info("Loading file...")

from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OverlapGranularity(str, Enum):
    SENTENCE_BASED = "sentence_based"
    WORD_BASED = "word_based"


class Settings(BaseSettings):

    # APIs
    TITLE: str = "Financial Report AI Assistant"
    SUB_TITLE: str = "Powered by the publicly available 2025 Systems Limited annual report"
    BACKEND_BASE_URL: str

    # Security
    API_KEY: str
    RATE_LIMIT: int = Field(default=10, gt=0)           # Total requests limit 
    WINDOW: int = Field(default=60, gt=0)               # Seconds
    PAYLOAD_LIMIT: int = Field(default=50_000, gt=0)    # Bytes

    @property
    def CHAT_URL(self) -> str:
        return self.BACKEND_BASE_URL + "/api/v1/chat"
    
    # LLM
    GEMINI_API_KEY: str
    OPENAI_API_KEY_SHARING: str
    HF_TOKEN: str
    GENERATOR_MODEL: str = "gpt-5.4-mini"
    REWRITER_MODEL: str = "gpt-5.4-nano"
    PDF_TRANSCRIBER_MODEL: str = "gemini-3.1-flash-lite"
    PDF_TRANSCRIBER_FALLBACK_MODEL: str = "gemini-2.5-flash-lite"
    EVAL_DATASET_GENERATOR_LLM: str = "gpt-5.4"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    GEMINI_OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    HF_BASE_URL: str = "https://router.huggingface.co/v1"

    # Embeddings
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"  # all-mpnet-base-v2: Max. seq. length: 384
    @property
    def EMBEDDING_MODEL_PATH(self) -> str:
        return BASE_DIR / "models" / "embedding" / self.EMBEDDING_MODEL
    
    RETRIEVAL_INSTRUCTION: str = ""
    EMBEDDING_DIMENSIONS: int = 768

    # Message rewriter
    REWRITER_KW_EXCLUDE_LIST: list[str] = ["Systems Limited", "Systems Ltd."]

    # Reranker
    RERANKER_MODEL: str = "ms-marco-MiniLM-L12-v2"   # ms-marco-MiniLM-L12-v2: 33.4M params. Max. seq. length: 512
    
    @property
    def RERANKER_MODEL_PATH(self) -> str:
        return BASE_DIR / "models" / "cross_encoder" / self.RERANKER_MODEL

    # DBs

    DB_POOL_MAX_CONNS: int = Field(default=10, gt=0)
        
    # - Usage tracker
    USAGE_TRACKER_DB_USER: str
    USAGE_TRACKER_DB_PASSWORD: str
    USAGE_TRACKER_DB_HOST: str
    USAGE_TRACKER_DB_PORT: int
    USAGE_TRACKER_DB_NAME: str

    USAGE_TRACKER_DB_POOL_MAX_CONNS: int = Field(default=10, gt=0)

    @property
    def USAGE_TRACKER_DB_URL(self) -> str:
        return (
            f"postgresql://{self.USAGE_TRACKER_DB_USER}:"
            f"{self.USAGE_TRACKER_DB_PASSWORD}@"
            f"{self.USAGE_TRACKER_DB_HOST}:"
            f"{self.USAGE_TRACKER_DB_PORT}/"
            f"{self.USAGE_TRACKER_DB_NAME}"
        )
    
    # Vector store
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 64
    HNSW_EF_SEARCH: int = 40

    # OpenSearch
    OPENSEARCH_USERNAME: str
    OPENSEARCH_PASSWORD: str
    OPENSEARCH_HOST: str
    OPENSEARCH_INDEX_NAME: str = "sys_annual_2025_rag"

    # RAG
    DENSE_TOP_K: int = Field(default=50, ge=1, le=100)
    SPARSE_TOP_K: int = Field(default=30, ge=1, le=50)
    FUSED_TOP_K: int = Field(default=20, ge=1, le=40)
    FINAL_TOP_K: int = Field(default=5, ge=1, le=50)

    # Chat
    CHAT_HISTORY_MAX_PAIRS: int = Field(default=5, ge=1)
    CHAT_HISTORY_TTL_SECONDS: int = Field(default=1800, ge=1)

    # Data
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

    # Evaluation
    EVAL_DATASET_DIR: Path = BASE_DIR / "evaluation" / "dataset"
    EVAL_RESULTS_DIR: Path = BASE_DIR / "evaluation" / "results"
    LLM_JUDGE_MODEL: str = "gpt-5.4-mini"

    # Ingestion
    CHUNK_SIZE: int = Field(default=350, gt=0)
    OVERLAP_TOKENS_PCT: int = Field(default=15, ge=0, lt=100)
    OVERLAP_GRANULARITY: OverlapGranularity = OverlapGranularity.SENTENCE_BASED
    SEPARATE_H2s: bool = True
    CROSS_SECTION_OVERLAP: bool = False
    CHUNKS_EMBEDDING_BATCH_SIZE: int = Field(default=2, gt=0)
    CHUNKS_SEARCH_INDEX_BATCH_SIZE: int = Field(default=64, gt=0)

    # Prompts
    ENTITY_NAME: str = "Systems Limited"
    ENTITY_DESCRIPTION: str = "a global IT services and software company providing digital transformation and technology solutions."

    # Misc
    LOG_LEVEL: LogLevel = LogLevel.INFO # DEBUG < INFO < WARNING < ERROR < CRITICAL
    USER_IN_MAX_CHARS: int = Field(default=250, gt=0)

    class Config:
        env_file = ".env"

settings = Settings()