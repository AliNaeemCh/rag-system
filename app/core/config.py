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


class Settings(BaseSettings):
    # LLM
    GEMINI_API_KEY: str
    OPENAI_API_KEY_SHARING: str
    HF_TOKEN: str
    GENERATOR_MODEL: str = "gpt-5.4-mini"
    REWRITER_MODEL: str = "gpt-5.4-nano"
    PDF_TRANSCRIBER_MODEL: str = "gemini-3.5-flash-lite-preview"
    PDF_TRANSCRIBER_FALLBACK_MODEL: str = "gemini-2.5-flash-lite"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    GEMINI_OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    HF_BASE_URL: str = "https://router.huggingface.co/v1"

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RETRIEVAL_INSTRUCTION: str = "Represent this sentence for searching relevant passages:"
    EMBEDDING_DIMENSIONS: int = 1024

    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L12-v2"   # cross-encoder/ms-marco-MiniLM-L12-v2: 33.4M params

    # Vector DB
    VECTOR_DB_TYPE: str = "faiss"
    INDEX_PATH: str = "data/processed/index"

    # RAG params
    RETRIEVER_INITIAL_K: int = Field(default=20, ge=1, le=200)
    TOP_K: int = Field(default=5, ge=1, le=50)

    # Chat
    CHAT_HISTORY_MAX_PAIRS: int = Field(default=5, ge=1)
    CHAT_HISTORY_TTL_SECONDS: int = Field(default=1800, ge=1)

    # Data
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

    # Misc
    LOG_LEVEL: LogLevel = LogLevel.INFO # DEBUG < INFO < WARNING < ERROR < CRITICAL

    class Config:
        env_file = ".env"