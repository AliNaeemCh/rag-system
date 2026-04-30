from app.core.config import Settings
from app.core.logger import setup_logging
import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.retriever import Retriever
from app.rag.reranker import re
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.infra.llm_engines.openai import LLMEngine
from openai import OpenAI
from sentence_transformers import CrossEncoder
from huggingface_hub import InferenceClient
from app.infra.vectorstore import FAISSVectorStore
from app.infra.clients import create_openai_client
from app.infra.embeddings.factory import create_embedding_model

settings = Settings()

openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
reranker_client = CrossEncoder(settings.RERANKER_MODEL)

llm_generator = LLMEngine(openai_client, settings.GENERATOR_MODEL)
llm_rewriter = LLMEngine(openai_client, settings.REWRITER_MODEL)

embedding_model = create_embedding_model(settings)
vectorstore = FAISSVectorStore(embedding_model, dim=settings.EMBEDDING_DIMENSIONS)


def build_pipeline():
    rewriter = MessageRewriter(llm_rewriter)
    retriever = Retriever()
    reranker = Reranker
    generator = Generator()

    return RAGPipeline(
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        generator=generator
    )

setup_logging()

# set log level from environment
logging.getLogger().setLevel(settings.LOG_LEVEL)
