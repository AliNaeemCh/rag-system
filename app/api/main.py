from app.core.config import settings
from app.core.logger import setup_logging
import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.retriever import Retriever
from app.rag.reranker import re
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.infra.llm_engines.openai.engine import OpenAIEngine
from openai import OpenAI
from app.infra.dependencies import vector_store
from app.infra.dependencies import openai_client, hf_inference_client, reranker_model
from app.infra.embeddings.hugging_face import HuggingFaceEmbeddingProvider

llm_generator = OpenAIEngine(openai_client, settings.GENERATOR_MODEL)
llm_rewriter = OpenAIEngine(openai_client, settings.REWRITER_MODEL)
embedding_model = HuggingFaceEmbeddingProvider(client=hf_inference_client, model=settings.EMBEDDING_MODEL)

def build_pipeline():
    rewriter = MessageRewriter(llm_rewriter)
    retriever = Retriever(embedding_model=embedding_model, vector_store=vector_store, top_k=settings.TOP_K)
    reranker = Reranker()
    generator = Generator()

    return RAGPipeline(
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        generator=generator
    )

setup_logging()

# set log level from environment
logging.getLogger().setLevel(settings.LOG_LEVEL.value)
