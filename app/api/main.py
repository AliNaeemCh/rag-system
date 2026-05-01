from app.core.config import Settings
from app.core.logger import setup_logging
import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.retriever import Retriever
from app.rag.reranker import re
from app.rag.generator import Generator
from app.rag.pipeline import RAGPipeline
from app.infra.llms.engines.openai.engine import OpenAIEngine
from openai import OpenAI
from sentence_transformers import CrossEncoder
from app.infra.vectorstore import FAISSVectorStore
from app.infra.clients import create_openai_client, create_hf_inference_client
from app.infra.embeddings.hugging_face import HuggingFaceEmbeddingProvider

settings = Settings()

openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
hf_inference_client = create_hf_inference_client(hf_token=settings.HF_TOKEN)
reranker_client = CrossEncoder(settings.RERANKER_MODEL)

llm_generator = OpenAIEngine(openai_client, settings.GENERATOR_MODEL)
llm_rewriter = OpenAIEngine(openai_client, settings.REWRITER_MODEL)

embedding_model = HuggingFaceEmbeddingProvider(client=hf_inference_client, model=settings.EMBEDDING_MODEL)
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
