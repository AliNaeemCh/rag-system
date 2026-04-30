from app.infra.embeddings.bge import BGEEmbeddingModel
from app.infra.clients import create_hf_inference_client
from app.core.config import Settings


def build_bge(settings: Settings): 
    client = create_hf_inference_client(settings.HF_TOKEN)
    return BGEEmbeddingModel(client=client, model=settings.EMBEDDING_MODEL)

EMBEDDING_MODEL_BUILDER_REGISTRY = {
    "BAAI/bge-large-en-v1.5": build_bge
}