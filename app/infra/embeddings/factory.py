from app.core.config import Settings
from app.infra.embeddings.registry import EMBEDDING_MODEL_BUILDER_REGISTRY

def create_embedding_model(settings: Settings):

    builder = EMBEDDING_MODEL_BUILDER_REGISTRY.get(settings.EMBEDDING_MODEL)

    if not builder:
        raise ValueError(
            f"Unsupported provider: {settings.EMBEDDING_MODEL}"
        )

    return builder(settings)