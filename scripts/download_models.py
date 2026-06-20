from app.core.logger import setup_logging
setup_logging()

from app.core.config import settings

import logging
logger = logging.getLogger("scripts.download_models")
logger.info("Loading file...")

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

MODELS_DIR = settings.BASE_DIR / "models"

MODELS = {
    "embedding": {
        "name": f"sentence-transformers/{settings.EMBEDDING_MODEL}",
        "path": MODELS_DIR / "embedding" / settings.EMBEDDING_MODEL,
    },
    "cross_encoder": {
        "name": f"cross-encoder/{settings.RERANKER_MODEL}",
        "path": MODELS_DIR / "cross_encoder" / settings.RERANKER_MODEL,
    },
}


def download_model(model_type: str, model_name: str, save_path: Path):
    if save_path.exists() and any(save_path.iterdir()):
        logger.info(f"Already exists: {save_path}")
        return
    
    logger.info(f"Downloading {model_name}...")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "cross_encoder":
        model = CrossEncoder(model_name)
    else:
        model = SentenceTransformer(model_name)
        
    model.save(str(save_path))

    logger.info(f"Saved to: {save_path}")


def main():
    for model_type, model in MODELS.items():
        download_model(
            model_type=model_type,
            model_name=model["name"],
            save_path=model["path"]
        )

    logger.info("All models downloaded successfully.")


if __name__ == "__main__":
    main()