from app.core.logger import setup_logging
setup_logging()

from app.core.config import settings

import logging
logger = logging.getLogger("scripts.create_onnx_model")
logger.info("Loading file...")

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_DIR = settings.RERANKER_MODEL_PATH
OUTPUT_DIR = settings.RERANKER_ONX_MODEL_PATH

# Export to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    export=True,
)

model.save_pretrained(OUTPUT_DIR)

# Copy tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info(f"Exported to {OUTPUT_DIR}")