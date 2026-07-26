from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer


def load_model(model_path: Path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    session = ort.InferenceSession(
        str(model_path / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    return session, tokenizer