from huggingface_hub import InferenceClient
from openai import OpenAI
from app.core.config import settings
from sentence_transformers import CrossEncoder

def create_openai_client(api_key: str, base_url: str | None = None):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)

openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
gemini_openai_client = create_openai_client(api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_OPENAI_BASE_URL)
hf_inference_client = create_hf_inference_client(hf_token=settings.HF_TOKEN)
reranker_client = CrossEncoder(settings.RERANKER_MODEL_PATH)