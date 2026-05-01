from huggingface_hub import InferenceClient
from openai import OpenAI

def create_openai_client(api_key: str, base_url: str | None = None):
    return OpenAI(api_key=api_key, base_url=base_url)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)
