from huggingface_hub import InferenceClient
from openai import OpenAI

def create_openai_client(api_key: str):
    return OpenAI(api_key)

def create_hf_inference_client(hf_token: str):
    return InferenceClient(provider="hf-inference", api_key=hf_token)
