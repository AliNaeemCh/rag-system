import logging
logger = logging.getLogger("app.infra.llm_engines.openai.adapters.base")
logger.info("Loading file...")

from abc import ABC, abstractmethod
from openai import AsyncOpenAI

class OpenAIBaseAdapter(ABC):

    @abstractmethod
    def build_messages(self, system_prompt: str | None, user_prompt: str | None, image_urls: str | list[str] | None, history: list[dict] | None):
        pass

    @abstractmethod
    def stream(self, model_name: str, client: AsyncOpenAI, request: dict):
        pass

    @abstractmethod
    def create(self, model_name: str, client: AsyncOpenAI, request: dict):
        pass

    @abstractmethod
    def extract_text(self, response):
        pass

    @abstractmethod
    def extract_usage(self, response: dict) -> dict:
        pass

    @abstractmethod
    def _build_payload(self, model_name: str, request: dict):
        pass