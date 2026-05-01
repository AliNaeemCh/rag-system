import json
from abc import ABC, abstractmethod


class BaseLLMEngine(ABC):

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        history: list[dict] | None = None,
        *,
        stream: bool = False,
        schema=None,
        temperature=None,
        reasoning=None,
        image=None
    ):
        history = history or []

        request = self.build_request(
            system_prompt,
            user_prompt,
            history,
            temperature,
            schema,
            reasoning,
            image,
        )

        if stream:
            return self.stream(request)

        response = self.create(request)
        text = self.extract_text(response)

        if schema:
            return json.loads(text)

        return text

    # ---------------- abstract hooks ----------------

    @abstractmethod
    def build_request(
        self,
        system_prompt,
        user_prompt,
        history,
        temperature,
        schema,
        reasoning,
        image,
    ):
        pass

    @abstractmethod
    def stream(self, request: dict):
        pass

    @abstractmethod
    def create(self, request: dict):
        pass

    @abstractmethod
    def extract_text(self, response):
        pass