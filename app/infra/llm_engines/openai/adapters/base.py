from abc import ABC, abstractmethod

class OpenAIBaseAdapter(ABC):

    @abstractmethod
    def build_messages(self, system_prompt: str | None, user_prompt: str | None, image_urls: str | list[str] | None, history: list[dict] | None):
        pass

    @abstractmethod
    def build_payload(self, request):
        pass

    @abstractmethod
    def stream(self, request):
        pass

    @abstractmethod
    def create(self, request):
        pass

    @abstractmethod
    def extract_text(self, response):
        pass