from app.infra.llms.engines.base import BaseLLMEngine
from app.infra.llms.engines.openai.adapters.responses import OpenAIResponsesAdapter
from app.infra.llms.engines.openai.adapters.chat_completions import OpenAIChatCompletionsAdapter
from enum import Enum

class OpenAIAPI(str, Enum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"

class OpenAIEngine(BaseLLMEngine):

    def __init__(self, client, model, api: OpenAIAPI = OpenAIAPI.RESPONSES):
        self.client = client
        self.model = model
        self.api = api

        # choose adapter
        if api == OpenAIAPI.RESPONSES:
            self.adapter = OpenAIResponsesAdapter(client, model)
        elif api == OpenAIAPI.CHAT_COMPLETIONS:
            self.adapter = OpenAIChatCompletionsAdapter(client, model)
        else:
            raise ValueError("Invalid API")
        
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
        messages = self.adapter.build_messages(
            system_prompt,
            user_prompt,
            image,
            history,
        )

        return {
            "messages": messages,
            "temperature": temperature,
            "schema": schema,
            "reasoning": reasoning,
        }

    def stream(self, request):
        return self.adapter.stream(request)

    def create(self, request):
        return self.adapter.create(request)

    def extract_text(self, response):
        return self.adapter.extract_text(response)