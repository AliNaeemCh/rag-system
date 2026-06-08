from app.infra.llm_engines.base import BaseLLMEngine
from app.infra.llm_engines.openai.adapters.responses import OpenAIResponsesAdapter
from app.infra.llm_engines.openai.adapters.chat_completions import OpenAIChatCompletionsAdapter
from app.infra.usage_tracking.tracker import UsageTracker

import logging
logger = logging.getLogger("app.infra.llm_engines.openai.engine")
logger.info("Loading file...")

from enum import Enum
from openai import OpenAI

class OpenAIAPI(str, Enum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"

class OpenAIEngine(BaseLLMEngine):

    def __init__(self, model_name: str, client: OpenAI, api: OpenAIAPI = OpenAIAPI.RESPONSES, usage_tracker: UsageTracker | None = None, check_usage: bool = True):
        super().__init__(model_name=model_name, usage_tracker=usage_tracker, check_usage=check_usage)
        self.client = client

        # choose adapter
        if api == OpenAIAPI.RESPONSES:
            self.adapter = OpenAIResponsesAdapter()
        elif api == OpenAIAPI.CHAT_COMPLETIONS:
            self.adapter = OpenAIChatCompletionsAdapter()
        else:
            raise ValueError("Invalid API")
        
    def _build_request(
        self,
        system_prompt,
        user_prompt,
        history,
        temperature,
        schema,
        reasoning,
        image_urls,
    ):
        messages = self.adapter.build_messages(
            system_prompt,
            user_prompt,
            image_urls,
            history,
        )

        return {
            "messages": messages,
            "temperature": temperature,
            "schema": schema,
            "reasoning": reasoning,
        }

    def _stream(self, request):
        gen, state = self.adapter.stream(
            model_name=self.model_name,
            client=self.client,
            request=request,
        )

        def wrapper():
            yield from gen
            final_response = state["final_response"]
            if final_response:
                self._increment_usage(
                    response=final_response,
                    model_name=self.model_name,
                )

        return wrapper()

    def _create(self, request):
        return self.adapter.create(model_name=self.model_name, client=self.client, request=request)

    def _extract_text(self, response):
        return self.adapter.extract_text(response)

    def _extract_usage(self, response: dict) -> dict:
        return self.adapter.extract_usage(response)