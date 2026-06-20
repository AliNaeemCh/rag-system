from app.infra.usage_tracking.tracker import UsageTracker
from app.infra.executor import executor

import logging
logger = logging.getLogger("app.infra.llm_engines.base")
logger.info("Loading file...")

import json
from abc import ABC, abstractmethod


class BaseLLMEngine(ABC):
    def __init__(self, model_name: str, usage_tracker: UsageTracker | None = None, check_usage: bool = True):
        self.model_name = model_name
        self.usage_tracker = usage_tracker
        self.check_usage = check_usage

    def generate(
        self,
        user_prompt: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        history: list[dict] | None = None,
        *,
        stream: bool = False,
        schema=None,
        temperature=None,
        reasoning=None,
        image_urls: str | list[str] | None = None,
        return_full_response: bool = False
    ):
        if self.usage_tracker and self.check_usage:
            if self.usage_tracker.usage_exceeded(model_names=[self.model_name], safety_margin_tokens=5000):
                raise Exception("Usage limit exceeded!")
            
        history = history or []

        request = self._build_request(
            system_prompt,
            user_prompt,
            history,
            temperature,
            schema,
            reasoning,
            image_urls,
        )

        if stream:
            return self._stream(request)

        response = self._create(request)

        executor.submit(self._increment_usage, response=response, model_name=self.model_name)   # Token increment runs in bg

        text = self._extract_text(response)

        if schema:
            return json.loads(text)
        
        if return_full_response:
            return text, response
        
        return text

    # ---------------- abstract hooks ----------------

    @abstractmethod
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
        pass

    @abstractmethod
    def _stream(self, request: dict):
        pass

    @abstractmethod
    def _create(self, request: dict):
        pass

    @abstractmethod
    def _extract_text(self, response: dict):
        pass

    @abstractmethod
    def _extract_usage(self, response: dict) -> dict:
        pass

    def _increment_usage(self, response: dict, model_name: str):
        if not self.usage_tracker:
            return

        usage = self._extract_usage(response)
        total_tokens = usage['total_tokens']
        # adjust mapping to your DB schema
        self.usage_tracker.increment(model_name=model_name, tokens=total_tokens)
