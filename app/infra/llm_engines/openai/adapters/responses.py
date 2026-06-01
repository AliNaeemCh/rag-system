from app.infra.llm_engines.openai.adapters.base import OpenAIBaseAdapter
from app.core.retry_policies import openai_retry

import logging
logger = logging.getLogger("app.infra.llms_engines.openai.adapters.responses")
logger.info("Loading file...")

from openai import OpenAI

class OpenAIResponsesAdapter(OpenAIBaseAdapter):

    def build_messages(self, system_prompt: str | None, user_prompt: str | None, image_urls: str | list[str] | None, history: list[dict] | None):
        history = history or []
        if image_urls is not None:
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            user_msg_obj = {'role': 'user', 'content': []}
            if user_prompt is not None:
                user_msg_obj['content'].append({'type': 'input_text', 'text': user_prompt})
            for url in image_urls:
                user_msg_obj['content'].append({'type': 'input_image', 'image_url': url})
        elif user_prompt is not None:
            user_msg_obj = {"role": "user", "content": user_prompt}
        else:
            raise Exception("Both image and user prompt can't be undefined!")
        system = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        user = [user_msg_obj]
        return (
            system +
            history +
            user
        )
    
    @openai_retry(logger)
    def stream(self, model_name: str, client: OpenAI, request: dict):
        payload = self._build_payload(model_name=model_name, request=request)
        response = client.responses.stream(**payload)
        state = {"final_response": None}
        def gen():
            with response as s:
                for event in s:
                    if event.type == "response.output_text.delta" and event.delta:
                        yield event.delta
                # available after stream completes
                state["final_response"] = s.get_final_response()

        return gen(), state

    @openai_retry(logger)
    def create(self, model_name: str, client: OpenAI, request: dict):
        payload = self._build_payload(model_name=model_name, request=request)
        return client.responses.create(**payload)

    def extract_text(self, response):
        output_text = response.output_text
        return output_text.strip() if output_text is not None else output_text

    def extract_usage(self, response: dict) -> dict:
        usage = getattr(response, "usage", None)

        if usage is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    
    def _build_payload(self, model_name: str, request: dict):
        payload = {
            "model": model_name,
            "input": request["messages"],
            "temperature": request["temperature"] if not request.get("reasoning") else None
        }

        if request.get("reasoning"):
            payload["reasoning"] = {"effort": request["reasoning"]}

        if request.get("schema"):
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "schema",
                    "schema": request["schema"],
                    "strict": True,
                }
            }

        return payload