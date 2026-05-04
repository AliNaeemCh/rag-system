from app.infra.llms.engines.openai.adapters.base import OpenAIBaseAdapter
import logging
from app.core.retry_policies import openai_retry

logger = logging.getLogger("app.infra.llms.engines.openai.adapters.chat_completions")

class OpenAIChatCompletionsAdapter(OpenAIBaseAdapter):

    def __init__(self, client, model):
        self.client = client
        self.model = model
    
    def build_messages(self, system_prompt: str | None, user_prompt: str | None, image_urls: str | list[str] | None, history: list[dict] | None):
        history = history or []
        if image_urls is not None:
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            user_msg_obj = {'role': 'user', 'content': []}
            if user_prompt is not None:
                user_msg_obj['content'].append({'type': 'text', 'text': user_prompt})
            for url in image_urls:
                user_msg_obj['content'].append({'type': 'image_url', 'image_url': {'url': url}})
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
    
    def build_payload(self, request):
        payload = {
            "model": self.model,
            "messages": request["messages"],
            "temperature": request.get("temperature") if not request.get('reasoning') else None,
        }

        if request.get('reasoning'):
            payload['reasoning_effort'] = request.get('reasoning')

        if request.get("schema"):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema",
                    "schema": request["schema"],
                    "strict": True,
                },
            }

        return payload

    @openai_retry(logger)
    def stream(self, request):
        payload = self.build_payload(request)
        payload["stream"] = True

        response = self.client.chat.completions.create(**payload)

        def gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return gen()
    
    @openai_retry(logger)
    def create(self, request):
        payload = self.build_payload(request)
        return self.client.chat.completions.create(**payload)

    def extract_text(self, response):
        output_text = response.choices[0].message.content
        return output_text.strip() if output_text is not None else output_text