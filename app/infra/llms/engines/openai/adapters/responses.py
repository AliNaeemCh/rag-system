from app.infra.llms.engines.openai.adapters.base import OpenAIBaseAdapter

class OpenAIResponsesAdapter(OpenAIBaseAdapter):

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

    def build_payload(self, request):
        payload = {
            "model": self.model,
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

    def stream(self, request):
        payload = self.build_payload(request)
        response = self.client.responses.stream(payload)

        def gen():
            with response.stream() as s:
                for event in s:
                    if event.type == "response.output_text.delta":
                        yield event.delta

        return gen()

    def create(self, request):
        payload = self.build_payload(request)
        return self.client.responses.create(**payload)

    def extract_text(self, response):
        return response.output_text