import json
import time
import logging
from app.core.retry_policies import openai_retry
from app.prompts.prompt_utils import build_openai_messages

logger = logging.getLogger("infra.llm_engines.openai")


class OpenAIEngine:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    @openai_retry(logger)
    def call_llm(self, payload: dict):
        start = time.time()

        logger.debug(f"LLM call started | model={self.model}")

        result = self.client.responses.stream(payload)

        logger.info(
            f"LLM call success | model={self.model} | latency={time.time() - start:.2f}s"
        )

        return result

    # ---------------- HIGH LEVEL GENERATE ----------------
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        history: list[dict] = [],
        *,
        stream=False,
        schema=None,
        reasoning=None,
        temperature=1,
    ):
        """
        Handles:
        - prompt orchestration
        - schema formatting
        - streaming vs normal mode
        """

        input_messages = build_openai_messages(system_prompt, user_prompt, history)

        def build_payload():
            payload = {
                "model": self.model,
                "input": input_messages,
                "temperature": temperature if not reasoning else None,
                "reasoning": {"effort": reasoning} if reasoning else None,
            }

            if schema:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "schema",
                        "schema": schema,
                        "strict": True
                    }
                }

            return payload

        payload = build_payload()

        # ---------------- STREAMING ----------------
        if stream:
            def stream_gen():
                response = self.call_llm(payload)

                with response.stream() as stream_obj:
                    for event in stream_obj:
                        if event.type == "response.output_text.delta":
                            yield event.delta

            return stream_gen()

        # ---------------- NORMAL MODE ----------------
        response = self.call_llm(payload)

        text = "".join(
            event.delta for event in response.output
            if hasattr(event, "delta")
        ).strip()

        if schema:
            return json.loads(text)

        return text