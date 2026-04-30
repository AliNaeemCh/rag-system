import time
import logging
from app.core.retry_policies import gemini_retry
from app.prompts.prompt_utils import build_gemini_messages


logger = logging.getLogger("app.infra.llm_engines.gemini")


class GeminiEngine:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    # ---------------- LOW LEVEL CALL ----------------
    @gemini_retry(logger)
    def call_llm(self, payload: dict):
        start = time.time()

        logger.debug(f"Gemini call started | model={self.model}")

        result = self.client.models.generate_content_stream(
            model=self.model,
            **payload
        )

        logger.info(
            f"Gemini call success | model={self.model} | latency={time.time() - start:.2f}s"
        )

        return result

    # ---------------- HIGH LEVEL GENERATE ----------------
    def generate(
        self,
        *,
        system_instruction: str,
        history: list = [],
        user_prompt: str | None = None,
        image=None,
        stream: bool = False,
        temperature: float | None = None,
    ):
        """
        High-level API (same style as OpenAI engine)
        """

        payload = build_gemini_messages(
            system_instruction=system_instruction,
            history=history,
            user_prompt=user_prompt,
            image=image
        )

        # inject runtime config values
        if temperature is not None:
            payload["config"].temperature = temperature

        # ---------------- STREAMING ----------------
        if stream:
            def stream_gen():
                response = self.call_llm(payload)

                for chunk in response:
                    if hasattr(chunk, "text"):
                        yield chunk.text

            return stream_gen()

        # ---------------- NORMAL MODE ----------------

        response = self.call_llm(payload)
        text = "".join(
            chunk.text for chunk in response
            if hasattr(chunk, "text")
        ).strip()

        return text