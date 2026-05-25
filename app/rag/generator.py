import logging
from app.core.config import settings
from app.infra.llm_engines.openai.engine import OpenAIEngine

logger = logging.getLogger("rag.generator")


class Generator:
    def __init__(self, llm: OpenAIEngine, system_prompt: str):
        """
        llm: LLM wrapper (OpenAI, HuggingFace, vLLM, etc.)
        system_prompt: System prompt for generation
        """
        self.llm = llm
        self.system_prompt = system_prompt

    def generate(
        self,
        user_message: str,
        retrieved_context: str,
        stream: bool,
        history: list[dict] | None = None,
        temperature: float = 0,
    ):
        """
        1. Build prompt/messages
        2. Call LLM
        3. Return response
        """

        logger.debug(f"Generation started | User message = {user_message}")

        system_prompt = self.system_prompt.format(retrieved_context=retrieved_context)
        response = self.llm.generate(
            user_prompt=user_message,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            stream=stream
        )

        if not stream:
            logger.info("Generation completed successfully")
            logger.debug(f"Response: {response}")
            return response

        else:
            def gen():
                full_text = ""

                try:
                    for chunk in response:
                        if chunk:
                            full_text += chunk
                            yield chunk

                except Exception:
                    logger.exception(f"Streaming error")
                    raise

            return gen()