from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("rag.generator")
logger.info("Loading file...")


class Generator:
    def __init__(self, llm: BaseLLMEngine, system_prompt: str):
        """
        llm: LLM wrapper (OpenAI, HuggingFace, vLLM, etc.)
        system_prompt: System prompt for generation
        """
        self.llm = llm
        self.system_prompt = system_prompt

    async def generate(
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
        response = await self.llm.generate(
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
            async def gen():
                try:
                    async for chunk in response:
                        if chunk:
                            yield chunk
                except Exception:
                    logger.exception("Streaming error")
                    raise

            return gen()