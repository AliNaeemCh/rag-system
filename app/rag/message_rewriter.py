from app.prompts.prompt_utils import build_rewriter_user_prompt
from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("app.rag.rewriter")
logger.info("Loading file...")


class MessageRewriter:
    def __init__(self, llm: BaseLLMEngine, system_prompt: str, output_schema: dict):
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    def rewrite(self, message: str, chat_history: list = [], keyword_exclusion_list: list[str] = [], temperature: float = 0) -> str:
        """
        Rewrite user query into retrieval-optimized query.
        """
        logger.info("Message rewriting started")

        # 1. Build user prompt
        user_prompt = build_rewriter_user_prompt(message, chat_history)

        # 2. Call LLM
        response = self.llm.generate(user_prompt=user_prompt, system_prompt=self.system_prompt, temperature=temperature, schema=self.output_schema)

        # 3. Extract required fields
        rewritten_message = response["rewritten_message"]

        logger.info("Message rewriting completed")

        return rewritten_message