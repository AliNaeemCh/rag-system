import logging
from app.prompts.rag import REWRITER_SYSTEM_PROMPT, REWRITER_SCHEMA
from app.prompts.prompt_utils import build_rewriter_user_prompt, build_openai_messages
from app.infra.llm_engines.openai import OpenAIEngine

logger = logging.getLogger("app.rag.rewriter")


class MessageRewriter:
    def __init__(self, llm: OpenAIEngine):
        self.llm = llm

    def rewrite(self, message: str, chat_history: list = []) -> str:
        """
        Rewrite user query into retrieval-optimized query.
        """

        # 1. Build user prompt
        user_prompt = build_rewriter_user_prompt(message, chat_history)

        # 2. Call LLM
        rewritten_message, keywords = self.llm.generate(user_prompt=user_prompt, system_prompt=REWRITER_SYSTEM_PROMPT, temperature=0, schema=REWRITER_SCHEMA)

        return rewritten_message, keywords