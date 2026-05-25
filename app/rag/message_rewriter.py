import logging
from app.prompts.rag import REWRITER_SYSTEM_PROMPT, REWRITER_SCHEMA
from app.prompts.prompt_utils import build_rewriter_user_prompt
from app.infra.llm_engines.openai.engine import OpenAIEngine

logger = logging.getLogger("app.rag.rewriter")


class MessageRewriter:
    def __init__(self, llm: OpenAIEngine):
        self.llm = llm

    def rewrite(self, message: str, chat_history: list = [], keyword_exclusion_list: list[str] = []) -> str:
        """
        Rewrite user query into retrieval-optimized query.
        """
        logger.info("Message rewriting started")

        # 1. Build user prompt
        user_prompt = build_rewriter_user_prompt(message, chat_history)

        # 2. Call LLM
        response = self.llm.generate(user_prompt=user_prompt, system_prompt=REWRITER_SYSTEM_PROMPT, temperature=0, schema=REWRITER_SCHEMA)
        
        # 3. Extract required fields
        rewritten_message = response["rewritten_message"]
        rewritten_message_lower = response["rewritten_message"].lower()
        keywords = response["keywords"]
        exclusion_set = {kw.lower() for kw in keyword_exclusion_list}
        keywords = [kw for kw in keywords if (kw_lower := kw.lower()) not in exclusion_set and kw_lower not in rewritten_message_lower]
        keywords_str = ", ".join(keywords)

        logger.info("Message rewriting completed")

        return rewritten_message, keywords_str