import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.chat_history import ChatHistory
from app.rag.retriever import Retriever

logger = logging.getLogger("app.rag.pipeline")

class RAGPipeline:
    def __init__(self, chat_history: ChatHistory, rewriter: MessageRewriter, retriever: Retriever, reranker, generator):
        self.chat_history = chat_history
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def run(self, user_message: str, session_id: str):
        logger.info(f"User message received: {user_message}")

        # 1. Get chat history
        chat_history = self.chat_history.get_recent(session_id)

        # 2. Store user message
        self.chat_history.add(session_id, role="user", content=user_message)

        # 3. Rewrite query (uses context)
        rewritten_message, keywords = self.rewriter.rewrite(user_message, chat_history)
        logger.debug(f"Rewritten user message: {rewritten_message}")

        message_for_retriever = rewritten_message + '\n' + keywords

        # 4. Retrieve documents
        docs = self.retriever.retrieve(message_for_retriever)
        logger.info(f"Retrieved {len(docs)} docs")

        # 5. Rerank documents
        ranked_docs = self.reranker.rerank(rewritten_query, docs)

        # 6. Build final context
        context = [doc.page_content for doc in ranked_docs]

        # 7. Generate answer
        answer = self.generator.generate(
            query=query,
            context=context,
            chat_context=chat_context
        )

        logger.info("Answer generated")

        # 8. Store assistant response
        self.chat_history.add(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in ranked_docs]
        }