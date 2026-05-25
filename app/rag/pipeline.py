import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.chat_history import ChatHistory
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.core.config import settings

logger = logging.getLogger("app.rag.pipeline")

logger.info("Loading file...")

class RAGPipeline:
    def __init__(self, chat_history: ChatHistory, rewriter: MessageRewriter, retriever: Retriever, reranker: Reranker, generator: Generator):
        self.chat_history = chat_history
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def run(self, user_message: str, session_id: str, stream: bool = True, eval_mode: bool = False) -> str:
        try:
            logger.info(f"User message received")
            
            if not eval_mode:
                # 1. Get chat history
                chat_history = self.chat_history.get_recent(session_id)
                logger.debug(f"Chat history considered: {chat_history}")

                # 2. Store user message
                self.chat_history.add(session_id, role="user", content=user_message)
            
            else:
                chat_history = []

            # 3. Rewrite message (uses context)
            rewritten_message, keywords = self.rewriter.rewrite(message=user_message,
                                                                chat_history=chat_history,
                                                                keyword_exclusion_list=settings.REWRITER_KW_EXCLUDE_LIST)
            message_for_retriever = rewritten_message + '\n' + keywords

            logger.debug(f"Rewritten message for retriever: {message_for_retriever}")

            # 4. Retrieve documents
            docs = self.retriever.retrieve(message_for_retriever, ef_search=settings.PGVECTOR_HNSW_EF_SEARCH)

            # 5. Rerank documents
            top_ranked_docs = self.reranker.rerank(rewritten_message, docs)
            logger.debug(f"Top ranked docs are: {top_ranked_docs}")

            # 6. Build final context
            retrieved_context = "\n---\n".join([doc.content for doc in top_ranked_docs])

            # 7. Generate answer
            if stream:

                def gen():
                    full_response = ""
                    log_streaming_started = True
                    for chunk in self.generator.generate(
                        user_message=rewritten_message,
                        retrieved_context=retrieved_context,
                        history=chat_history,
                        stream=True
                    ):
                        if log_streaming_started:
                            logger.info("Streaming started")
                            log_streaming_started = False
                        full_response += chunk
                        yield chunk

                    logger.info("Streaming completed")

                    if not eval_mode:
                        # store assistant message AFTER streaming ends
                        self.chat_history.add(
                            session_id,
                            role="assistant",
                            content=full_response
                        )

                return gen()

            else:

                response = self.generator.generate(
                    user_message=rewritten_message,
                    retrieved_context=retrieved_context,
                    history=chat_history,
                    stream=False
                )

                logger.info("Answer generated")

                if not eval_mode:
                    # store assistant response
                    self.chat_history.add(
                        session_id,
                        role="assistant",
                        content=response
                    )

                return response

        except Exception:
            logger.exception("Error in RAG pipeline")
            raise