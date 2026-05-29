import logging
from app.rag.message_rewriter import MessageRewriter
from app.rag.chat_history import ChatHistory
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.core.config import settings
from app.rag.config import ResponseMode

logger = logging.getLogger("app.rag.pipeline")

logger.info("Loading file...")

class RAGPipeline:
    def __init__(self, chat_history: ChatHistory, rewriter: MessageRewriter, retriever: Retriever, reranker: Reranker, generator: Generator):
        self.chat_history = chat_history
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def run(self, user_message: str, session_id: str, stream: bool = True, eval_mode: bool = False, response_mode: ResponseMode = ResponseMode.ADVANCED) -> str:
        try:
            logger.info(f"User message received")
            
            if not eval_mode:
                # Get chat history
                chat_history = self.chat_history.get_recent(session_id)
                if not chat_history:
                    logger.info("New session request")
                logger.debug(f"Chat history considered: {chat_history}")
            
            else:
                chat_history = []

            message_for_retriever = rewritten_message = user_message

            # 3. Rewrite message (uses context)
            if response_mode != ResponseMode.FAST:
                rewritten_message, keywords = self.rewriter.rewrite(message=user_message,
                                                                    chat_history=chat_history,
                                                                    keyword_exclusion_list=settings.REWRITER_KW_EXCLUDE_LIST)
                
                message_for_retriever = rewritten_message + '\n' + keywords

                logger.debug(f"Rewritten message for retriever: {message_for_retriever}")

            # 4. Retrieve documents
            docs = self.retriever.retrieve(message_for_retriever, ef_search=settings.PGVECTOR_HNSW_EF_SEARCH)

            logger.debug(f"Retrived docs are:\n{docs}")

            top_ranked_docs = docs[:self.reranker.top_k]

            if response_mode == ResponseMode.ADVANCED:
                # 5. Rerank documents
                top_ranked_docs = self.reranker.rerank(rewritten_message, docs)
                logger.debug(f"Top ranked docs are:\n{top_ranked_docs}")

            # 6. Build final context
            retrieved_context = "\n---\n".join([doc.content for doc in top_ranked_docs])

            # 7. Generate answer
            logger.info("Calling generator LLM")

            if stream:

                def gen():
                    full_response = ""
                    log_streaming_started = True
                    for chunk in self.generator.generate(
                        user_message=user_message,
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
                        # store user and assistant  messages
                        self.chat_history.add(session_id, role="user", content=user_message)
                        self.chat_history.add(
                            session_id,
                            role="assistant",
                            content=full_response
                        )

                return gen()

            else:
                response = self.generator.generate(
                    user_message=user_message,
                    retrieved_context=retrieved_context,
                    history=chat_history,
                    stream=False
                )

                logger.info("Answer generated")

                if not eval_mode:
                    # store user and assistant messages
                    self.chat_history.add(session_id, role="user", content=user_message)
                    self.chat_history.add(
                        session_id,
                        role="assistant",
                        content=response
                    )

                return response

        except Exception:
            if not eval_mode:
                raise
            logger.exception("Error in RAG pipeline")