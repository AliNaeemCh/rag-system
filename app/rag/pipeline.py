from app.rag.message_rewriter import MessageRewriter
from app.rag.chat_history import ChatHistory
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.core.config import settings
from app.rag.config import ResponseMode
from app.infra.usage_tracking.tracker import usage_tracker
from app.rag.utils import reciprocal_rank_fusion
from app.models import RetrievedDocument

import logging
logger = logging.getLogger("app.rag.pipeline")
logger.info("Loading file...")

class RAGPipeline:
    def __init__(self, chat_history: ChatHistory, rewriter: MessageRewriter, retriever: Retriever, reranker: Reranker, generator: Generator):
        self.chat_history = chat_history
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def run(self, user_message: str, session_id: str, stream: bool = True, eval_mode: bool = False, response_mode: ResponseMode = ResponseMode.ADVANCED, rewriter_temperature: float = 0, generator_temperature: float = 0) -> str | dict[str, list[RetrievedDocument] | str]:
        try:

            logger.info(f"User message received: {user_message}")

            # Usage tracking
            if usage_tracker:
                model_names = [self.generator.llm.model_name]
                if response_mode != ResponseMode.FAST:
                    model_names.append(self.rewriter.llm.model_name)
                usage_exceeded = usage_tracker.usage_exceeded(model_names=model_names)
                if usage_exceeded:
                    raise Exception ("Usage limit exceeded!")
                
                logger.info("Usage status: under limit")

            if not eval_mode:
                # Get chat history
                chat_history = self.chat_history.get_recent(session_id)
                if not chat_history:
                    logger.info("New session request")
                logger.debug(f"Chat history considered: {chat_history}")
            
            else:
                chat_history = []

            rewritten_message = user_message

            # 3. Rewrite message (uses context)
            if response_mode != ResponseMode.FAST:
                rewritten_message = self.rewriter.rewrite(message=user_message, chat_history=chat_history, temperature=rewriter_temperature)

                logger.debug(f"Rewritten message for retriever: {rewritten_message}")

            # 4. Retrieve documents
            docs = self.retriever.retrieve(rewritten_message, ef_search=settings.HNSW_EF_SEARCH)

            logger.debug(f"Retrived docs are:\n{docs}")

            if docs['dense_docs'] and docs['sparse_docs']:
                # 5. Rank fusion
                docs = reciprocal_rank_fusion(
                    dense_docs=docs['dense_docs'],
                    sparse_docs=docs['sparse_docs'],
                    top_k=settings.FUSED_TOP_K
                )

                logger.debug(f"Reciprocal rank fused docs are:\n{docs}")
            
            else:
                if docs['dense_docs']:
                    docs = docs['dense_docs']
                elif docs['sparse_docs']:
                    docs = docs['sparse_docs']

            final_top_docs = docs[:settings.FINAL_TOP_K]

            if response_mode == ResponseMode.ADVANCED:
                # 6. Rerank documents
                final_top_docs = self.reranker.rerank(rewritten_message, docs)
            
            logger.debug(f"Top ranked docs are:\n{final_top_docs}")

            # 6. Build final context
            retrieved_context = "\n---\n".join([doc.content for doc in final_top_docs])

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
                        stream=True,
                        temperature=generator_temperature
                    ):
                        if log_streaming_started:
                            logger.info("Streaming started")
                            log_streaming_started = False
                        full_response += chunk
                        yield chunk

                    logger.info("Streaming completed")

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
                    stream=False,
                    temperature=generator_temperature
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
                
                else:
                    # Eval mode
                    return {
                        "final_top_docs": final_top_docs,
                        "response": response
                    }

        except Exception as e:
            if not eval_mode:
                raise
            logger.exception(f"Error in RAG pipeline.")