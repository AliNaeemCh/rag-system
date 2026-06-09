from app.core.config import settings
from app.models import RetrievedDocument

import logging
logger = logging.getLogger("evaluation.retriever.evaluator")
logger.info("Loading file...")

class RetrieverEvaluator:

    def __init__(self):
        pass

    def calculate_mrr(self, relevant_docs: list[dict], retrieved_docs: list[RetrievedDocument]) -> float:
        relevant_doc_ids = []
        relevant_doc_answers = []
        for doc in relevant_docs:
            relevant_doc_ids.append(doc['chunk_id'])
            answer = doc.get("answer")
            answer = answer.lower() if answer is not None else None
            relevant_doc_answers.append(answer)
        pos = 1
        for doc in retrieved_docs:
            if doc.id in relevant_doc_ids or \
            any(s in doc.content.lower() for s in relevant_doc_answers if s is not None):
                return 1/pos
            pos += 1
        return 0