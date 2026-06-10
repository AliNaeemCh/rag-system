from app.core.config import settings
from app.models import RetrievedDocument
from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("evaluation.retriever.evaluator")
logger.info("Loading file...")

class RetrievalEvaluator:

    def __init__(self, llm_judge: BaseLLMEngine, llm_output_schema: dict, system_prompt: str):
        self.llm_judge = llm_judge
        self.llm_output_schema = llm_output_schema
        self.system_prompt = system_prompt
    
    def _get_answerable_questions_via_llm(self, questions: list[str | None], reference_answers: list[str | None], context: str, temperature: float) -> list[int]:
        user_prompt = f"""Context:\n\n\"\"\"{context}\"\"\""""
        for i, (q, a) in enumerate(zip(questions, reference_answers)):
            if q and a:
                user_prompt += "\n\n---\n\n"
                user_prompt += f"Question {i}: {q}\nReference answer: {a}"

        answerable_qs = self.llm_judge.generate(system_prompt=self.system_prompt, user_prompt=user_prompt, schema=self.llm_output_schema, temperature=temperature)
        
        return answerable_qs
    
    def evaluate(self, questions: list[str], relevant_docs: list[dict], retrieved_docs: list[RetrievedDocument], temperature: float = 0) -> float:
        # Metrics
        mrr = 0
        recall_k = 0

        relevant_doc_ids = []
        relevant_doc_answers = []
        relevant_doc_answers_lower = []
        total_qs = len(questions)
        answerable_qs_indices = set()
        answerable_q_idx_to_method = {}
        llm_calls_count = 0
        
        for doc in relevant_docs:
            relevant_doc_ids.append(doc['chunk_id'])
            relevant_doc_answers.append(doc['answer'])
            relevant_doc_answers_lower.append(doc['answer'].lower())
        
        pos = 0
        while pos < len(retrieved_docs) and len(answerable_qs_indices) < total_qs:
            pos_r = 1/(pos+1)

            # Document id matching
            if retrieved_docs[pos].id in relevant_doc_ids:
                idx = relevant_doc_ids.index(doc.id)
                answerable_qs_indices.add(idx)
                if idx not in answerable_q_idx_to_method:
                    answerable_q_idx_to_method[idx] = "doc_id"
                mrr = max(pos_r, mrr)

            if len(answerable_qs_indices) < total_qs:
                # Answer finding
                found_answer_indices = []
                for idx, s in enumerate(relevant_doc_answers_lower):
                    if s in retrieved_docs[pos].content.lower():
                        found_answer_indices.append(idx)
                        if idx not in answerable_q_idx_to_method:
                            answerable_q_idx_to_method[idx] = "answer"
                if found_answer_indices:
                    answerable_qs_indices.update(found_answer_indices)
                    mrr = max(pos_r, mrr)

                if len(answerable_qs_indices) < total_qs:
                    # Evaluation by judge LLM
                    filtered_questions, filtered_answers = zip(*[
                        ((None, None) if i in answerable_qs_indices else (x, y))
                        for i, (x, y) in enumerate(zip(questions, relevant_doc_answers))
                    ])

                    if any(x is not None for x in filtered_questions):
                        answerable_qs_indices_by_llm = self._get_answerable_questions_via_llm(
                            questions=filtered_questions,
                            reference_answers=filtered_answers,
                            context=retrieved_docs[pos].content,
                            temperature=temperature
                        )
                        llm_calls_count += 1
                        if answerable_qs_indices_by_llm:
                            for idx in answerable_qs_indices_by_llm:
                                if idx not in answerable_q_idx_to_method:
                                    answerable_q_idx_to_method[idx] = "llm"

                            answerable_qs_indices.update(answerable_qs_indices_by_llm)
                            mrr = max(pos_r, mrr)
            pos += 1
        
        recall_k = len(answerable_qs_indices) / total_qs

        return {
            "mrr": mrr,
            "recall_k": recall_k,
            "answerable_qs_indices": answerable_qs_indices,
            "answerable_q_idx_to_method": answerable_q_idx_to_method,
            "llm_calls_count": llm_calls_count
        }

