from app.models import RetrievedDocument
from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("evaluation.retrieval_eval")
logger.info("Loading file...")

class RetrievalEvaluator:

    def __init__(self, llm_judge: BaseLLMEngine, llm_output_schema: dict, system_prompt: str):
        self.llm_judge = llm_judge
        self.llm_output_schema = llm_output_schema
        self.system_prompt = system_prompt
    
    async def _get_answerable_questions_via_llm(self, questions: list[str | None], reference_answers: list[str | None], context: str, temperature: float) -> tuple[list[int], dict]:
        user_prompt = f"""Context:\n\n\"\"\"{context}\"\"\""""
        for i, (q, a) in enumerate(zip(questions, reference_answers)):
            if q and a:
                user_prompt += "\n\n---\n\n"
                user_prompt += f"Question {i}: {q}\nReference answer: {a}"

        output = await self.llm_judge.generate(system_prompt=self.system_prompt, user_prompt=user_prompt, schema=self.llm_output_schema, temperature=temperature)

        answerable_q_ids = [obj['id'] for obj in output["questions"] if obj["answerable"]]
        
        return answerable_q_ids, output
    
    async def evaluate(self, questions: list[str], relevant_doc_ids: list[int], reference_answers: list[str], retrieved_docs: list[RetrievedDocument], temperature: float = 0) -> float:
        # Metrics
        mrr = 0
        recall_k = 0

        reference_answers_lower = [answer.lower() for answer in reference_answers]
        total_qs = len(questions)
        answerable_qs_indices = set()
        answerable_q_idx_to_method = {}
        answerable_q_idx_to_doc_id = {}
        retrieved_doc_idx_to_llm_output = {}
        llm_calls_count = 0
        
        pos = 0
        while pos < len(retrieved_docs) and len(answerable_qs_indices) < total_qs:
            pos_r = 1/(pos+1)

            # Document id matching
            if retrieved_docs[pos].id in relevant_doc_ids:
                idx = relevant_doc_ids.index(retrieved_docs[pos].id)
                answerable_qs_indices.add(idx)
                if idx not in answerable_q_idx_to_method:
                    answerable_q_idx_to_method[idx] = "doc_id"
                    answerable_q_idx_to_doc_id[idx] = retrieved_docs[pos].id
                mrr = max(pos_r, mrr)

            if len(answerable_qs_indices) < total_qs:
                # Answer finding
                found_answer_indices = []
                for idx, s in enumerate(reference_answers_lower):
                    if s in retrieved_docs[pos].content.lower():
                        found_answer_indices.append(idx)
                        if idx not in answerable_q_idx_to_method:
                            answerable_q_idx_to_method[idx] = "answer"
                            answerable_q_idx_to_doc_id[idx] = retrieved_docs[pos].id
                if found_answer_indices:
                    answerable_qs_indices.update(found_answer_indices)
                    mrr = max(pos_r, mrr)

                if len(answerable_qs_indices) < total_qs:
                    # Evaluation by judge LLM
                    filtered_questions, filtered_answers = zip(*[
                        ((None, None) if i in answerable_qs_indices else (x, y))
                        for i, (x, y) in enumerate(zip(questions, reference_answers))
                    ])

                    if any(x is not None for x in filtered_questions):
                        answerable_qs_indices_by_llm, output = await self._get_answerable_questions_via_llm(
                            questions=filtered_questions,
                            reference_answers=filtered_answers,
                            context=retrieved_docs[pos].content,
                            temperature=temperature
                        )
                        retrieved_doc_idx_to_llm_output[pos] = output
                        llm_calls_count += 1
                        if answerable_qs_indices_by_llm:
                            for idx in answerable_qs_indices_by_llm:
                                if idx not in answerable_q_idx_to_method:
                                    answerable_q_idx_to_method[idx] = "llm"
                                    answerable_q_idx_to_doc_id[idx] = retrieved_docs[pos].id

                            answerable_qs_indices.update(answerable_qs_indices_by_llm)
                            mrr = max(pos_r, mrr)
            pos += 1
        
        total_answerable_qs = len(answerable_qs_indices)
        recall_k = total_answerable_qs / total_qs

        return {
            "mrr": mrr,
            "recall_k": recall_k,
            "total_answerable_qs": total_answerable_qs,
            "answerable_qs_indices": list(answerable_qs_indices),
            "answerable_q_idx_to_doc_id": answerable_q_idx_to_doc_id,
            "answerable_q_idx_to_method": answerable_q_idx_to_method,
            "llm_calls_count": llm_calls_count,
            "retrieved_doc_idx_to_llm_output": retrieved_doc_idx_to_llm_output
        }

