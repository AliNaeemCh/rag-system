from app.models import RetrievedDocument
from app.infra.llm_engines.base import BaseLLMEngine
from evaluation.dataset.generation.config import EvalQuestionType

import logging
logger = logging.getLogger("evaluation.generation_eval")
logger.info("Loading file...")

class GenerationEvaluator:

    def __init__(self, llm_judge: BaseLLMEngine):
        self.llm_judge = llm_judge

    def _create_retrieved_context(self, retrieved_docs: list[RetrievedDocument]):
        retrieved_context = ""
        len_retrieved_docs = len(retrieved_docs)
        for i in range(len_retrieved_docs):
            retrieved_context += retrieved_docs[i].content
            if i < len_retrieved_docs-1:
                retrieved_context += "\n\n---\n\n"

        return retrieved_context

    def _create_user_prompt(self, questions: list[str], generated_answer: str, reference_answers: list[str] | None = None, retrieved_docs: list[RetrievedDocument] | None = None) -> str:
        if len(questions) == 1:
            user_prompt = f"""Question:\n\"{questions[0]}\""""
            if reference_answers:
                user_prompt += f"""\n\nReference answer:\n\"{reference_answers[0]}\""""
            elif retrieved_docs:
                retrieved_context = self._create_retrieved_context(retrieved_docs)
                user_prompt += f"""\n\nRetrieved context:\n{retrieved_context}"""
        else:
            question = ""
            ref_answer = ""
            for i, q in enumerate(questions):
                question += f"""{i+1}. {q}\n"""
                if reference_answers:
                    ref_answer += f"""{i+1}. {reference_answers[i]}\n"""

            user_prompt = "Question:\n" + question.strip()

            if ref_answer:
                user_prompt += "\n\n" + "Reference answer:\n" + ref_answer.strip()
            
            elif retrieved_docs:
                retrieved_context = self._create_retrieved_context(retrieved_docs)
                user_prompt += "\n\n" + "Retrieved context:\n" + retrieved_context
        
        user_prompt += f"""\n\n---\n\nGenerated answer:\n\"{generated_answer}\""""
        
        return user_prompt
    
    def evaluate_correctness(self,
                             question_type: EvalQuestionType,
                             questions: list[str],
                             reference_answers: list[str],
                             generated_answer: str,
                             system_prompt: str,
                             output_schema: dict,
                             temperature: float) -> dict[str, dict[str, str | bool]]:
        
        user_prompt = self._create_user_prompt(questions=questions, reference_answers=reference_answers, generated_answer=generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )

        total_points = sum(d["relevant"] for d in output["points"])

        if total_points > 0:
            correct_points = sum(d["correct"] and d['relevant'] for d in output["points"])
            correctness = correct_points/total_points

        else:
            correct_points = 0
            correctness = 0 if question_type != EvalQuestionType.OUT_OF_KNOWLEDGE else 1

        return {"correctness": correctness, "correct_points": correct_points, "total_points": total_points} | output

    def evaluate_faithfulness(self,
                             questions: list[str],
                             retrieved_docs: list[RetrievedDocument],
                             generated_answer: str,
                             system_prompt: str,
                             output_schema: dict,
                             temperature: float) -> dict[str, dict[str, str | bool]]:
        
        user_prompt = self._create_user_prompt(questions=questions, retrieved_docs=retrieved_docs, generated_answer=generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )

        total_points = sum(d["relevant"] for d in output["points"])

        if total_points > 0:
            correct_points = sum(d["correct"] and d['relevant'] for d in output["points"])
            faithfulness = correct_points/total_points

        else:
            correct_points = 0
            faithfulness = 1
        
        return {"faithfulness": faithfulness, "correct_points": correct_points, "total_points": total_points} | output

    def evaluate_relevance(self, questions: list[str], generated_answer: str, system_prompt: str, output_schema: dict, temperature: float) -> dict[str, dict[str, str | bool]]:
        user_prompt = self._create_user_prompt(questions=questions, generated_answer=generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )

        total_points = len(output['points'])

        if total_points > 0:
            relevant_points = sum(d["relevant"] for d in output["points"])
            relevance = relevant_points/total_points
        else:
            relevant_points = 0
            relevance = 1
        
        return {"relevance": relevance, "relevant_points": relevant_points, "total_points": total_points} | output