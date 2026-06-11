from app.models import RetrievedDocument
from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("evaluation.generation_eval")
logger.info("Loading file...")

class GenerationEvaluator:

    def __init__(self, llm_judge: BaseLLMEngine):
        self.llm_judge = llm_judge

    def _create_user_prompt(self, question: str, reference_answer: str, generated_answer: str) -> str:
        user_prompt = f"""Question:\n\"{question}\"Reference answer:\n\"{reference_answer}\""""
        user_prompt += f"""\n\n---\n\nGenerated answer:\n\"{generated_answer}\""""
        
        return user_prompt
    
    def evaluate_correctness(self, question: str, reference_answer: str, generated_answer: str, system_prompt: str, output_schema: dict, temperature: float) -> dict[str, float | int]:
        user_prompt = self._create_user_prompt(question, reference_answer, generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )
        correctness = (output['total_points'] - output['incorrect_points']) / output['total_points'] if output['total_points'] > 0 else 1
        
        return {
            "correctness": correctness,
            "total_points": output['total_points'],
            "incorrect_points": output['incorrect_points']
        }
    
    def evaluate_completeness(self, question: str, reference_answer: str, generated_answer: str, system_prompt: str, output_schema: dict, temperature: float) -> dict[str, float | int]:
        user_prompt = self._create_user_prompt(question, reference_answer, generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )
        completeness = output['matched_points'] / output['total_points'] if output['total_points'] > 0 else 1
        
        return {
            "completeness": completeness,
            "total_points": output['total_points'],
            "matched_points": output['matched_points']
        }

    def evaluate_relevance(self, question: str, reference_answer: str, generated_answer: str, system_prompt: str, output_schema: dict, temperature: float) -> dict[str, float | int]:
        user_prompt = self._create_user_prompt(question, reference_answer, generated_answer)

        output = self.llm_judge.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=output_schema,
            temperature=temperature
        )
        relevance = (output['total_points'] - output['irrelevant_points']) / output['total_points'] if output['total_points'] > 0 else 1
        
        return {
            "relevance": relevance,
            "total_points": output['total_points'],
            "irrelevant_points": output['irrelevant_points']
        }