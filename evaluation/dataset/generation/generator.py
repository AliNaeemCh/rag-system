from app.core.utils import get_jsonl_object
from app.infra.llm_engines.base import BaseLLMEngine

import logging
logger = logging.getLogger("evaluation.dataset.generation.generator")
logger.info("Loading file...")

import random
from pathlib import Path

class EvalDatasetGenerator:

    def __init__(self, chunks_index: list[int], chunks_path: Path, llm: BaseLLMEngine, seed: int = 11, min_chunk_tokens: int = 0):
        self.chunks_index = chunks_index
        self.chunks_path = chunks_path
        self.llm = llm
        self.total_chunks = len(chunks_index)
        self.seed = seed
        self.rng = random.Random(seed)
        self.exclusion_ids: list[int] = []
        self.min_chunk_tokens = min_chunk_tokens
    
    # Chunks selection for dataset

    def _available_ids(self) -> list[int]:
        """Compute ids not in exclusion set."""
        return list(set(range(1, self.total_chunks+1)) - set(self.exclusion_ids))

    def pick_random_ids(
        self,
        eval_set_size: int
    ) -> list[int]:
        """
        Randomly pick unique ids excluding provided ones + internal exclusion list.
        """
        available_ids = self._available_ids()

        if len(available_ids) < eval_set_size:
            raise ValueError("Not enough available ids.")
        
        selected = []

        while len(selected) < eval_set_size:
            id = self.rng.choice(available_ids)
            chunk_total_tokens = get_jsonl_object(self.chunks_path, index=self.chunks_index, line_number=id)['metadata']['total_tokens']
            if chunk_total_tokens >= self.min_chunk_tokens:
                selected.append(id)
            available_ids.remove(id)

        return selected

    def pick_multiple_id_lists(
        self,
        eval_set_size: int,
        ids_count: int
    ) -> list[list[int]]:
        """
        Generate multiple lists of unique ids.
        Each list contains `ids_count` unique items.
        """
        available_ids = self._available_ids()
        if len(available_ids) < ids_count * eval_set_size:
            raise ValueError("Not enough available ids.")

        results = []

        for _ in range(eval_set_size):

            selected = []

            while len(selected) < ids_count:
                id = self.rng.choice(available_ids)
                chunk_total_tokens = get_jsonl_object(self.chunks_path, index=self.chunks_index, line_number=id)['metadata']['total_tokens']
                if chunk_total_tokens >= self.min_chunk_tokens:
                    selected.append(id)
                available_ids.remove(id)

            results.append(sorted(selected))

        return results

    def extend_global_exclusion_ids(self, ids: list[int] | int) -> None:
        """
        Flatten input and add to internal exclusion list.
        """
        flat = []
        for item in ids:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        for item in flat:
            if item not in self.exclusion_ids:
                self.exclusion_ids.append(item)
    
    def reset_global_exclusion_ids(self) -> None:
        self.exclusion_ids.clear()
    
    # Questions Creation

    def create_question(self, chunk_ids: list[int], system_prompt: str, output_schema: dict, temperature: float = 1) -> dict[str, list[str]]:
        questions = []
        answers = []
        for chunk_id in chunk_ids:
            chunk_text = get_jsonl_object(self.chunks_path, self.chunks_index, line_number=chunk_id)['content']
            user_prompt = f"Chunk:\n\"{chunk_text}\""
            response = self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=temperature
            )
            questions.append(response['question'])
            if response.get('answer'):
                answers.append(response['answer'])

        return {
            "questions": questions,
            "answers": answers
        }