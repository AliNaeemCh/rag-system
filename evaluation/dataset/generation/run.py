from app.core.logger import setup_logging
setup_logging()

from evaluation.dataset.generation.generator import EvalDatasetGenerator
from app.core.utils import load_pickle, load_jsonl, extract_last_jsonl_object, reset_jsonl, write_jsonl
from app.core.config import settings
from evaluation.dataset.generation.config import EvalDatasetGeneratorConfig, EvalQuestionType
from app.prompts.eval_dataset_generator import FACTUAL_QS_GENERATOR_SYSTEM_PROMPT, INFERENCE_QS_GENERATOR_SYSTEM_PROMPT, QA_SCHEMA
from app.infra.usage_tracking.tracker import usage_tracker
from app.infra.dependencies import create_openai_client
from app.infra.llm_engines.openai.engine import OpenAIEngine
from ingestion.chunks_generation.config import ChunkingConfig

import logging
logger = logging.getLogger("evaluation.dataset.generation.run")
logger.info("Loading file...")

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def generate_chunk_ids(eval_dataset_generator: EvalDatasetGenerator, config: EvalDatasetGeneratorConfig) -> dict[EvalQuestionType, list[int] | list[list[int]]]:
    total_question_types = len(EvalQuestionType)
    questions_per_type = math.ceil(config.eval_set_size / total_question_types)

    config.eval_set_size = total_question_types * questions_per_type

    # - Question types: factual + inference
    base_ids = eval_dataset_generator.pick_random_ids(eval_set_size=questions_per_type * 2)

    factual_qs_chunk_ids = base_ids[:questions_per_type]
    inference_qs_chunk_ids = base_ids[questions_per_type:]

    eval_dataset_generator.extend_global_exclusion_ids(base_ids)

    # - Question type: multi_chunk
    def generate_multi_chunk_ids():
        qs_per_num_chunks = math.floor(
            questions_per_type / (config.multi_chunk_max_qs - 1)
        )

        multi_chunk_qs_chunk_ids = []
        qs_counter = 0

        for num_chunks in range(2, config.multi_chunk_max_qs + 1):
            chunk_lists = eval_dataset_generator.pick_multiple_id_lists(
                eval_set_size=qs_per_num_chunks,
                ids_count=num_chunks
            )
            qs_counter += len(chunk_lists)
            multi_chunk_qs_chunk_ids.extend(chunk_lists)

            eval_dataset_generator.extend_global_exclusion_ids(chunk_lists)

        # fill remainder with 2-chunk questions
        if qs_counter < questions_per_type:
            rem = questions_per_type - qs_counter
            chunk_lists = eval_dataset_generator.pick_multiple_id_lists(
                eval_set_size=rem,
                ids_count=2
            )
            multi_chunk_qs_chunk_ids[:0] = chunk_lists

            eval_dataset_generator.extend_global_exclusion_ids(chunk_lists)
        
        return multi_chunk_qs_chunk_ids
        
    multi_chunk_qs_chunk_ids = generate_multi_chunk_ids()

    return {
        EvalQuestionType.FACTUAL: factual_qs_chunk_ids,
        EvalQuestionType.INFERENCE: inference_qs_chunk_ids,
        EvalQuestionType.MULTI_CHUNK: multi_chunk_qs_chunk_ids
    }

def run_pipeline(eval_dataset_generator: EvalDatasetGenerator, config: EvalDatasetGeneratorConfig, dataset_path: Path):
    try:
        logger.info("Initializing pipeline...")

        completed_chunk_ids = []
        total_tasks = config.eval_set_size
        completed_tasks = 0
        example_id = 1

        if config.resume:
            for obj in load_jsonl(path=dataset_path):
                if obj:
                    completed_chunk_ids.extend(obj['chunk_ids'])
                    completed_tasks += 1
                    if obj['example_id'] > example_id:
                        example_id = obj['example_id']

            if completed_tasks > 0:
                example_id += 1
                logger.info("Eval dataset generation resumed")
            else:
                logger.info("Eval dataset generation started")
        else:
            last_obj = extract_last_jsonl_object(dataset_path)
            if last_obj:
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously generated dataset will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        break
                    else:
                        print("Invalid input. Try again!")
            reset_jsonl(dataset_path)
            logger.info("Eval dataset generation started")

        pbar = tqdm(total=100)
        pbar.n = int((completed_tasks / total_tasks) * 100)
        pbar.refresh()

        # Chunk ids generation
        question_type_to_chunk_ids = generate_chunk_ids(eval_dataset_generator, config)

        # Questions generation
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = []
            future_to_metadata = {}
            for q_type, chunk_ids in question_type_to_chunk_ids.items():
                system_prompt = FACTUAL_QS_GENERATOR_SYSTEM_PROMPT
                if q_type == EvalQuestionType.INFERENCE:
                    system_prompt = INFERENCE_QS_GENERATOR_SYSTEM_PROMPT
                for chunk_id in chunk_ids:
                    if isinstance(chunk_id, int):
                        chunk_id = [chunk_id]
                    if all(x in completed_chunk_ids for x in chunk_id):
                        continue
                    # 1. Submit all
                    future = executor.submit(eval_dataset_generator.create_question, chunk_id, system_prompt, QA_SCHEMA, config.llm_temperature)
                    futures.append(future)
                    future_to_metadata[future] = {
                        "chunk_ids": chunk_id,
                        "question_type": q_type.value
                    }
            # 2. Process as they complete
            for future in as_completed(futures):
                result = future.result()
                completed_tasks += 1
                write_jsonl(
                    data={
                        "example_id": example_id,
                        "question_type": future_to_metadata[future]['question_type'],
                        "chunk_ids": future_to_metadata[future]['chunk_ids'],
                        "questions": result['questions'],
                        "answers": result['answers']
                    },
                    output_path=dataset_path
                )
                example_id += 1
                pbar.n = int((completed_tasks / total_tasks) * 100)
                pbar.refresh()
        logger.exception("Eval dataset generation completed")
    except Exception:
        logger.exception("Eval dataset generation failed!")

chunks_index = load_pickle(file_path=settings.PROCESSED_DATA_DIR / "chunks_jsonl_index.pkl")
chunks_path = settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl"
dataset_path = settings.EVAL_DATASET_DIR / "eval_dataset.jsonl"
config = EvalDatasetGeneratorConfig(resume=True)
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY)
eval_dataset_generator_llm = OpenAIEngine(model_name=settings.EVAL_DATASET_GENERATOR_LLM, client = openai_client, usage_tracker=usage_tracker)
eval_dataset_generator = EvalDatasetGenerator(chunks_index=chunks_index, chunks_path=chunks_path, llm=eval_dataset_generator_llm, min_chunk_tokens=ChunkingConfig().chunk_size // 2, seed=config.seed)

run_pipeline(
    eval_dataset_generator=eval_dataset_generator,
    config=config,
    dataset_path=dataset_path
)