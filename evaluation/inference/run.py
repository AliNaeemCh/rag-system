from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.infra.dependencies import build_rag_pipeline
from app.core.utils import load_jsonl, extract_last_jsonl_object, reset_jsonl, write_jsonl
from evaluation.inference.config import EvalInferenceConfig
from app.rag.config import ResponseMode

import logging
logger = logging.getLogger("evaluation.inference.run")
logger.info("Loading file...")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def run_inference_pipeline(config: EvalInferenceConfig, rag_pipeline: RAGPipeline, dataset_path: Path, inference_path: Path):
    try:
        def create_config_object():
            config_obj = {
                "response_mode": config.response_mode,
                "embedding_model": settings.EMBEDDING_MODEL
            }
            if config.response_mode == ResponseMode.ADVANCED:
                config_obj.update(
                    {
                        "rewriter": {"model": rag_pipeline.rewriter.llm.model_name, "temperature": config.rewriter_temperature},
                        "reranker_model": settings.RERANKER_MODEL
                    }
                )
            elif config.response_mode == ResponseMode.BALANCED:
                config_obj.update(
                    {
                        "rewriter_model": {"model": rag_pipeline.rewriter.llm.model_name, "temperature": config.rewriter_temperature}
                    }
                )
            config_obj.update(
                {
                    "generator": {"model": rag_pipeline.generator.llm.model_name, "temperature": config.generator_temperature},
                    "hnsw_index": {
                        "m": settings.HNSW_M,
                        "ef_construction": settings.HNSW_EF_CONSTRUCTION,
                        "ef_search": settings.HNSW_EF_SEARCH
                    }
                }
            )
            return config_obj
        
        logger.info("Initializing pipeline...")

        completed_tasks = 0
        total_tasks = 0
        completed_example_ids = []
        last_obj = extract_last_jsonl_object(inference_path)
        if config.resume:
            for obj in load_jsonl(path=inference_path):
                if obj.get('example_id'):
                    completed_example_ids.append(obj['example_id'])
            if len(completed_example_ids) > 0:
                logger.info("Inference resumed")
            else:
                logger.info("Inference started")
        else:
            if last_obj.get('example_id'):
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously generated inference will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        break
                    else:
                        print("Invalid input. Try again!")
            reset_jsonl(inference_path)
            last_obj = {}
            logger.info("Inference started")
            
        if not last_obj:
            config_obj = create_config_object()
            write_jsonl(
                data=config_obj,
                output_path=inference_path
            )

        pbar = tqdm(total=100)

        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = []
            future_to_metadata = {}
            for obj in load_jsonl(dataset_path):
                if obj['example_id'] in completed_example_ids:
                    completed_tasks += 1
                    total_tasks += 1
                    continue
                question = " ".join(obj['questions'])
                future = executor.submit(rag_pipeline.run,
                                         user_message = question,
                                         session_id="1",
                                         stream=False,
                                         eval_mode=True,
                                         response_mode=config.response_mode,
                                         rewriter_temperature = config.rewriter_temperature,
                                         generator_temperature=config.generator_temperature)
                future_to_metadata[future] = {
                    "example_id": obj['example_id']
                }
                total_tasks += 1

            pbar.n = int((completed_tasks / total_tasks) * 100)
            pbar.refresh()

            for future in as_completed(futures):
                result = future.result()
                completed_tasks += 1

    
    except:
        logger.exception("Inference failed!")