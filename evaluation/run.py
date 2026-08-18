from app.core.logger import setup_logging
setup_logging()

from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.infra.dependencies import build_rag_pipeline, create_openai_client
from app.core.utils import load_jsonl, extract_last_jsonl_object, reset_jsonl, write_jsonl, write_json
from app.infra.usage_tracking.tracker import usage_tracker
from evaluation.config import EvalConfig
from app.rag.config import ResponseMode
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.generation_eval import GenerationEvaluator
from evaluation.dataset.generation.config import EvalQuestionType
from app.infra.llm_engines.openai.engine import OpenAIEngine
from app.prompts.retrieval_evaluator import ANSWERABLE_QS_SYSTEM_PROMPT, ANSWERABLE_QS_SCHEMA
from app.prompts.generation_evaluator import REFERENCE_COVERAGE_EVAL_SYSTEM_PROMPT, REFERENCE_COVERAGE_EVAL_SCHEMA, \
                                            FAITHFULNESS_EVAL_SYSTEM_PROMPT, FAITHFULNESS_EVAL_SCHEMA
from app.prompts.eval_dataset_generator import FACTUAL_QS_GENERATOR_SYSTEM_PROMPT, INFERENCE_QS_GENERATOR_SYSTEM_PROMPT, QA_SCHEMA
from evaluation.dataset.generation.config import EvalQuestionType
from evaluation.dataset.generation.config import EvalDatasetGeneratorConfig
from evaluation.utils import update_eval_results, aggregate_eval_results, get_eval_results_obj
from ingestion.chunks_generation.config import ChunkingConfig

import logging
logger = logging.getLogger("evaluation.run")
logger.info("Loading file...")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

def process_example(rag_pipeline: RAGPipeline,
                    retrieval_eval: RetrievalEvaluator,
                    generation_eval: GenerationEvaluator,
                    question_type: EvalQuestionType,
                    questions: list[str],
                    relevant_doc_ids: list[int],
                    reference_answers: list[str],
                    response_mode: ResponseMode,
                    rewriter_temperature: float = 0,
                    generator_temperature: float = 0,
                    retrieval_eval_temperature: float = 0,
                    generation_eval_temperature: float = 0) -> dict:
    
    question = " ".join(questions)

    # Inference
    output = rag_pipeline.run(
        user_message=question,
        session_id="1",
        stream=False,
        eval_mode=True,
        response_mode=response_mode,
        rewriter_temperature=rewriter_temperature,
        generator_temperature=generator_temperature
        )
    
    retrieved_docs = output['final_top_docs']
    generated_answer = output['response']

    # Retrieval eval
    retrieval_results = retrieval_eval.evaluate(
        questions=questions,
        relevant_doc_ids=relevant_doc_ids,
        reference_answers=reference_answers,
        retrieved_docs=retrieved_docs,
        temperature=retrieval_eval_temperature
    )

    # Generation eval
    reference_coverage_result = generation_eval.evaluate_reference_coverage(
        questions=questions,
        reference_answers=reference_answers,
        generated_answer=generated_answer,
        system_prompt=REFERENCE_COVERAGE_EVAL_SYSTEM_PROMPT,
        output_schema=REFERENCE_COVERAGE_EVAL_SCHEMA,
        temperature=generation_eval_temperature
    )

    faithfulness_result = generation_eval.evaluate_faithfulness(
        questions=questions,
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
        system_prompt=FAITHFULNESS_EVAL_SYSTEM_PROMPT,
        output_schema=FAITHFULNESS_EVAL_SCHEMA,
        temperature=generation_eval_temperature
    )

    return {
        "question_type": question_type.value,
        "total_questions": len(questions),
        "questions": questions,
        "reference_answers": reference_answers,
        "generated_answer": generated_answer,
        "relevant_doc_ids": relevant_doc_ids,
        "retrieved_doc_ids": [doc.id for doc in retrieved_docs],
        "retrieval_results": retrieval_results,
        "generation_results": {
            "reference_coverage": reference_coverage_result,
            "faithfulness": faithfulness_result
        }
    }

def run_pipeline(config: EvalConfig,
                 rag_pipeline: RAGPipeline,
                 retrieval_eval: RetrievalEvaluator,
                 generation_eval: GenerationEvaluator,
                 dataset_path: Path,
                 config_path: Path,
                 raw_result_path: Path,
                 final_result_path: Path,):
    try:
        def create_config_object():

            dataset_gen_config = EvalDatasetGeneratorConfig()
            chunking_config = ChunkingConfig()

            config_obj = {
                "response_mode": config.response_mode.value,
                "embedding_model": settings.EMBEDDING_MODEL
            }
            if config.response_mode == ResponseMode.ADVANCED:
                config_obj.update(
                    {
                        "rewriter": {"model": rag_pipeline.rewriter.llm.model_name, "temperature": config.rewriter_temperature, "system_prompt": rag_pipeline.rewriter.system_prompt, "schema": rag_pipeline.rewriter.output_schema},
                        "reranker_model": settings.RERANKER_MODEL
                    }
                )
            elif config.response_mode == ResponseMode.BALANCED:
                config_obj.update(
                    {
                        "rewriter_model": {"model": rag_pipeline.rewriter.llm.model_name, "temperature": config.rewriter_temperature, "system_prompt": rag_pipeline.rewriter.system_prompt, "schema": rag_pipeline.rewriter.output_schema}
                    }
                )
            config_obj.update(
                {
                    "generator": {"model": rag_pipeline.generator.llm.model_name, "temperature": config.generator_temperature, "system_prompt": rag_pipeline.generator.system_prompt},
                    "hnsw_index": {
                        "m": settings.HNSW_M,
                        "ef_construction": settings.HNSW_EF_CONSTRUCTION,
                        "ef_search": settings.HNSW_EF_SEARCH
                    },
                    "chunking": {
                        "chunk_size": chunking_config.chunk_size,
                        "chunk_overlap_pct": chunking_config.chunk_overlap_pct,
                        "cross_section_overlap": chunking_config.cross_section_overlap,
                        "overlap_granularity": chunking_config.overlap_granularity.value
                    },
                    "top_ks": {
                        "dense_retrieval": settings.DENSE_TOP_K,
                        "sparse_retrieval": settings.SPARSE_TOP_K,
                        "fused": settings.FUSED_TOP_K,
                        "final": settings.FINAL_TOP_K
                    },
                    "eval_llm_judge": {
                        "model": settings.LLM_JUDGE_MODEL,
                        "retrieval": {
                            "model": settings.LLM_JUDGE_MODEL,
                            "temperature": config.retrieval_eval_temperature,
                            "system_prompt": ANSWERABLE_QS_SYSTEM_PROMPT,
                            "schema": ANSWERABLE_QS_SCHEMA
                        },
                        "generation": {
                            "model": settings.LLM_JUDGE_MODEL,
                            "temperature": config.generation_eval_temperature,
                            "recall_coverage": {
                                "system_prompt": REFERENCE_COVERAGE_EVAL_SYSTEM_PROMPT,
                                "schema": REFERENCE_COVERAGE_EVAL_SCHEMA
                            },
                            "faithfulness": {
                                "system_prompt": FAITHFULNESS_EVAL_SYSTEM_PROMPT,
                                "schema": FAITHFULNESS_EVAL_SCHEMA
                        },
                        "retrieval_eval_temperature": config.retrieval_eval_temperature,
                        "generation_eval_temperature": config.generation_eval_temperature
                    }},
                    "dataset": {
                        "generation_llm": {
                            "model": settings.EVAL_DATASET_GENERATOR_LLM,
                            "temperature": dataset_gen_config.llm_temperature
                        },
                        "total_question_types": len(EvalQuestionType),
                        "question_types": [e.value for e in EvalQuestionType],
                        "system_prompts": {
                            f"{EvalQuestionType.FACTUAL.value}, {EvalQuestionType.MULTI_CHUNK.value}": FACTUAL_QS_GENERATOR_SYSTEM_PROMPT,
                            EvalQuestionType.INFERENCE.value: INFERENCE_QS_GENERATOR_SYSTEM_PROMPT
                        },
                        "schema": QA_SCHEMA
                    }
                }
            )
            return config_obj
        
        logger.info("Initializing pipeline...")

        completed_example_ids = []
        eval_results = get_eval_results_obj()
        last_obj = extract_last_jsonl_object(raw_result_path)

        if config.resume:
            for obj in load_jsonl(path=raw_result_path):
                update_eval_results(eval_results, new_result=obj)
                completed_example_ids.append(obj['example_id'])
            if len(completed_example_ids) > 0:
                logger.info("Evaluation resumed")
            else:
                logger.info("Evaluation started")
        else:
            if last_obj.get('example_id'):
                while True:
                    user_in = input(f"\033[93mWarning:\033[0m Previously generated evaluation results will be deleted. Type 'confirm' to proceed: ")
                    if user_in == "confirm":
                        break
                    else:
                        print("Invalid input. Try again!")
            reset_jsonl(raw_result_path)
            last_obj = {}
            logger.info("Evaluation started")
        
        logging.getLogger().setLevel(logging.WARNING)   # Silencing logs
            
        if not last_obj:
            config_obj = create_config_object()
            write_json(
                data=config_obj,
                output_path=config_path
            )

        pbar = tqdm(total=100)

        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = set()
            future_to_metadata = {}
            initial_progress_refresh = False
            for obj, progress in load_jsonl(dataset_path, return_progress=True):
                if obj['example_id'] in completed_example_ids:
                    initial_progress_refresh = True
                    continue

                if initial_progress_refresh:
                    pbar.n = int(progress * 100)
                    pbar.refresh()
                    initial_progress_refresh = False

                question_type = EvalQuestionType(obj['question_type'])
                relevant_doc_ids = []
                reference_answers = []

                for i, id in enumerate(obj['chunk_ids']):
                    relevant_doc_ids.append(id)
                    reference_answers.append(obj['answers'][i])
                
                if len(futures) >= config.num_workers:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        write_jsonl(
                            data= {"example_id": future_to_metadata[future]["example_id"]} | result,
                            output_path=raw_result_path
                        )
                        update_eval_results(eval_results, new_result=result)
                        del future_to_metadata[future]
                        futures.remove(future)

                    pbar.n = int(progress * 100)
                    pbar.refresh()

                future = executor.submit(process_example,
                                         rag_pipeline=rag_pipeline,
                                         retrieval_eval=retrieval_eval,
                                         generation_eval=generation_eval,
                                         question_type=question_type,
                                         questions=obj['questions'],
                                         relevant_doc_ids=relevant_doc_ids,
                                         reference_answers=reference_answers,
                                         response_mode=config.response_mode,
                                         rewriter_temperature=config.rewriter_temperature,
                                         generator_temperature=config.generator_temperature,
                                         retrieval_eval_temperature=config.retrieval_eval_temperature,
                                         generation_eval_temperature=config.generation_eval_temperature
                                         )
                futures.add(future)
                future_to_metadata[future] = {
                    "example_id": obj['example_id']
                }

            # Drain remaining futures
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for future in done:
                    result = future.result()
                    write_jsonl(
                        data= {"example_id": future_to_metadata[future]["example_id"]} | result,
                        output_path=raw_result_path
                    )
                    update_eval_results(eval_results, new_result=result)
                    del future_to_metadata[future]
                    futures.remove(future)

                pbar.n = int(progress * 100)
                pbar.refresh()

        logging.getLogger().setLevel(logging.INFO)  # Restoring logs

        aggregate_eval_results(eval_results)
        write_json(eval_results, output_path=final_result_path)

        logger.info("Evaluation completed")

    except:
        logger.exception("Evaluation failed!")

config = EvalConfig(resume=True)
rag_pipeline = build_rag_pipeline()
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY)
llm_judge = OpenAIEngine(model_name=settings.LLM_JUDGE_MODEL, client=openai_client, usage_tracker=usage_tracker)

retrieval_eval = RetrievalEvaluator(llm_judge=llm_judge, system_prompt=ANSWERABLE_QS_SYSTEM_PROMPT, llm_output_schema=ANSWERABLE_QS_SCHEMA)
generation_eval = GenerationEvaluator(llm_judge=llm_judge)

run_pipeline(
    config=config,
    rag_pipeline=rag_pipeline,
    retrieval_eval=retrieval_eval,
    generation_eval=generation_eval,
    dataset_path=settings.EVAL_DATASET_DIR / "eval_dataset.jsonl",
    config_path=settings.EVAL_RESULTS_DIR / "config.json",
    raw_result_path=settings.EVAL_RESULTS_DIR / "raw_results.jsonl",
    final_result_path=settings.EVAL_RESULTS_DIR / "final_result.json"
)