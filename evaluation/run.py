from app.core.logger import setup_logging
setup_logging()

from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.infra.dependencies import build_rag_pipeline, create_openai_client
from app.core.utils import load_jsonl, extract_last_jsonl_object, reset_jsonl, write_jsonl, load_pickle
from app.infra.usage_tracking.tracker import usage_tracker
from evaluation.config import EvalConfig
from app.rag.config import ResponseMode
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.generation_eval import GenerationEvaluator
from evaluation.dataset.generation.config import EvalQuestionType
from app.infra.llm_engines.openai.engine import OpenAIEngine
from app.prompts.retrieval_evaluator import ANSWERABLE_QS_SYSTEM_PROMPT, ANSWERABLE_QS_SCHEMA
from app.prompts.generation_evaluator import CORRECTNESS_EVAL_SYSTEM_PROMPT, CORRECTNESS_EVAL_SCHEMA, \
                                            COMPLETENESS_EVAL_SYSTEM_PROMPT, COMPLETENESS_EVAL_SCHEMA, \
                                            RELEVANCE_EVAL_SYSTEM_PROMPT, RELEVANCE_EVAL_SCHEMA
                                                
import logging
logger = logging.getLogger("evaluation.run")
logger.info("Loading file...")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    if question_type != EvalQuestionType.OUT_OF_KNOWLEDGE:
        reference_answers = []
        for _ in range(len(questions)):
            reference_answers.append("The question is invalid or logically flawed. The generated answer is expected to either refute it or not provide a direct response.")

    reference_answer = ""
    for ref_answer in reference_answers:
        reference_answer += ref_answer

        if reference_answer[-1] not in [".", "?", "!", "…", ";"]:
            reference_answer += ". "
        else:
            reference_answer += " "

    reference_answer = reference_answer.strip()

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
    retrieval_results = None

    if question_type != EvalQuestionType.OUT_OF_KNOWLEDGE:
        retrieval_results = retrieval_eval.evaluate(
            questions=questions,
            relevant_doc_ids=relevant_doc_ids,
            reference_answers=reference_answers,
            retrieved_docs=retrieved_docs,
            temperature=retrieval_eval_temperature
        )

    # Generation eval
    correctness_result = generation_eval.evaluate_correctness(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        system_prompt=CORRECTNESS_EVAL_SYSTEM_PROMPT,
        output_schema=CORRECTNESS_EVAL_SCHEMA,
        temperature=generation_eval_temperature
    )

    completeness_result = None

    if question_type != EvalQuestionType.OUT_OF_KNOWLEDGE:
        completeness_result = generation_eval.evaluate_completeness(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            system_prompt=COMPLETENESS_EVAL_SYSTEM_PROMPT,
            output_schema=COMPLETENESS_EVAL_SCHEMA,
            temperature=generation_eval_temperature
        )

    relevance_result = generation_eval.evaluate_relevance(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        system_prompt=RELEVANCE_EVAL_SYSTEM_PROMPT,
        output_schema=RELEVANCE_EVAL_SCHEMA,
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
            "correctness": correctness_result,
            "completeness": completeness_result,
            "relevance": relevance_result
        }
    }

def run_pipeline(config: EvalConfig,
                 rag_pipeline: RAGPipeline,
                 retrieval_eval: RetrievalEvaluator,
                 generation_eval: GenerationEvaluator,
                 chunks_path: Path,
                 chunks_index: list[int],
                 dataset_path: Path,
                 result_path: Path):
    try:
        def create_config_object():
            config_obj = {
                "response_mode": config.response_mode.value,
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
        last_obj = extract_last_jsonl_object(result_path)
        if config.resume:
            for obj in load_jsonl(path=result_path):
                if obj.get('example_id'):
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
            reset_jsonl(result_path)
            last_obj = {}
            logger.info("Evaluation started")
            
        if not last_obj:
            config_obj = create_config_object()
            write_jsonl(
                data=config_obj,
                output_path=result_path
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

                question_type = EvalQuestionType(obj['question_type'])
                relevant_doc_ids = []
                reference_answers = []

                for i, id in enumerate(obj['chunk_ids']):
                    relevant_doc_ids.append(id)
                    if question_type != EvalQuestionType.OUT_OF_KNOWLEDGE:
                        reference_answers.append(obj['answers'][i])

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
                                         generation_eval_temperatur=config.generation_eval_temperature
                                         )
                
                future_to_metadata[future] = {
                    "example_id": obj['example_id']
                }
                total_tasks += 1
            pbar.n = int((completed_tasks / total_tasks) * 100)
            pbar.refresh()
            print('here')
            print('futures are: ', futures)
            for future in as_completed(futures):
                result = future.result()
                write_jsonl(
                    data= {"example_id": future_to_metadata[future]["example_id"]} | result,
                    output_path=result_path
                )
                completed_tasks += 1
                pbar.n = int((completed_tasks / total_tasks) * 100)
                pbar.refresh()
    
    except:
        logger.exception("Evaluation failed!")

config = EvalConfig(resume=True)
rag_pipeline = build_rag_pipeline()
openai_client = create_openai_client(api_key=settings.OPENAI_API_KEY_SHARING)
llm_judge = OpenAIEngine(model_name=settings.LLM_JUDGE_MODEL, client=openai_client, usage_tracker=usage_tracker)
chunks_index = load_pickle(file_path=settings.PROCESSED_DATA_DIR / "chunks_jsonl_index.pkl")

retrieval_eval = RetrievalEvaluator(llm_judge=llm_judge, system_prompt=ANSWERABLE_QS_SYSTEM_PROMPT, llm_output_schema=ANSWERABLE_QS_SCHEMA)
generation_eval = GenerationEvaluator(llm_judge=llm_judge)

run_pipeline(
    config=config,
    rag_pipeline=rag_pipeline,
    retrieval_eval=retrieval_eval,
    generation_eval=generation_eval,
    chunks_path=settings.PROCESSED_DATA_DIR / "sys_annual_2025_chunks.jsonl",
    chunks_index=chunks_index,
    dataset_path=settings.EVAL_DATASET_DIR / "eval_dataset.jsonl",
    result_path=settings.EVAL_RESULTS_DIR / "raw_results.jsonl"
)