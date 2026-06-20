from evaluation.dataset.generation.config import EvalQuestionType
from app.core.config import settings

import logging
logger = logging.getLogger("evaluation.utils")
logger.info("Loading file...")

import copy

def get_eval_results_obj():

    retrieval_result_obj = {
        "mrr": {
            "sum": 0,
            "values": 0
        },
        "recall_k": {
            "sum": 0,
            "values": 0
        }
    }

    generation_result_obj = {
        "reference_coverage": {
            "sum": 0,
            "values": 0
        },
        "faithfulness": {
            "sum": 0,
            "values": 0
        }
    }  

    eval_results = {
        "retrieval": {
            "question_type_to_result": {},
            "final_result": copy.deepcopy(retrieval_result_obj)
        },
        "generation": {
            "question_type_to_result": {},
            "final_result": copy.deepcopy(generation_result_obj)
        }
    }

    for q_type in EvalQuestionType:

        if q_type != EvalQuestionType.MULTI_CHUNK.value:
            eval_results['retrieval']['question_type_to_result'][q_type.value] = copy.deepcopy(retrieval_result_obj)
            eval_results['generation']['question_type_to_result'][q_type.value] = copy.deepcopy(generation_result_obj)

        else:
            eval_results['retrieval']['question_type_to_result'][q_type.value] = {
                "num_chunks_to_result": {},
                "avg_result": copy.deepcopy(retrieval_result_obj)
            }
            eval_results['generation']['question_type_to_result'][q_type.value] = {
                "num_chunks_to_result": {},
                "avg_result": copy.deepcopy(generation_result_obj)
            }
            for i in range(2, settings.FINAL_TOP_K+1):
                eval_results['retrieval']['question_type_to_result'][q_type.value]["num_chunks_to_result"][i] = copy.deepcopy(retrieval_result_obj)
                eval_results['generation']['question_type_to_result'][q_type.value]["num_chunks_to_result"][i] = copy.deepcopy(generation_result_obj)

    return eval_results

def update_eval_results(eval_results: dict, new_result: dict) -> None:
    mrr = new_result['retrieval_results']['mrr']
    recall_k = new_result['retrieval_results']['recall_k']

    if new_result['question_type'] != EvalQuestionType.MULTI_CHUNK.value:
        # MRR
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]['mrr']['sum'] += mrr
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]['mrr']['values'] += 1
        # Recall@k
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]['recall_k']['sum'] += recall_k
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]['recall_k']['values'] += 1
    
    else:
        num_chunks = len(new_result['relevant_doc_ids'])
        # MRR
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['mrr']['sum'] += mrr
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['mrr']['values'] += 1
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["avg_result"]['mrr']['sum'] += mrr
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["avg_result"]['mrr']['values'] += 1
        # Recall@k
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['recall_k']['sum'] += recall_k
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['recall_k']['values'] += 1
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["avg_result"]['recall_k']['sum'] += recall_k
        eval_results['retrieval']['question_type_to_result'][new_result['question_type']]["avg_result"]['recall_k']['values'] += 1

    eval_results['retrieval']['final_result']['mrr']['sum'] += mrr
    eval_results['retrieval']['final_result']['mrr']['values'] += 1
    eval_results['retrieval']['final_result']['recall_k']['sum'] += recall_k
    eval_results['retrieval']['final_result']['recall_k']['values'] += 1

    reference_coverage = new_result['generation_results']['reference_coverage']['reference_coverage']
    faithfulness = new_result['generation_results']['faithfulness']['faithfulness']

    if new_result['question_type'] != EvalQuestionType.MULTI_CHUNK.value:
        # Reference coverage
        eval_results['generation']['question_type_to_result'][new_result['question_type']]['reference_coverage']['sum'] += reference_coverage
        eval_results['generation']['question_type_to_result'][new_result['question_type']]['reference_coverage']['values'] += 1
        # Faithfulness
        eval_results['generation']['question_type_to_result'][new_result['question_type']]['faithfulness']['sum'] += faithfulness
        eval_results['generation']['question_type_to_result'][new_result['question_type']]['faithfulness']['values'] += 1

    
    else:
        num_chunks = len(new_result['relevant_doc_ids'])

        # Reference coverage
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['reference_coverage']['sum'] += reference_coverage
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['reference_coverage']['values'] += 1
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["avg_result"]['reference_coverage']['sum'] += reference_coverage
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["avg_result"]['reference_coverage']['values'] += 1
        # Faithfulness
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['faithfulness']['sum'] += faithfulness
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["num_chunks_to_result"][num_chunks]['faithfulness']['values'] += 1
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["avg_result"]['faithfulness']['sum'] += faithfulness
        eval_results['generation']['question_type_to_result'][new_result['question_type']]["avg_result"]['faithfulness']['values'] += 1

    eval_results['generation']['final_result']['reference_coverage']['sum'] += reference_coverage
    eval_results['generation']['final_result']['reference_coverage']['values'] += 1
    eval_results['generation']['final_result']['faithfulness']['sum'] += faithfulness
    eval_results['generation']['final_result']['faithfulness']['values'] += 1

def aggregate_eval_results(eval_results: dict) -> None:
    # Retrieval
    for q_type, results in eval_results['retrieval']['question_type_to_result'].items():
        if q_type != EvalQuestionType.MULTI_CHUNK:
            results['mrr']['mrr'] = results['mrr']['sum'] / results['mrr']['values']
            results['recall_k']['recall_k'] = results['recall_k']['sum'] / results['recall_k']['values']
        else:
            for num_chunks, results_2 in results['num_chunks_to_result'].items():
                results_2['mrr']['mrr'] = results_2['mrr']['sum'] / results_2['mrr']['values']
                results_2['recall_k']['recall_k'] = results_2['recall_k']['sum'] / results_2['recall_k']['values']

    multi_chunk_ret_avg_result_obj = eval_results['retrieval']['question_type_to_result'][EvalQuestionType.MULTI_CHUNK.value]['avg_result']
    multi_chunk_ret_avg_result_obj['mrr']['mrr'] = multi_chunk_ret_avg_result_obj['mrr']['sum'] / multi_chunk_ret_avg_result_obj['mrr']['values']
    multi_chunk_ret_avg_result_obj['recall_k']['recall_k'] = multi_chunk_ret_avg_result_obj['recall_k']['sum'] / multi_chunk_ret_avg_result_obj['recall_k']['values']

    # Generation
    for q_type, results in eval_results['generation']['question_type_to_result'].items():
        if q_type != EvalQuestionType.MULTI_CHUNK:
            results['reference_coverage']['reference_coverage'] = results['reference_coverage']['sum'] / results['reference_coverage']['values']
            results['faithfulness']['faithfulness'] = results['faithfulness']['sum'] / results['faithfulness']['values']
        else:
            for num_chunks, results_2 in results['num_chunks_to_result'].items():
                results_2['reference_coverage']['reference_coverage'] = results_2['reference_coverage']['sum'] / results_2['reference_coverage']['values']
                results_2['faithfulness']['faithfulness'] = results_2['faithfulness']['sum'] / results_2['faithfulness']['values']

    multi_chunk_gen_avg_result_obj = eval_results['generation']['question_type_to_result'][EvalQuestionType.MULTI_CHUNK.value]['avg_result']
    multi_chunk_gen_avg_result_obj['reference_coverage']['reference_coverage'] = multi_chunk_gen_avg_result_obj['reference_coverage']['sum'] / multi_chunk_gen_avg_result_obj['reference_coverage']['values']
    multi_chunk_gen_avg_result_obj['faithfulness']['faithfulness'] = multi_chunk_gen_avg_result_obj['faithfulness']['sum'] / multi_chunk_gen_avg_result_obj['faithfulness']['values']

    # Final result
    eval_results['retrieval']['final_result']['mrr']['mrr'] = eval_results['retrieval']['final_result']['mrr']['sum'] / eval_results['retrieval']['final_result']['mrr']['values']
    eval_results['retrieval']['final_result']['recall_k']['recall_k'] = eval_results['retrieval']['final_result']['recall_k']['sum'] / eval_results['retrieval']['final_result']['recall_k']['values']

    eval_results['generation']['final_result']['reference_coverage']['reference_coverage'] = eval_results['generation']['final_result']['reference_coverage']['sum'] / eval_results['generation']['final_result']['reference_coverage']['values']
    eval_results['generation']['final_result']['faithfulness']['faithfulness'] = eval_results['generation']['final_result']['faithfulness']['sum'] / eval_results['generation']['final_result']['faithfulness']['values']
        
