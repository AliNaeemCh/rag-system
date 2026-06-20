from evaluation.dataset.generation.config import EvalQuestionType
from app.core.config import settings


import logging
logger = logging.getLogger("evaluation.visualization.utils")
logger.info("Loading file...")

import pandas as pd

def percentage(value):
    return round(value * 100, 1)

def create_question_type_df(results: dict):

    retrieval = results["retrieval"]
    generation = results["generation"]

    rows = []

    for qtype in  [x.value for x in EvalQuestionType]:
        if qtype != EvalQuestionType.MULTI_CHUNK:
            rows.append({
                "Category": qtype.capitalize(),
                "MRR": percentage(
                    retrieval["question_type_to_result"][qtype]["mrr"]["mrr"]
                ),
                f"Recall@{settings.FINAL_TOP_K}": percentage(
                    retrieval["question_type_to_result"][qtype]["recall_k"]["recall_k"]
                ),
                "Ref. Coverage": percentage(
                    generation["question_type_to_result"][qtype]["reference_coverage"]["reference_coverage"]
                ),
                "Faithfulness": percentage(
                    generation["question_type_to_result"][qtype]["faithfulness"]["faithfulness"]
                )
            })
        else:
            rows.append({
                "Category": "Multi-Chunk",
                "MRR": percentage(
                    retrieval["question_type_to_result"][qtype]['avg_result']["mrr"]["mrr"]
                ),
                f"Recall@{settings.FINAL_TOP_K}": percentage(
                    retrieval["question_type_to_result"][qtype]['avg_result']["recall_k"]["recall_k"]
                ),
                "Ref. Coverage": percentage(
                    generation["question_type_to_result"][qtype]['avg_result']["reference_coverage"]["reference_coverage"]
                ),
                "Faithfulness": percentage(
                    generation["question_type_to_result"][qtype]['avg_result']["faithfulness"]["faithfulness"]
                )
            })

    return pd.DataFrame(rows)

def create_multi_chunk_df(results: dict):

    retrieval = results["retrieval"]
    generation = results["generation"]

    rows = []

    multi_chunk = retrieval["question_type_to_result"]["multi_chunk"]["num_chunks_to_result"]
    generation_multi = generation["question_type_to_result"]["multi_chunk"]["num_chunks_to_result"]

    for chunks in range(2, settings.FINAL_TOP_K+1):
        chunks_str = str(chunks)
        rows.append({
            "Chunks": int(chunks_str),

            "MRR": percentage(
                multi_chunk[chunks_str]["mrr"]["mrr"]
            ),

            f"Recall@{settings.FINAL_TOP_K}": percentage(
                multi_chunk[chunks_str]["recall_k"]["recall_k"]
            ),

            "Ref. Coverage": percentage(
                generation_multi[chunks_str]["reference_coverage"]["reference_coverage"]
            ),

            "Faithfulness": percentage(
                generation_multi[chunks_str]["faithfulness"]["faithfulness"]
            )
        })

    return pd.DataFrame(rows)

def create_heatmap_df(
    question_type_df: pd.DataFrame,
    multi_chunk_df: pd.DataFrame
):

    rows = []

    # factual
    rows.append(
        question_type_df.iloc[0].to_dict()
    )

    # inference
    rows.append(
        question_type_df.iloc[1].to_dict()
    )

    # multi chunk breakdown
    for _, row in multi_chunk_df.iterrows():

        rows.append({
            "Category": f"Multi-{row['Chunks']}",
            "MRR": row["MRR"],
            f"Recall@{settings.FINAL_TOP_K}": row[f"Recall@{settings.FINAL_TOP_K}"],
            "Ref. Coverage": row["Ref. Coverage"],
            "Faithfulness": row["Faithfulness"]
        })

    # overall
    rows.append(
        question_type_df.iloc[-1].to_dict()
    )

    return (
        pd.DataFrame(rows)
        .set_index("Category")
    )

def create_summary_df(results):

    retrieval = results["retrieval"]
    generation = results["generation"]

    # -------------------------
    # Single Chunk (factual + inference average)
    # -------------------------

    factual = retrieval["question_type_to_result"]["factual"]
    inference = retrieval["question_type_to_result"]["inference"]

    factual_gen = generation["question_type_to_result"]["factual"]
    inference_gen = generation["question_type_to_result"]["inference"]


    single_chunk = {
        "Category": "Single-Chunk",

        "MRR": percentage(
            (
                factual["mrr"]["sum"] +
                inference["mrr"]["sum"]
            ) / (factual["mrr"]["values"] + inference["mrr"]["values"])
        ),

        f"Recall@{settings.FINAL_TOP_K}": percentage(
            (
                factual["recall_k"]["sum"] +
                inference["recall_k"]["sum"]
            ) / (factual["recall_k"]["values"] + inference["recall_k"]["values"])
        ),

        "Ref. Coverage": percentage(
            (
                factual_gen["reference_coverage"]["sum"] +
                inference_gen["reference_coverage"]["sum"]
            ) /  (factual_gen["reference_coverage"]["values"] + inference_gen["reference_coverage"]["values"])
        ),

        "Faithfulness": percentage(
            (
                factual_gen["faithfulness"]["sum"] +
                inference_gen["faithfulness"]["sum"]
            ) / (factual_gen["faithfulness"]["values"] + inference_gen["faithfulness"]["values"])
        )
    }


    # -------------------------
    # Multi Chunk average
    # -------------------------

    multi = retrieval["question_type_to_result"]["multi_chunk"]["avg_result"]
    multi_gen = generation["question_type_to_result"]["multi_chunk"]["avg_result"]


    multi_chunk = {
        "Category": "Multi-Chunk",

        "MRR": percentage(
            multi["mrr"]["mrr"]
        ),

        f"Recall@{settings.FINAL_TOP_K}": percentage(
            multi["recall_k"]["recall_k"]
        ),

        "Ref. Coverage": percentage(
            multi_gen["reference_coverage"]["reference_coverage"]
        ),

        "Faithfulness": percentage(
            multi_gen["faithfulness"]["faithfulness"]
        )
    }


    # -------------------------
    # Overall
    # -------------------------

    overall = {
        "Category": "Overall",

        "MRR": percentage(
            retrieval["final_result"]["mrr"]["mrr"]
        ),

        f"Recall@{settings.FINAL_TOP_K}": percentage(
            retrieval["final_result"]["recall_k"]["recall_k"]
        ),

        "Ref. Coverage": percentage(
            generation["final_result"]["reference_coverage"]["reference_coverage"]
        ),

        "Faithfulness": percentage(
            generation["final_result"]["faithfulness"]["faithfulness"]
        )
    }


    return pd.DataFrame([
        single_chunk,
        multi_chunk,
        overall
    ])