import logging
logger = logging.getLogger("app.prompts.generation_evaluator")
logger.info("Loading file...")

CORRECTNESS_EVAL_SYSTEM_PROMPT = """You are an evaluation model. Evaluate correctness only.

Correctness = factual consistency between the generated answer and the reference answer.

Step 1 — Answer Presence Check:
If the generated answer does NOT attempt to answer the question
(e.g., says "I don't know", "no information found", or is irrelevant),
then set both fields (total_points and incorrect_points) to 0 and RETURN.

Step 2 — Extract Points:
Identify all distinct factual points in the generated answer that directly answer the question. Ignore any irrelevant, tangential, background, or contextual information that does not contribute to answering the question.

Step 3 — Compare:
For each point:

* If it is supported by or consistent with the reference answer → correct
* If it contradicts or is not supported → incorrect

Important:

* Missing information is NOT incorrect
* Only contradictions count as incorrect

Step 4 — Output: Return the counts of total relevant factual points and incorrect factual points in the required schema."""

CORRECTNESS_EVAL_SCHEMA = {
  "name": "correctness_evaluation",
  "schema": {
    "type": "object",
    "properties": {
      "total_points": {
        "type": "integer",
        "description": "Number of distinct factual points in the generated answer that directly answer the question."
      },
      "incorrect_points": {
        "type": "integer",
        "description": "Number of those answer-relevant factual points that contradict or are not supported by the reference answer."
      }
    },
    "required": [
      "total_points",
      "incorrect_points"
    ],
    "additionalProperties": False
  }
}

COMPLETENESS_EVAL_SYSTEM_PROMPT = """You are an evaluation model. Score completeness only.

Completeness measures how many factual points from the reference answer are present in the generated answer.

The reference answer may contain multiple distinct factual points. Evaluate whether each point is present in the generated answer (approximate match is sufficient; exact match is not required).

Rules:

* Ignore extra information in the generated answer that is not in the reference.
* Do not evaluate correctness, style, or relevance—only whether reference points are present.
* Partial or ambiguous mentions do NOT count as matched.

Return counts for total_points and matched_points."""

COMPLETENESS_EVAL_SCHEMA = {
  "name": "completeness_evaluation",
  "schema": {
    "type": "object",
    "properties": {
      "total_points": {
        "type": "integer",
        "description": "Total number of distinct factual points in the reference answer."
      },
      "matched_points": {
        "type": "integer",
        "description": "Number of reference factual points that are present in the generated answer."
      }
    },
    "required": [
      "total_points",
      "matched_points"
    ],
    "additionalProperties": False
  }
}

RELEVANCE_EVAL_SYSTEM_PROMPT = """You are an evaluation model. Score relevance only.

Relevance measures how much of the generated answer is directly related to the question.

Break the generated answer into distinct points. A point is relevant if it helps answer the question; otherwise it is irrelevant.

Rules:
- Ignore correctness and completeness—only relevance matters.
- Count a point as irrelevant if it is off-topic, tangential, or does not help answer the question.
- Do NOT count greetings, polite phrases, safety disclaimers, or follow-up suggestions as irrelevant if they do not change the factual answer content.

Return counts for total_points and irrelevant_points."""

RELEVANCE_EVAL_SCHEMA = {
  "name": "relevance_evaluation",
  "schema": {
    "type": "object",
    "properties": {
      "total_points": {
        "type": "integer",
        "description": "Total number of distinct factual points in the generated answer."
      },
      "irrelevant_points": {
        "type": "integer",
        "description": "Number of points in the generated answer that are off-topic, tangential, or do not help answer the question."
      }
    },
    "required": [
      "total_points",
      "irrelevant_points"
    ],
    "additionalProperties": False
  }
}