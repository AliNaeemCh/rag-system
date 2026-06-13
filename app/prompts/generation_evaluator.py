import logging
logger = logging.getLogger("app.prompts.generation_evaluator")
logger.info("Loading file...")

import copy

CORRECTNESS_FAITHFULNESS_EVAL_BASE_SYSTEM_PROMPT = """You are an evaluation model. Evaluate {metric_name} only.

{metric_name} = factual consistency between the generated answer and the {reference}.

Step 1 — Answer Presence Check:
Determine whether the generated answer provides any information that answers the question.

If the generated answer does not answer the question (e.g., it only states that information is unavailable, unknown, missing, or cannot be determined, or is otherwise irrelevant), return an empty `points` list.

Step 2 — Extract Points:
Extract all distinct answer points from the generated answer.

Only extract claims that could be substituted directly as an answer to the question.

Do not extract claims about missing information, uncertainty, the contents of the source, or the ability to answer the question.

If after exclusions, no point is left which directly answers the question, return an empty `points` list.

Step 3 — Compare:
For each extracted point:

* If the point is supported by or consistent with the {reference}, set `correct = true`.
* If the point contradicts the {reference} or is not supported by it, set `correct = false`.

Step 4 — Output:
Return all extracted points with their `correct` labels."""

CORRECTNESS_FAITHFULNESS_EVAL_BASE_SCHEMA = {
  "type": "object",
  "properties": {
    "points": {
      "type": "array",
      "description": "Distinct factual points in the generated answer that directly answer the question.",
      "items": {
        "type": "object",
        "properties": {
          "point": {
            "type": "string",
            "description": "One factual point from the generated answer related to the question."
          },
          "correct": {
            "type": "boolean",
            "description": "Whether this point is supported by or consistent with the {reference}."
          }
        },
        "required": [
          "point",
          "correct"
        ],
        "additionalProperties": False
      }
    }
  },
  "required": [
    "points"
  ],
  "additionalProperties": False
}

CORRECTNESS_EVAL_SYSTEM_PROMPT = CORRECTNESS_FAITHFULNESS_EVAL_BASE_SYSTEM_PROMPT.format(metric_name="Correctness", reference="reference answer")
CORRECTNESS_EVAL_SCHEMA = copy.deepcopy(CORRECTNESS_FAITHFULNESS_EVAL_BASE_SCHEMA)
CORRECTNESS_EVAL_SCHEMA["properties"]["points"]["items"]["properties"]["correct"]["description"] = \
    CORRECTNESS_EVAL_SCHEMA["properties"]["points"]["items"]["properties"]["correct"]["description"].format(
        reference="reference answer"
    )

FAITHFULNESS_EVAL_SYSTEM_PROMPT = CORRECTNESS_FAITHFULNESS_EVAL_BASE_SYSTEM_PROMPT.format(metric_name="Faithfulness", reference="retrieved context")
FAITHFULNESS_EVAL_SCHEMA = copy.deepcopy(CORRECTNESS_FAITHFULNESS_EVAL_BASE_SCHEMA)
FAITHFULNESS_EVAL_SCHEMA["properties"]["points"]["items"]["properties"]["correct"]["description"] = \
    FAITHFULNESS_EVAL_SCHEMA["properties"]["points"]["items"]["properties"]["correct"]["description"].format(
        reference="retrieved context"
    )

RELEVANCE_EVAL_SYSTEM_PROMPT = """You are an evaluation model. Score Relevance only.

Relevance measures how much of the generated answer is directly related to the question.

Break the generated answer into distinct factual points.

A point is relevant if it helps answer the question; otherwise it is irrelevant.

Count a point as irrelevant if it is off-topic, tangential, or does not help answer the question.

For each extracted point:
- If it helps answer the question → set relevant = true
- Otherwise → set relevant = false

Exceptions:
- Do NOT include inability-to-answer statements (e.g., "no information found", "insufficient information", "unknown") in the points list.

Important:
- If after exceptions, there is no point left, return empty `points` list.

Return all extracted points with their relevance labels."""

RELEVANCE_EVAL_SCHEMA = {
  "type": "object",
  "properties": {
    "points": {
      "type": "array",
      "description": "All distinct factual points in the generated answer.",
      "items": {
        "type": "object",
        "properties": {
          "point": {
            "type": "string",
            "description": "One factual point from the generated answer."
          },
          "relevant": {
            "type": "boolean",
            "description": "Whether this point directly helps answer the question."
          }
        },
        "required": [
          "point",
          "relevant"
        ],
        "additionalProperties": False
      }
    }
  },
  "required": [
    "points"
  ],
  "additionalProperties": False
}