import logging
logger = logging.getLogger("app.prompts.generation_evaluator")
logger.info("Loading file...")

import copy

CORRECTNESS_FAITHFULNESS_EVAL_BASE_SYSTEM_PROMPT = """You are an evaluation model. Evaluate {metric_name} only.

{metric_name} = factual consistency between the generated answer and the {reference}.

Step 1 — Extract Points:
Extract all distinct factual points from the generated answer.

Step 2 — Relevance:
For each point, determine whether it directly answers the specific question asked.

* Set `relevant = true` only if the point provides an explicit answer value that satisfies the exact question asked, meaning it fills the required slot with the requested information (e.g., correct entity, number, date, metric, event, or result matching the question’s subject, attribute, and timeframe).
* Set `relevant = false` if the point does not provide an actual answer value, including cases where it restates or refers to the question without answering it, states or implies the answer is missing/unknown/not available/not stated, provides explanations, reasoning, or context instead of the answer, includes unrelated or off-topic content, offers substitute or alternative information in place of the requested answer, or contains follow-up advice, suggestions, or next-step instructions.

For each relevance decision, provide a one-line concise `relevance_evidence` explaining why the label was assigned.

Step 3 — Correctness:
For each point, evaluate correctness.

- Set `correct = true` if the point is supported by or consistent with the {reference}; differences in wording, format, or numerical representation (including scaled or rounded forms) should not be treated as disagreement.
- Otherwise set `correct = false`.

For each correctness decision, provide a one-line concise `correctness_evidence` explaining why the label was assigned.

Step 4 — Output:
Return all extracted points with:
- point
- relevant
- relevance_evidence
- correct
- correctness_evidence"""

CORRECTNESS_FAITHFULNESS_EVAL_BASE_SCHEMA = {
  "type": "object",
  "properties": {
    "points": {
      "type": "array",
      "description": "Distinct factual points extracted from the generated answer.",
      "items": {
        "type": "object",
        "properties": {
          "point": {
            "type": "string",
            "description": "A factual statement extracted from the generated answer."
          },
          "relevant": {
            "type": "boolean",
            "description": "Whether the point directly answers the specific question asked."
          },
          "relevance_evidence": {
            "type": "string",
            "description": "One-line justification explaining why the relevance label was assigned."
          },
          "correct": {
            "type": "boolean",
            "description": "Whether the point is broadly supported by or consistent with the {reference}."
          },
          "correctness_evidence": {
            "type": "string",
            "description": "One-line justification explaining why the correctness label was assigned."
          }
        },
        "required": ["point", "relevant", "relevance_evidence", "correct", "correctness_evidence"],
        "additionalProperties": False
      }
    }
  },
  "required": ["points"],
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

Relevance measures how much of the generated answer relates to the question.

Break the generated answer into distinct factual points.

A point is relevant if it contributes to answering the question, including direct answers, relevant reasoning, supporting argumentation, inability-to-answer statements, closely related substitute information provided when the requested information is unavailable, generic offer for further help, or useful follow-up guidance (e.g., advice or next steps).

A point is irrelevant only if it is completely unrelated to the question.

For each extracted point:
- If relevant → set `relevant = true`
- Otherwise → set `relevant = false`

For each relevance decision, provide a one-line concise `evidence` explaining why the label was assigned.

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
            "description": "Whether this point contributes to answering the question under the evaluation rules."
          },
          "evidence": {
            "type": "string",
            "description": "One-line justification explaining why the relevance label was assigned."
          },
        },
        "required": [
          "point",
          "relevant",
          "evidence"
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