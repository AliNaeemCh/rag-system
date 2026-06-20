import logging
logger = logging.getLogger("app.prompts.generation_evaluator")
logger.info("Loading file...")

FAITHFULNESS_EVAL_SYSTEM_PROMPT = """You are an evaluation model. Evaluate faithfulness only.

Faithfulness = factual consistency between the generated answer and the retrieved context.

Step 1 — Extract Points:
Extract all distinct factual points from the generated answer.

Step 2 — Relevance:
For each point, determine whether it directly answers the specific question asked.

* Set `relevant = true` only if the point provides an explicit answer value that satisfies the exact question asked, meaning it fills the required slot with the requested information (e.g., correct entity, number, date, metric, event, or result matching the question’s subject, attribute, and timeframe).
* Set `relevant = false` if the point does not provide an actual answer value, including cases where it restates or refers to the question without answering it, states or implies the answer is missing/unknown/not available/not stated/not found, provides explanations, reasoning, or context instead of the answer, includes unrelated or off-topic content, offers substitute or alternative information in place of the requested answer, or contains follow-up advice, suggestions, or next-step instructions.

For each relevance decision, provide a one-line concise `relevance_evidence` explaining why the label was assigned.

Step 3 — Correctness:
For each point, evaluate correctness.

- Set `correct = true` if the point is supported by or consistent with the retrieved context; differences in wording, format, or numerical representation (including scaled or rounded forms) should not be treated as disagreement.
- Otherwise set `correct = false`.

For each correctness decision, provide a one-line concise `correctness_evidence` explaining why the label was assigned.

Step 4 — Output:
Return the output according to the defined schema."""

FAITHFULNESS_EVAL_SCHEMA = {
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
            "description": "Whether the point is broadly supported by or consistent with the retrieved context."
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

REFERENCE_COVERAGE_EVAL_SYSTEM_PROMPT = """You are an evaluation model that measures **reference coverage only**.

Your task is to determine how much of the **reference answer’s factual content** is present in the **generated answer**.

First, break the reference answer into a list of distinct factual points. Each point should represent one clear fact or claim.

Then, for each factual point, check whether it is present in the generated answer.

Guidelines:

* Focus only on whether each factual point appears in the generated answer.
* Ignore extra information in the generated answer that is not present in the reference.
* Approximate matches are allowed:

  * Different wording is acceptable.
  * Differences in format, phrasing, or structure are acceptable.
  * Numerical values that are equivalent but expressed differently (e.g., rounded, scaled, or reformatted) should still be considered a match.
* Only mark a point as not present if it is clearly absent or meaningfully different.
* Do not count partial or ambiguous references as a match.

For each factual point, record whether it exists in the generated answer and provide brief evidence from the generated answer supporting your decision.

Return the output according to the defined schema."""

REFERENCE_COVERAGE_EVAL_SCHEMA = {
  "type": "object",
  "properties": {
    "points": {
      "type": "array",
      "description": "Factual points and their coverage evaluation.",
      "items": {
        "type": "object",
        "properties": {
          "point": {
            "type": "string",
            "description": "Factual point from reference answer."
          },
          "exists": {
            "type": "boolean",
            "description": "Whether the point appears in the generated answer."
          },
          "evidence": {
            "type": "string",
            "description": "One-line justification explaining why the correctness label was assigned."
          }
        },
        "required": ["point", "exists", "evidence"],
        "additionalProperties": False
      }
    }
  },
  "required": ["points"],
  "additionalProperties": False
}