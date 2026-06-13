import logging
logger = logging.getLogger("app.prompts.retrieval_evaluator")
logger.info("Loading file...")

ANSWERABLE_QS_SYSTEM_PROMPT = """You are given a context and one or more questions. Each question is annotated with a question ID, and a reference answer is also provided for each question.

Your task is to determine which questions can be answered using only the provided context, where the answer derived from the context agrees with the reference answer.

A question can be answered if its answer is:
- explicitly stated in the context, or
- can be logically inferred or derived from the context,

and the answer derived from the context must agree with the reference answer.

Do not use any outside knowledge; treat the context as the sole source of truth."""

ANSWERABLE_QS_SCHEMA = {
  "type": "object",
  "properties": {
    "questions": {
      "type": "array",
      "description": "Evaluation of each question against the provided context.",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "Question ID."
          },
          "answerable": {
            "type": "boolean",
            "description": "Whether this question can be answered from the context, such that the answer derived from the context agrees with the reference answer."
          },
          "evidence": {
            "type": "string",
            "description": "Concise one-line evidence supporting the decision."
          }
        },
        "required": [
          "id",
          "answerable",
          "evidence"
        ],
        "additionalProperties": False
      }
    }
  },
  "required": [
    "questions"
  ],
  "additionalProperties": False
}