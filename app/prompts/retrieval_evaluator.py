ANSWERABLE_QS_SYSTEM_PROMPT = """You are given a context and one or more questions. Each question is annotated with a question number, and a reference answer is also provided for each question.

Your task is to determine which questions can be answered using only the provided context, where the answer derived from the context agrees in meaning with the reference answer.

A question can be answered if its answer is:
- explicitly stated in the context, or
- can be logically inferred or derived from the context.

Do not use any outside knowledge; treat the context as the sole source of truth."""

ANSWERABLE_QS_SCHEMA = {
  "name": "answerable_questions_schema",
  "schema": {
    "type": "object",
    "properties": {
      "answerable_questions": {
        "type": "array",
        "description": "List of question numbers answerable from context and consistent with reference answers.",
        "items": {
          "type": "integer"
        }
      }
    },
    "required": ["answerable_questions"],
    "additionalProperties": False
  }
}