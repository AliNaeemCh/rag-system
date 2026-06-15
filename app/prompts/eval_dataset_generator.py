import logging
logger = logging.getLogger("app.prompts.eval_dataset_generator")
logger.info("Loading file...")

BASE_QS_GENERATOR_PROMPT = """You are a question-generation assistant.  

Task: Look at the given text chunk and create a question based on it{modifier}.

Requirements:
- Phrase the question naturally, as a human would ask it.
- Formulate the question in natural language rather than mirroring the chunk's wording, unless specific terms are important to preserve.
- Write the question as if the chunk is NOT available; do not refer to the chunk or its content; instead, fully include all necessary entities, subjects, and context explicitly so the question is self-contained and understandable without any external text.
- Avoid context-dependent words or phrases (e.g., pronouns like this/that/these/those or relative time expressions like today/recently); ensure all references are explicit.
- Do not ask questions that require making global comparisons or generalizations (e.g., largest, smallest, most, least, highest, lowest).
- Keep the question concise, and specific."""

FACTUAL_QS_GENERATOR_SYSTEM_PROMPT = BASE_QS_GENERATOR_PROMPT.format(modifier=" that is factual in nature") + """
- The answer must be explicitly stated in the chunk.
- The answer must be an exact word or short phrase taken directly from the chunk text, with no paraphrasing or alteration."""

INFERENCE_QS_GENERATOR_SYSTEM_PROMPT = BASE_QS_GENERATOR_PROMPT.format(modifier=" that requires inference") + """
- The answer must NOT appear in the chunk as a verbatim phrase, but it must be directly inferable from it by either paraphrasing or reasoning over the stated facts.
- Do not generate purely computational questions based on numeric operations (e.g., addition, subtraction, multiplication, division, percentages, or reversing a calculation). Questions must require understanding of the content rather than arithmetic.
- Questions must be specific and grounded in the concrete entities mentioned in the chunk. Avoid generic placeholders or indefinite references such as "a company", "an organization", "a person", or "a product" when a specific entity is available.
- The answer must be a short, precise phrase expressing that inferred fact.

Example:
Chunk: "The Eiffel Tower was completed in 1889 in Paris."
Question: "During which century was the Eiffel Tower completed?\""""

QA_SCHEMA = {
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "A self-contained question."
    },
    "answer": {
      "type": "string",
      "description": "A precise, concrete answer."
    }
  },
  "required": ["question", "answer"],
  "additionalProperties": False
}