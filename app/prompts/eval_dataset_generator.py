BASE_QS_GENERATOR_PROMPT = """You are a question-generation assistant.  

Task: Look at the given text chunk and create a question based on it.

Requirements:
- Phrase the question naturally, as a human would ask it.
- Formulate the question in natural language rather than mirroring the chunk's wording, unless specific terms are important to preserve.
- Write the question as if the chunk is NOT available; do not refer to the chunk or its content.  
- Avoid context-dependent words or phrases (e.g., pronouns like this/that/these/those or relative time expressions like today/recently); ensure all references are explicit.
- Do not ask questions that require making global comparisons or generalizations (e.g., largest, smallest, most, least, highest, lowest).
- Keep the question concise, self-contained, and specific."""

FACTUAL_QS_GENERATOR_SYSTEM_PROMPT = BASE_QS_GENERATOR_PROMPT + """
- The answer must be explicitly stated in the chunk.
- The answer must be an exact word or short phrase taken directly from the chunk text, with no paraphrasing or alteration."""

INFERENCE_QS_GENERATOR_SYSTEM_PROMPT = BASE_QS_GENERATOR_PROMPT + """
- The answer must NOT appear in the chunk as a verbatim phrase, but it must be directly inferable from it by either paraphrasing or reasoning over the stated facts.
- The answer must be a short, precise phrase expressing that inferred fact.

Example:
Chunk: "The Eiffel Tower was completed in 1889 in Paris."
Question: "During which century was the Eiffel Tower completed?\""""

OUT_OF_KNOWLEDGE_QS_GENERATOR_SYSTEM_PROMPT = BASE_QS_GENERATOR_PROMPT + """
- Base the question on the chunk but it must contain a direct contradiction of an explicitly stated fact in the chunk.

Example:
Chunk: "Thomas Edison invented the phonograph in 1877."
Question: "Who assisted Thomas Edison in inventing the phonograph in 1800?\""""

QA_OUTPUT_SCHEMA = {
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

Q_OUTPUT_SCHEMA = {
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "A self-contained question."
    }
  },
  "required": ["question"],
  "additionalProperties": False
}

# INFERENCE_QS_GENERATOR_SYSTEM_PROMPT = """You are a question-answer-generation assistant.
# Your task: Look at the given text chunk and create **one clear question** whose answer is NOT explicitly stated in the chunk but can be logically inferred from it.

# Requirements:  
# - Phrase the question naturally, as a human would ask it.  
# - The answer must be derivable from the information in the chunk, but it should not be directly stated in the text.  
# - Prefer natural paraphrasing over exact wording, but keep similar phrasing when it feels natural or important.
# - Write the question as if the chunk is NOT available; do not refer to the chunk or its content.  
# - Do not use any context-dependent words or phrases whose meaning depends on information outside the question itself. This includes pronouns (this, that, these, those) and relative time expressions (this year, last year, today, recently). All references must be explicit and self-contained. 
# - Make sure the question is concise, self-contained, and has a clear, concrete, and definite answer rather than a vague or subjective one.
# - Do not ask questions that require making global comparisons or generalizations (e.g., largest, smallest, most, least, highest, lowest).
# - Focus the question on a single explicit fact or relationship within the chunk, not a global evaluation across entities.
# - The answer must be a complete and precise word or short phrase as inferred from the chunk text. Prioritize correctness and completeness over brevity; be concise only if no information is lost."""
