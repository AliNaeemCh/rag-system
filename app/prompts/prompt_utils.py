from google.genai import types

def build_openai_messages(system_prompt: str, user_prompt: str, history: list = []):
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_prompt}]
    )

def build_rewriter_user_prompt(query, chat_history):
    context = "\n".join([
        f"{'User' if i % 2 == 0 else 'Assistant'}: {obj['content'].strip()}"
        for i, obj in enumerate(chat_history)
    ]) if len(chat_history) > 0 else "Not available"
    user_prompt = f"""Chat history (for context only):
{context}
---
User's message to rewrite:
{query}
"""
    return user_prompt


def build_gemini_messages(
    system_instruction: str,
    history: list = [],
    user_prompt: str | None = None,
    image=None
):
    contents = []

    # ---------------- HISTORY ----------------
    contents.extend(history)

    # ---------------- USER TURN ----------------
    if user_prompt is not None or image is not None:
        parts = []

        if user_prompt is not None:
            parts.append(types.Part(text=user_prompt))

        if image is not None:
            parts.append(image)

        contents.append(
            types.Content(
                role="user",
                parts=parts
            )
        )

    return {
        "contents": contents,
        "config": types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    }