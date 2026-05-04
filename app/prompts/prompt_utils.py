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