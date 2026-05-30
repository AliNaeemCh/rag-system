
from app.core.config import settings

import logging
logger = logging.getLogger("app.rag.chat_history")
logger.info("Loading file...")

import time

class ChatHistory:
    def __init__(self):
        """
        store structure:
        {
            session_id: {
                "messages": [ {role, content}, ... ],
                "last_active": timestamp
            }
        }
        """
        self.store = {}

    def add(self, session_id: str, role: str, content: str):
        session = self.store.setdefault(session_id, {
            "messages": [],
            "last_active": time.time()
        })

        session["messages"].append({
            "role": role,
            "content": content
        })

        session["last_active"] = time.time()

        logger.debug(f"Session {session_id}: added {role} message")

    def get_recent(self, session_id: str, k: int = settings.CHAT_HISTORY_MAX_PAIRS) -> list[dict]:
        session = self.store.get(session_id)

        if not session:
            return []

        return session["messages"][-2*k:]

    def delete_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
            logger.info(f"Deleted session {session_id}")

    def cleanup(self, ttl_seconds: int = settings.CHAT_HISTORY_TTL_SECONDS):
        """
        Remove sessions inactive for longer than ttl_seconds.
        """
        now = time.time()

        to_delete = [
            sid for sid, session in self.store.items()
            if now - session["last_active"] > ttl_seconds
        ]

        for sid in to_delete:
            del self.store[sid]

        if to_delete:
            logger.info(f"Cleaned {len(to_delete)} inactive sessions")

    def clear(self):
        """
        Clear all sessions
        """
        self.store.clear()
        logger.warning("Cleared all chat history")

chat_history = ChatHistory()