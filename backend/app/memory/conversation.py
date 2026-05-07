"""
RAGe — Conversation Memory

In-memory multi-turn conversation history with follow-up detection.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Turn:
    """A single conversation turn."""
    user_query: str
    assistant_answer: str
    timestamp: float = field(default_factory=time.time)


PRONOUN_PATTERNS = re.compile(
    r'\b(it|that|they|this|its|their|them|those|these|he|she|previous|earlier|above)\b',
    re.IGNORECASE,
)

REFERRING_PHRASES = [
    "about that", "from before", "you mentioned", "you said", "previously",
    "tell me more", "go on", "continue", "what about", "and also", "follow up",
]


class ConversationMemory:
    """Per-conversation turn history."""

    MAX_TURNS = 5
    WINDOW = 3

    def __init__(self):
        self._store: Dict[str, List[Turn]] = {}

    def add_turn(self, conv_id: str, user: str, assistant: str) -> None:
        turns = self._store.setdefault(conv_id, [])
        turns.append(Turn(user_query=user, assistant_answer=assistant))
        if len(turns) > self.MAX_TURNS:
            self._store[conv_id] = turns[-self.MAX_TURNS:]

    def get_history(self, conv_id: str) -> Optional[List[Dict[str, str]]]:
        turns = self._store.get(conv_id, [])
        if not turns:
            return None
        return [
            {"user": t.user_query, "assistant": t.assistant_answer[:250]}
            for t in turns[-self.WINDOW:]
        ]

    def is_followup(self, query: str, conv_id: str) -> bool:
        if conv_id not in self._store or not self._store[conv_id]:
            return False
        q = query.lower().strip()
        if PRONOUN_PATTERNS.search(q):
            return True
        if len(q.split()) < 5:
            return True
        if any(phrase in q for phrase in REFERRING_PHRASES):
            return True
        return False
