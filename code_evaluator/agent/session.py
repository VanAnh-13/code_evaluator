"""
Agent Session Management
Manages conversation history, step tracking, and session state for the AI agent.
"""

import time
import uuid
import json
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of agent execution steps."""
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE = "response"
    ERROR = "error"


class SessionStatus(str, Enum):
    """Status of an agent session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    step_number: int
    type: StepType
    content: str = ""
    tool_name: str = ""
    tool_args: Dict = field(default_factory=dict)
    tool_result: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "step_number": self.step_number,
            "type": self.type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result[:2000] if self.tool_result else "",
            "timestamp": self.timestamp,
        }


class AgentSession:
    """
    Manages state for a single agent session including
    conversation history, execution steps, and context.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_steps: int = 15,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Dict] = []
        self.steps: List[AgentStep] = []
        self.status = SessionStatus.PENDING
        self.created_at = time.time()
        self.updated_at = time.time()
        self.max_steps = max_steps
        self.current_step = 0
        self.context: Dict[str, Any] = {
            "files_analyzed": [],
            "issues_found": 0,
            "fixes_applied": 0,
            "fixes_verified": 0,
        }
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None

        # Subscribers for SSE streaming
        self._step_callbacks: List = []

    def add_message(self, role: str, content: Any, **kwargs) -> None:
        """Add a message to conversation history."""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self.updated_at = time.time()

    def add_step(self, step: AgentStep) -> None:
        """Add an execution step and notify subscribers."""
        self.steps.append(step)
        self.current_step = step.step_number
        self.updated_at = time.time()

        # Notify SSE subscribers
        for callback in self._step_callbacks:
            try:
                callback(step)
            except Exception as e:
                logger.debug(f"Step callback error: {e}")

    def subscribe(self, callback) -> None:
        """Subscribe to step updates (for SSE streaming)."""
        self._step_callbacks.append(callback)

    def unsubscribe(self, callback) -> None:
        """Unsubscribe from step updates."""
        try:
            self._step_callbacks.remove(callback)
        except ValueError:
            pass

    @property
    def can_continue(self) -> bool:
        """Check if the agent can execute more steps."""
        return (
            self.current_step < self.max_steps
            and self.status in (SessionStatus.PENDING, SessionStatus.RUNNING)
        )

    def set_running(self) -> None:
        self.status = SessionStatus.RUNNING
        self.updated_at = time.time()

    def set_completed(self, result: Optional[Dict] = None) -> None:
        self.status = SessionStatus.COMPLETED
        self.result = result
        self.updated_at = time.time()

    def set_error(self, error: str) -> None:
        self.status = SessionStatus.ERROR
        self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> Dict:
        """Serialize session to dict (for API response)."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "result": self.result,
            "error": self.error,
        }


class SessionManager:
    """
    Manages multiple agent sessions with in-memory storage.
    Provides session lifecycle operations and TTL-based cleanup.
    """

    def __init__(self, ttl: int = 3600):
        """
        Args:
            ttl: Session time-to-live in seconds (default: 1 hour).
        """
        self._sessions: Dict[str, AgentSession] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def create(self, max_steps: int = 15) -> AgentSession:
        """Create a new agent session."""
        session = AgentSession(max_steps=max_steps)
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info(f"Created agent session: {session.session_id}")
        return session

    def get(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "status": s.status.value,
                    "created_at": s.created_at,
                    "step_count": len(s.steps),
                }
                for s in self._sessions.values()
            ]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted agent session: {session_id}")
                return True
        return False

    def cleanup_expired(self) -> int:
        """Remove sessions older than TTL. Returns number of cleaned sessions."""
        now = time.time()
        expired = []
        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.updated_at > self._ttl:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired agent sessions.")
        return len(expired)

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return len(self._sessions)
