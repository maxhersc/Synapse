from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
import uuid


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Priority(IntEnum):
    """Message priority used for ordering work on the bus."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(str, Enum):
    """Lifecycle state for a task managed by the coordinator."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Goal:
    """Top-level objective submitted to a Synapse team."""

    description: str
    id: str = field(default_factory=_uuid_str)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass
class Task:
    """Unit of work derived from a goal and assigned to an agent."""

    description: str
    goal_id: str
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    id: str = field(default_factory=_uuid_str)
    created_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, result: Any) -> None:
        self.result = result
        self.status = TaskStatus.COMPLETE
        self.completed_at = _utc_now()

    def fail(self, reason: Any) -> None:
        self.result = reason
        self.status = TaskStatus.FAILED
        self.completed_at = _utc_now()

    def block(self) -> None:
        self.status = TaskStatus.BLOCKED


@dataclass
class Message:
    """Message exchanged between agents over the Synapse bus."""

    sender_id: str
    recipient_id: str
    content: str | dict[str, Any]
    priority: Priority = Priority.NORMAL
    thread_id: str = field(default_factory=_uuid_str)
    reply_to: str | None = None
    id: str = field(default_factory=_uuid_str)
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reply(self, sender_id: str, content: str | dict[str, Any]) -> Message:
        return Message(
            sender_id=sender_id,
            recipient_id=self.sender_id,
            content=content,
            priority=self.priority,
            thread_id=self.thread_id,
            reply_to=self.id,
        )

    def __lt__(self, other: Message) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


@dataclass
class HelpRequest:
    """Formal signal that an agent is blocked and needs help."""

    from_agent_id: str
    task_id: str
    reason: str
    capability_needed: str | None = None
    id: str = field(default_factory=_uuid_str)
    created_at: datetime = field(default_factory=_utc_now)
