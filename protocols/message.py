"""
Synapse v0.2 — Protocol definitions.

All message types that flow between agents: Messages, Tasks, Goals,
HelpRequests, and their associated enums (Priority, TaskStatus).
"""

from __future__ import annotations

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any


class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED_PENDING_INPUT = "blocked_pending_input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Message:
    """A natural-language message between two agents."""

    sender: str
    recipient: str
    content: str
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reply(self, sender: str, content: str) -> "Message":
        """Create a reply to this message."""
        return Message(
            sender=sender,
            recipient=self.sender,
            content=content,
            reply_to=self.message_id,
        )


@dataclass
class ScopeContract:
    allowed_outputs: str = ""
    forbidden_outputs: str = ""
    max_responsibility: str = ""
    output_format: str = ""


@dataclass
class Task:
    """A unit of work assigned to a specific agent."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    assigned_to: str = ""
    created_by: str = "coordinator"
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.NORMAL
    result: Optional[str] = None
    partial_output: Optional[str] = None
    pending_question: Optional[str] = None
    pause_reason: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    contract: Optional[ScopeContract] = None
    timestamp: float = field(default_factory=time.time)

    def complete(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result

    def fail(self, reason: str) -> None:
        self.status = TaskStatus.FAILED
        self.result = reason


@dataclass
class Goal:
    """A high-level objective submitted by the user, broken into tasks."""

    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    tasks: list[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    final_result: Optional[str] = None
    schema: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HelpRequest:
    """An agent asking another agent for assistance mid-task."""

    from_agent: str = ""
    to_agent: str = ""
    question: str = ""
    context: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    response: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
