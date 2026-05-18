"""
Synapse v0.3 — Protocol definitions.

All message types that flow between agents: Messages, Research, ResearchOperation,
HelpRequests, and their associated enums (Priority, NodeStatus).
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


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED_PENDING_INPUT = "blocked_pending_input"
    COMPLETED = "completed"
    FAILED = "failed"


class ClaimStatus(Enum):
    VERIFIED = "verified"
    DISPUTED = "disputed"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


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
class Evidence:
    """Evidence is directly bound to claims, inline."""
    quote: str
    source: str
    location: str
    retrieved_by: str
    confidence_score: float


@dataclass
class Claim:
    """Structured claim backed by inline evidence."""
    claim: str
    evidence: list[Evidence] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    supporting_sources: list[str] = field(default_factory=list)
    contradicting_sources: list[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class ScopeContract:
    allowed_outputs: str = ""
    forbidden_outputs: str = ""
    max_responsibility: str = ""
    output_format: str = ""


@dataclass
class ResearchOperation:
    """A node in the DAG representing a research operation."""

    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    assigned_to: str = ""
    created_by: str = "coordinator"
    status: NodeStatus = NodeStatus.PENDING
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
        self.status = NodeStatus.COMPLETED
        self.result = result

    def fail(self, reason: str) -> None:
        self.status = NodeStatus.FAILED
        self.result = reason


@dataclass
class Research:
    """Primary abstraction representing a structured research process."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    plan: list[ResearchOperation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    final_output: Optional[str] = None
    schema: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.PENDING


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
