from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Node:
    """Executable DAG node that binds an agent to dependencies and retry policy."""

    id: str
    agent: str
    depends_on: list[str] = field(default_factory=list)
    timeout: float | None = None
    retries: int = 0


@dataclass
class NodeExecution:
    """Execution metadata recorded for a DAG node run."""

    node_id: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: str = "pending"
    retry_count: int = 0
