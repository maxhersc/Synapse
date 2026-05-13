from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    agent_id: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
