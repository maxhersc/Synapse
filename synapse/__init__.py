"""Synapse is a coordination layer for multi-agent AI systems. It provides the core developer-facing primitives for defining agents, goals, tasks, and runtime execution."""

from synapse.agents.base import AgentProfile, SynapseAgent
from synapse.core.runtime import Runtime
from synapse.protocols.message import Goal, HelpRequest, Message, Priority, Task, TaskStatus

__version__ = "0.1.0"

__all__ = [
    "AgentProfile",
    "Goal",
    "HelpRequest",
    "Message",
    "Priority",
    "Runtime",
    "SynapseAgent",
    "Task",
    "TaskStatus",
    "__version__",
]
