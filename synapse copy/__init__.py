"""
Synapse v0.2 — Public API.

    from synapse import Runtime, Goal, SynapseAgent, AgentProfile
"""

from core.runtime import Runtime
from protocols.message import Goal
from agents.base import SynapseAgent, AgentProfile

__all__ = ["Runtime", "Goal", "SynapseAgent", "AgentProfile"]
__version__ = "0.2.0"
