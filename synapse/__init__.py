"""
Synapse v0.2 — Public API.

    from synapse import Runtime, Research, SynapseAgent, AgentProfile
"""

from core.runtime import Runtime
from protocols.message import Research
from agents.base import SynapseAgent, AgentProfile

__all__ = ["Runtime", "Research", "SynapseAgent", "AgentProfile"]
__version__ = "0.2.0"
