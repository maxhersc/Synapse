"""
Public API for Synapse.
"""

from agents.base import AgentProfile, SynapseAgent
from core.runtime import Runtime
from protocols.message import (
    Artifact,
    ArtifactType,
    CompanyContext,
    Decision,
    DecisionState,
    Department,
    HumanParticipant,
    Organization,
    OrganizationalAgent,
    Permission,
    Priority,
    Project,
    Review,
    ReviewStatus,
    RoleDefinition,
    Task,
    TaskExecutionResult,
    TaskStatus,
    TaskType,
    Workspace,
    WorkspaceEvent,
    WorkspaceSnapshot,
)

__all__ = [
    "AgentProfile",
    "Artifact",
    "ArtifactType",
    "CompanyContext",
    "Decision",
    "DecisionState",
    "Department",
    "HumanParticipant",
    "Organization",
    "OrganizationalAgent",
    "Permission",
    "Priority",
    "Project",
    "Review",
    "ReviewStatus",
    "RoleDefinition",
    "Runtime",
    "SynapseAgent",
    "Task",
    "TaskExecutionResult",
    "TaskStatus",
    "TaskType",
    "Workspace",
    "WorkspaceEvent",
    "WorkspaceSnapshot",
]
__version__ = "0.3.0"
