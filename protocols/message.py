"""
Core protocol objects for Synapse.

The primary abstractions model a persistent organizational workspace:
organizations, roles, agents, projects, tasks, artifacts, decisions,
reviews, and auditable workspace events.

Legacy research-oriented types remain available as compatibility shims
while the repository transitions to the workspace model.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _id() -> str:
    return uuid.uuid4().hex[:12]


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


class ActorType(Enum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


class Permission(Enum):
    MANAGE_ORGANIZATION = "manage_organization"
    MANAGE_ROLES = "manage_roles"
    MANAGE_WORKFLOWS = "manage_workflows"
    MANAGE_PROJECTS = "manage_projects"
    MANAGE_TASKS = "manage_tasks"
    ASSIGN_TASKS = "assign_tasks"
    EXECUTE_TASKS = "execute_tasks"
    REVIEW_WORK = "review_work"
    APPROVE_WORK = "approve_work"
    MANAGE_ARTIFACTS = "manage_artifacts"
    RECORD_DECISIONS = "record_decisions"
    VIEW_WORKSPACE = "view_workspace"


class TaskType(Enum):
    GENERAL = "general"
    DELIVERY = "delivery"
    REVIEW = "review"
    APPROVAL = "approval"
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    DESIGN = "design"
    OPERATIONS = "operations"


class TaskStatus(Enum):
    CREATED = "created"
    AVAILABLE = "available"
    CLAIMED = "claimed"
    BACKLOG = "backlog"
    READY = "ready"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    APPROVED = "approved"
    COMPLETED = "completed"
    CANCELED = "canceled"


class ArtifactType(Enum):
    DOCUMENT = "document"
    CODE = "code"
    DESIGN = "design"
    PLAN = "plan"
    API_SPEC = "api_spec"
    DATASET = "dataset"
    OTHER = "other"


class ReviewStatus(Enum):
    PENDING = "pending"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    REJECTED = "rejected"


class DecisionState(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EventType(Enum):
    ORGANIZATION_CREATED = "organization_created"
    ROLE_CREATED = "role_created"
    WORKSPACE_CREATED = "workspace_created"
    AGENT_REGISTERED = "agent_registered"
    PROJECT_CREATED = "project_created"
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_ASSIGNED = "task_assigned"
    TASK_STATUS_CHANGED = "task_status_changed"
    TASK_EXECUTION_RECORDED = "task_execution_recorded"
    ARTIFACT_CREATED = "artifact_created"
    REVIEW_CREATED = "review_created"
    REVIEW_UPDATED = "review_updated"
    DECISION_RECORDED = "decision_recorded"
    AUTONOMY_CYCLE = "autonomy_cycle"


class ClaimStatus(Enum):
    VERIFIED = "verified"
    DISPUTED = "disputed"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


@dataclass
class Message:
    """Structured message for coordination or notifications."""

    sender: str
    recipient: str
    content: str
    message_id: str = field(default_factory=_id)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reply(self, sender: str, content: str) -> "Message":
        return Message(
            sender=sender,
            recipient=self.sender,
            content=content,
            reply_to=self.message_id,
        )


@dataclass
class CompanyContext:
    mission: str = ""
    goals: list[str] = field(default_factory=list)
    writing_style: str = ""
    communication_style: str = ""
    product_standards: list[str] = field(default_factory=list)
    design_standards: list[str] = field(default_factory=list)
    domain_knowledge: list[str] = field(default_factory=list)
    brand_voice: str = ""


@dataclass
class Department:
    department_id: str = field(default_factory=_id)
    name: str = ""
    description: str = ""
    parent_department_id: Optional[str] = None


@dataclass
class RoleDefinition:
    role_id: str = field(default_factory=_id)
    name: str = ""
    department_id: Optional[str] = None
    description: str = ""
    permissions: set[Permission] = field(default_factory=set)
    decision_rules: list[str] = field(default_factory=list)
    communication_standards: list[str] = field(default_factory=list)


@dataclass
class OrganizationalAgent:
    agent_id: str = field(default_factory=_id)
    name: str = ""
    role_id: str = ""
    department_id: Optional[str] = None
    model: str = ""
    skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    permissions: set[Permission] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanParticipant:
    participant_id: str = field(default_factory=_id)
    name: str = ""
    role_title: str = ""
    permissions: set[Permission] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    project_id: str = field(default_factory=_id)
    name: str = ""
    description: str = ""
    owner_id: Optional[str] = None
    status: TaskStatus = TaskStatus.READY
    task_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str = field(default_factory=_id)
    title: str = ""
    description: str = ""
    task_type: TaskType = TaskType.GENERAL
    status: TaskStatus = TaskStatus.CREATED
    priority: Priority = Priority.NORMAL
    project_id: Optional[str] = None
    assigned_role_id: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    created_by: str = "system"
    dependencies: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    review_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    watchers: list[str] = field(default_factory=list)
    required_approvals: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    status_reason: str = ""
    transition_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class Artifact:
    artifact_id: str = field(default_factory=_id)
    name: str = ""
    artifact_type: ArtifactType = ArtifactType.OTHER
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    created_by: str = "system"
    version: int = 1
    review_state: ReviewStatus = ReviewStatus.PENDING
    uri: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Review:
    review_id: str = field(default_factory=_id)
    task_id: str = ""
    artifact_id: Optional[str] = None
    reviewer_id: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    summary: str = ""
    requested_changes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Decision:
    decision_id: str = field(default_factory=_id)
    title: str = ""
    summary: str = ""
    made_by: str = "system"
    state: DecisionState = DecisionState.PROPOSED
    related_task_ids: list[str] = field(default_factory=list)
    related_artifact_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class WorkspaceEvent:
    event_id: str = field(default_factory=_id)
    workspace_id: str = ""
    event_type: EventType = EventType.TASK_UPDATED
    actor_id: str = "system"
    actor_type: ActorType = ActorType.SYSTEM
    entity_type: str = ""
    entity_id: str = ""
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Workspace:
    workspace_id: str = field(default_factory=_id)
    organization_id: str = ""
    name: str = ""
    description: str = ""
    project_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    review_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Organization:
    organization_id: str = field(default_factory=_id)
    name: str = ""
    context: CompanyContext = field(default_factory=CompanyContext)
    departments: dict[str, Department] = field(default_factory=dict)
    roles: dict[str, RoleDefinition] = field(default_factory=dict)
    workflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    decision_rules: list[str] = field(default_factory=list)
    communication_standards: list[str] = field(default_factory=list)
    brand_identity: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class TaskExecutionResult:
    status: TaskStatus
    summary: str
    artifact_drafts: list[dict[str, Any]] = field(default_factory=list)
    requested_reviews: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    spawned_tasks: list[dict[str, Any]] = field(default_factory=list)
    decision_drafts: list[dict[str, Any]] = field(default_factory=list)
    outbound_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkspaceSnapshot:
    organization: Organization
    workspace: Workspace
    projects: list[Project]
    tasks: list[Task]
    artifacts: list[Artifact]
    decisions: list[Decision]
    reviews: list[Review]
    events: list[WorkspaceEvent]
    agents: list[OrganizationalAgent]
    humans: list[HumanParticipant]


# Legacy compatibility shims.
@dataclass
class Evidence:
    quote: str
    source: str
    location: str
    retrieved_by: str
    confidence_score: float


@dataclass
class Claim:
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
    operation_id: str = field(default_factory=_id)
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
    id: str = field(default_factory=_id)
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
    from_agent: str = ""
    to_agent: str = ""
    question: str = ""
    context: str = ""
    request_id: str = field(default_factory=_id)
    response: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
