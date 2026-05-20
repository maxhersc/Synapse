"""
Synapse runtime for AI-native organizational work.

The runtime owns the message bus, shared workspace store, registered
participants, and high-level workflow operations that mutate persistent
workspace state.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from agents.base import SynapseAgent
from core.bus import MessageBus
from core.memory import SharedMemory
from protocols.message import (
    ActorType,
    Artifact,
    ArtifactType,
    CompanyContext,
    Decision,
    DecisionState,
    Department,
    EventType,
    HumanParticipant,
    Message,
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
)


class Runtime:
    """Primary entry point for building a digital organization in Synapse."""

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.memory = SharedMemory()
        self._agents: dict[str, SynapseAgent] = {}
        self._display: Callable[[str, str], Awaitable[None]] | None = None

    def add_agent(self, agent: SynapseAgent) -> None:
        """Register an executable agent instance with the runtime."""
        self._agents[agent.name] = agent
        self.bus.register(agent)

    async def create_organization(
        self,
        name: str,
        *,
        context: Optional[CompanyContext] = None,
        decision_rules: Optional[list[str]] = None,
        communication_standards: Optional[list[str]] = None,
        brand_identity: Optional[dict[str, str]] = None,
    ) -> Organization:
        organization = Organization(
            name=name,
            context=context or CompanyContext(),
            decision_rules=decision_rules or [],
            communication_standards=communication_standards or [],
            brand_identity=brand_identity or {},
        )
        await self.memory.save_organization(organization)
        return organization

    async def add_department(
        self,
        organization_id: str,
        name: str,
        description: str = "",
        parent_department_id: str | None = None,
    ) -> Department:
        organization = self.memory.organizations[organization_id]
        department = Department(
            name=name,
            description=description,
            parent_department_id=parent_department_id,
        )
        organization.departments[department.department_id] = department
        await self.memory.save_organization(organization)
        return department

    async def add_role(
        self,
        organization_id: str,
        name: str,
        *,
        department_id: str | None = None,
        description: str = "",
        permissions: Optional[set[Permission]] = None,
        decision_rules: Optional[list[str]] = None,
        communication_standards: Optional[list[str]] = None,
    ) -> RoleDefinition:
        organization = self.memory.organizations[organization_id]
        role = RoleDefinition(
            name=name,
            department_id=department_id,
            description=description,
            permissions=permissions or set(),
            decision_rules=decision_rules or [],
            communication_standards=communication_standards or [],
        )
        organization.roles[role.role_id] = role
        await self.memory.save_organization(organization)
        return role

    async def register_worker(
        self,
        organization_id: str,
        name: str,
        role_id: str,
        *,
        department_id: str | None = None,
        model: str = "",
        skills: Optional[list[str]] = None,
        responsibilities: Optional[list[str]] = None,
        permissions: Optional[set[Permission]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> OrganizationalAgent:
        organization = self.memory.organizations[organization_id]
        role = organization.roles[role_id]
        worker = OrganizationalAgent(
            name=name,
            role_id=role_id,
            department_id=department_id or role.department_id,
            model=model,
            skills=skills or [],
            responsibilities=responsibilities or [],
            permissions=permissions or set(role.permissions),
            metadata=metadata or {},
        )
        await self.memory.save_agent(worker)
        return worker

    async def register_human(
        self,
        name: str,
        role_title: str,
        permissions: Optional[set[Permission]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> HumanParticipant:
        human = HumanParticipant(
            name=name,
            role_title=role_title,
            permissions=permissions or set(),
            metadata=metadata or {},
        )
        await self.memory.save_human(human)
        return human

    async def create_workspace(
        self,
        organization_id: str,
        name: str,
        description: str = "",
        metadata: Optional[dict[str, str]] = None,
    ) -> Workspace:
        workspace = Workspace(
            organization_id=organization_id,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.WORKSPACE_CREATED,
            actor_id="system",
            actor_type=ActorType.SYSTEM,
            entity_type="workspace",
            entity_id=workspace.workspace_id,
            summary=f"Workspace '{workspace.name}' created",
        )
        return workspace

    async def create_project(
        self,
        workspace_id: str,
        name: str,
        *,
        description: str = "",
        owner_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Project:
        workspace = self.memory.workspaces[workspace_id]
        project = Project(
            name=name,
            description=description,
            owner_id=owner_id,
            metadata=metadata or {},
        )
        workspace.project_ids.append(project.project_id)
        await self.memory.save_project(project)
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.PROJECT_CREATED,
            actor_id=owner_id or "system",
            actor_type=ActorType.AGENT if owner_id in self.memory.agents else ActorType.SYSTEM,
            entity_type="project",
            entity_id=project.project_id,
            summary=f"Project '{project.name}' created",
        )
        return project

    async def create_task(
        self,
        workspace_id: str,
        title: str,
        *,
        description: str = "",
        task_type: TaskType = TaskType.GENERAL,
        priority: Priority = Priority.NORMAL,
        project_id: str | None = None,
        assigned_role_id: str | None = None,
        created_by: str = "system",
        dependencies: Optional[list[str]] = None,
        required_approvals: int = 0,
        context: Optional[dict[str, object]] = None,
    ) -> Task:
        workspace = self.memory.workspaces[workspace_id]
        task = Task(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            project_id=project_id,
            assigned_role_id=assigned_role_id,
            created_by=created_by,
            dependencies=dependencies or [],
            required_approvals=required_approvals,
            context=context or {},
            status=TaskStatus.READY if not dependencies else TaskStatus.BACKLOG,
        )
        workspace.task_ids.append(task.task_id)
        if project_id and project_id in self.memory.projects:
            self.memory.projects[project_id].task_ids.append(task.task_id)
        await self.memory.save_task(task)
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.TASK_CREATED,
            actor_id=created_by,
            actor_type=self._actor_type_for_id(created_by),
            entity_type="task",
            entity_id=task.task_id,
            summary=f"Task '{task.title}' created",
            payload={"task_type": task.task_type.value, "priority": task.priority.value},
        )
        return task

    async def assign_task(self, workspace_id: str, task_id: str, agent_id: str, actor_id: str) -> Task:
        workspace = self.memory.workspaces[workspace_id]
        task = self.memory.tasks[task_id]
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.READY
        note = f"Assigned to {self.memory.agents[agent_id].name}"
        task.notes.append(note)
        await self.memory.save_task(task)
        await self._record_event(
            workspace,
            EventType.TASK_ASSIGNED,
            actor_id=actor_id,
            actor_type=self._actor_type_for_id(actor_id),
            entity_type="task",
            entity_id=task.task_id,
            summary=note,
            payload={"assigned_agent_id": agent_id},
        )
        return task

    async def update_task_status(
        self,
        workspace_id: str,
        task_id: str,
        status: TaskStatus,
        actor_id: str,
        note: str = "",
    ) -> Task:
        workspace = self.memory.workspaces[workspace_id]
        task = self.memory.tasks[task_id]
        task.status = status
        if note:
            task.notes.append(note)
        await self.memory.save_task(task)
        await self._record_event(
            workspace,
            EventType.TASK_STATUS_CHANGED,
            actor_id=actor_id,
            actor_type=self._actor_type_for_id(actor_id),
            entity_type="task",
            entity_id=task.task_id,
            summary=note or f"Task status changed to {status.value}",
            payload={"status": status.value},
        )
        return task

    async def create_artifact(
        self,
        workspace_id: str,
        name: str,
        *,
        artifact_type: ArtifactType,
        created_by: str,
        task_id: str | None = None,
        project_id: str | None = None,
        uri: str | None = None,
        content: str | None = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Artifact:
        workspace = self.memory.workspaces[workspace_id]
        artifact = Artifact(
            name=name,
            artifact_type=artifact_type,
            task_id=task_id,
            project_id=project_id,
            created_by=created_by,
            uri=uri,
            content=content,
            metadata=metadata or {},
        )
        workspace.artifact_ids.append(artifact.artifact_id)
        if task_id and task_id in self.memory.tasks:
            self.memory.tasks[task_id].artifact_ids.append(artifact.artifact_id)
        if project_id and project_id in self.memory.projects:
            self.memory.projects[project_id].artifact_ids.append(artifact.artifact_id)
        await self.memory.save_artifact(artifact)
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.ARTIFACT_CREATED,
            actor_id=created_by,
            actor_type=self._actor_type_for_id(created_by),
            entity_type="artifact",
            entity_id=artifact.artifact_id,
            summary=f"Artifact '{artifact.name}' created",
            payload={"artifact_type": artifact.artifact_type.value},
        )
        return artifact

    async def create_review(
        self,
        workspace_id: str,
        task_id: str,
        reviewer_id: str,
        *,
        artifact_id: str | None = None,
        summary: str = "",
    ) -> Review:
        workspace = self.memory.workspaces[workspace_id]
        review = Review(
            task_id=task_id,
            artifact_id=artifact_id,
            reviewer_id=reviewer_id,
            summary=summary,
        )
        workspace.review_ids.append(review.review_id)
        self.memory.tasks[task_id].review_ids.append(review.review_id)
        await self.memory.save_review(review)
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.REVIEW_CREATED,
            actor_id=reviewer_id,
            actor_type=self._actor_type_for_id(reviewer_id),
            entity_type="review",
            entity_id=review.review_id,
            summary=summary or "Review requested",
            payload={"task_id": task_id, "artifact_id": artifact_id},
        )
        return review

    async def update_review(
        self,
        workspace_id: str,
        review_id: str,
        status: ReviewStatus,
        actor_id: str,
        *,
        summary: str = "",
        requested_changes: Optional[list[str]] = None,
    ) -> Review:
        workspace = self.memory.workspaces[workspace_id]
        review = self.memory.reviews[review_id]
        review.status = status
        review.summary = summary or review.summary
        if requested_changes is not None:
            review.requested_changes = requested_changes
        await self.memory.save_review(review)
        await self._record_event(
            workspace,
            EventType.REVIEW_UPDATED,
            actor_id=actor_id,
            actor_type=self._actor_type_for_id(actor_id),
            entity_type="review",
            entity_id=review.review_id,
            summary=review.summary or f"Review marked {review.status.value}",
            payload={"status": review.status.value},
        )
        return review

    async def record_decision(
        self,
        workspace_id: str,
        title: str,
        *,
        summary: str,
        made_by: str,
        state: DecisionState = DecisionState.PROPOSED,
        related_task_ids: Optional[list[str]] = None,
        related_artifact_ids: Optional[list[str]] = None,
        rationale: Optional[list[str]] = None,
    ) -> Decision:
        workspace = self.memory.workspaces[workspace_id]
        decision = Decision(
            title=title,
            summary=summary,
            made_by=made_by,
            state=state,
            related_task_ids=related_task_ids or [],
            related_artifact_ids=related_artifact_ids or [],
            rationale=rationale or [],
        )
        workspace.decision_ids.append(decision.decision_id)
        await self.memory.save_decision(decision)
        await self.memory.save_workspace(workspace)
        await self._record_event(
            workspace,
            EventType.DECISION_RECORDED,
            actor_id=made_by,
            actor_type=self._actor_type_for_id(made_by),
            entity_type="decision",
            entity_id=decision.decision_id,
            summary=f"Decision recorded: {decision.title}",
            payload={"state": decision.state.value},
        )
        return decision

    async def execute_task(self, workspace_id: str, task_id: str) -> Task:
        workspace = self.memory.workspaces[workspace_id]
        task = self.memory.tasks[task_id]
        if not task.assigned_agent_id:
            raise ValueError(f"Task '{task.title}' is not assigned to an agent")

        agent_record = self.memory.agents[task.assigned_agent_id]
        runtime_agent = self._agents.get(agent_record.name.lower())
        if runtime_agent is None:
            raise ValueError(f"No executable agent registered for '{agent_record.name}'")

        task.status = TaskStatus.IN_PROGRESS
        await self.memory.save_task(task)

        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            raise ValueError(f"Workspace '{workspace_id}' not found")

        result = await runtime_agent.execute_task(task, snapshot)
        await self._apply_execution_result(workspace, task, task.assigned_agent_id, result)
        return self.memory.tasks[task_id]

    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        self._display = callback

        async def _observer(msg: Message) -> None:
            if msg.metadata.get("silent"):
                return
            await callback(
                f"{msg.sender} -> {msg.recipient}" if msg.recipient else msg.sender,
                msg.content,
            )

        self.bus.add_observer(_observer)

    async def start(self) -> None:
        for agent in self._agents.values():
            await agent.start()

    async def stop(self) -> None:
        for agent in self._agents.values():
            await agent.stop()

    async def run(self) -> None:
        await self.start()

    async def _apply_execution_result(
        self,
        workspace: Workspace,
        task: Task,
        agent_id: str,
        result: TaskExecutionResult,
    ) -> None:
        task.status = result.status
        task.notes.append(result.summary)
        task.context.update(result.context_updates)
        await self.memory.save_task(task)

        for draft in result.artifact_drafts:
            artifact_type = draft.get("artifact_type", ArtifactType.OTHER)
            if isinstance(artifact_type, str):
                artifact_type = ArtifactType(artifact_type)
            await self.create_artifact(
                workspace.workspace_id,
                draft["name"],
                artifact_type=artifact_type,
                created_by=agent_id,
                task_id=task.task_id,
                project_id=task.project_id,
                uri=draft.get("uri"),
                content=draft.get("content"),
                metadata=draft.get("metadata"),
            )

        for reviewer_id in result.requested_reviews:
            await self.create_review(
                workspace.workspace_id,
                task.task_id,
                reviewer_id,
                summary=f"Review requested by {self.memory.agents[agent_id].name}",
            )

        for escalation in result.escalations:
            task.notes.append(f"Escalation: {escalation}")

        await self.memory.save_task(task)
        await self._record_event(
            workspace,
            EventType.TASK_EXECUTION_RECORDED,
            actor_id=agent_id,
            actor_type=ActorType.AGENT,
            entity_type="task",
            entity_id=task.task_id,
            summary=result.summary,
            payload={
                "status": result.status.value,
                "requested_reviews": result.requested_reviews,
                "escalations": result.escalations,
            },
        )

    async def _record_event(
        self,
        workspace: Workspace,
        event_type: EventType,
        *,
        actor_id: str,
        actor_type: ActorType,
        entity_type: str,
        entity_id: str,
        summary: str,
        payload: Optional[dict[str, object]] = None,
    ) -> WorkspaceEvent:
        event = WorkspaceEvent(
            workspace_id=workspace.workspace_id,
            event_type=event_type,
            actor_id=actor_id,
            actor_type=actor_type,
            entity_type=entity_type,
            entity_id=entity_id,
            summary=summary,
            payload=payload or {},
        )
        workspace.event_ids.append(event.event_id)
        await self.memory.save_event(event)
        await self.memory.save_workspace(workspace)
        return event

    def _actor_type_for_id(self, actor_id: str) -> ActorType:
        if actor_id in self.memory.agents:
            return ActorType.AGENT
        if actor_id in self.memory.humans:
            return ActorType.HUMAN
        return ActorType.SYSTEM
