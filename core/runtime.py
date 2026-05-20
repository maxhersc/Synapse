"""
Synapse runtime for AI-native organizational work.

The runtime owns the message bus, shared workspace store, registered
participants, and high-level workflow operations that mutate persistent
workspace state.
"""

from __future__ import annotations

import asyncio
import threading
import time
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
    WorkspaceSnapshot,
)


class Runtime:
    """Primary entry point for building a digital organization in Synapse."""

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.memory = SharedMemory()
        self._agents: dict[str, SynapseAgent] = {}
        self._display: Callable[[str, str], Awaitable[None]] | None = None
        self._autonomy_interval_seconds = 2.0
        self._autonomy_stop = threading.Event()
        self._autonomy_thread: Optional[threading.Thread] = None

    def add_agent(self, agent: SynapseAgent) -> None:
        """Register an executable agent instance with the runtime."""
        self._agents[agent.name] = agent
        self.bus.register(agent)

    def _required_permission_for_task(self, task: Task) -> Permission:
        if task.task_type == TaskType.REVIEW:
            return Permission.REVIEW_WORK
        if task.task_type == TaskType.APPROVAL:
            return Permission.APPROVE_WORK
        return Permission.EXECUTE_TASKS

    async def _eligible_agents_for_task(
        self,
        task: Task,
        organization: Organization,
        snapshot: Optional["WorkspaceSnapshot"] = None,
    ) -> list[OrganizationalAgent]:
        required_permission = self._required_permission_for_task(task)
        candidates: list[OrganizationalAgent] = []
        for agent in self.memory.agents.values():
            if task.assigned_role_id and agent.role_id != task.assigned_role_id:
                continue
            if required_permission not in agent.permissions:
                continue
            role = organization.roles.get(agent.role_id)
            if role and required_permission not in role.permissions and required_permission not in agent.permissions:
                continue
            runtime_agent = self._agents.get(agent.name.lower())
            if runtime_agent is None:
                continue
            if snapshot is not None and not runtime_agent.can_accept_task(task, snapshot):
                continue
            candidates.append(agent)

        def score(agent: OrganizationalAgent) -> tuple[int, int, str]:
            direct_role = 1 if task.assigned_role_id and agent.role_id == task.assigned_role_id else 0
            skill_hits = sum(
                1
                for term in (task.task_type.value, task.title.lower(), task.description.lower())
                if any(skill.lower() in term for skill in agent.skills)
            )
            return (direct_role, skill_hits, agent.name)

        return sorted(candidates, key=score, reverse=True)

    def _dependencies_completed(self, task: Task) -> bool:
        for dependency_id in task.dependencies:
            dependency = self.memory.tasks.get(dependency_id)
            if dependency is None or dependency.status != TaskStatus.COMPLETED:
                return False
        return True

    def _allowed_transitions(self) -> dict[TaskStatus, set[TaskStatus]]:
        return {
            TaskStatus.CREATED: {TaskStatus.AVAILABLE, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.AVAILABLE: {TaskStatus.CLAIMED, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.CLAIMED: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.REVIEW: {TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.BLOCKED: {TaskStatus.AVAILABLE, TaskStatus.CLAIMED, TaskStatus.CANCELED},
            TaskStatus.BACKLOG: {TaskStatus.CREATED, TaskStatus.AVAILABLE, TaskStatus.CANCELED},
            TaskStatus.READY: {TaskStatus.AVAILABLE, TaskStatus.CLAIMED, TaskStatus.CANCELED},
            TaskStatus.ASSIGNED: {TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS, TaskStatus.CANCELED},
            TaskStatus.IN_REVIEW: {TaskStatus.REVIEW, TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELED},
            TaskStatus.APPROVED: {TaskStatus.COMPLETED, TaskStatus.BLOCKED},
            TaskStatus.COMPLETED: set(),
            TaskStatus.CANCELED: set(),
        }

    async def _transition_task(
        self,
        workspace_id: str,
        task_id: str,
        status: TaskStatus,
        actor_id: str,
        note: str = "",
    ) -> Task:
        workspace = self.memory.workspaces[workspace_id]
        task = self.memory.tasks[task_id]
        if task.status != status:
            allowed = self._allowed_transitions().get(task.status, set())
            if status not in allowed:
                raise ValueError(f"Invalid transition for task '{task.title}': {task.status.value} -> {status.value}")
            task.transition_history.append(
                {
                    "from": task.status.value,
                    "to": status.value,
                    "actor_id": actor_id,
                    "note": note,
                    "timestamp": time.time(),
                }
            )
        task.status = status
        task.status_reason = note
        if note:
            task.notes.append(note)
        if status == TaskStatus.COMPLETED:
            task.completed_at = time.time()
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
            status=TaskStatus.CREATED,
            status_reason="Waiting for coordination layer",
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
        task = self.memory.tasks[task_id]
        task.assigned_agent_id = agent_id
        note = f"Assigned to {self.memory.agents[agent_id].name}"
        await self.memory.save_task(task)
        workspace = self.memory.workspaces[workspace_id]
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
        return await self._transition_task(workspace_id, task_id, TaskStatus.CLAIMED, actor_id, note)

    async def update_task_status(
        self,
        workspace_id: str,
        task_id: str,
        status: TaskStatus,
        actor_id: str,
        note: str = "",
    ) -> Task:
        return await self._transition_task(workspace_id, task_id, status, actor_id, note)

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
        task = self.memory.tasks[task_id]
        if not task.assigned_agent_id:
            raise ValueError(f"Task '{task.title}' is not assigned to an agent")
        if not self._dependencies_completed(task):
            raise ValueError(f"Task '{task.title}' still has unresolved dependencies")

        agent_record = self.memory.agents[task.assigned_agent_id]
        runtime_agent = self._agents.get(agent_record.name.lower())
        if runtime_agent is None:
            raise ValueError(f"No executable agent registered for '{agent_record.name}'")

        await self._transition_task(
            workspace_id,
            task_id,
            TaskStatus.IN_PROGRESS,
            task.assigned_agent_id,
            f"{agent_record.name} began execution",
        )

        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            raise ValueError(f"Workspace '{workspace_id}' not found")

        workspace = self.memory.workspaces[workspace_id]
        result = await runtime_agent.execute_task(task, snapshot)
        await self._apply_execution_result(workspace, task, task.assigned_agent_id, result)
        return self.memory.tasks[task_id]

    async def run_autonomy_cycle(self, workspace_id: str, max_passes: int = 8) -> dict[str, object]:
        if workspace_id not in self.memory.workspaces:
            raise ValueError(f"Workspace '{workspace_id}' not found")

        actions: list[str] = []
        passes = 0
        for _ in range(max_passes):
            changed = False
            passes += 1

            if await self._publish_available_tasks(workspace_id, actions):
                changed = True
            if await self._agent_pull_tasks(workspace_id, actions):
                changed = True
            if await self._auto_execute_tasks(workspace_id, actions):
                changed = True
            if await self._auto_process_reviews(workspace_id, actions):
                changed = True
            if await self._advance_review_state(workspace_id, actions):
                changed = True

            if not changed:
                break

        workspace = self.memory.workspaces[workspace_id]
        await self._record_event(
            workspace,
            EventType.AUTONOMY_CYCLE,
            actor_id="system",
            actor_type=ActorType.SYSTEM,
            entity_type="workspace",
            entity_id=workspace_id,
            summary=f"Autonomy cycle ran {passes} pass(es)",
            payload={"actions": actions},
        )
        return {"passes": passes, "actions": actions}

    async def run_all_workspaces_cycle(self, max_passes: int = 8) -> dict[str, object]:
        results: dict[str, object] = {}
        for workspace_id in list(self.memory.workspaces.keys()):
            results[workspace_id] = await self.run_autonomy_cycle(workspace_id, max_passes=max_passes)
        return results

    def start_autonomy_service(self, interval_seconds: float = 2.0) -> None:
        self._autonomy_interval_seconds = interval_seconds
        if self._autonomy_thread and self._autonomy_thread.is_alive():
            return
        self._autonomy_stop.clear()
        self._autonomy_thread = threading.Thread(
            target=self._autonomy_worker,
            name="synapse-autonomy",
            daemon=True,
        )
        self._autonomy_thread.start()

    def stop_autonomy_service(self) -> None:
        self._autonomy_stop.set()
        if self._autonomy_thread and self._autonomy_thread.is_alive():
            self._autonomy_thread.join(timeout=1.0)

    def _autonomy_worker(self) -> None:
        while not self._autonomy_stop.is_set():
            try:
                asyncio.run(self.run_all_workspaces_cycle(max_passes=4))
            except Exception:
                pass
            self._autonomy_stop.wait(self._autonomy_interval_seconds)

    async def _publish_available_tasks(self, workspace_id: str, actions: list[str]) -> bool:
        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return False

        changed = False
        for task in snapshot.tasks:
            if task.status not in {TaskStatus.CREATED, TaskStatus.BACKLOG, TaskStatus.READY, TaskStatus.BLOCKED}:
                continue
            if not self._dependencies_completed(task):
                continue
            if task.status != TaskStatus.AVAILABLE:
                await self._transition_task(
                    workspace_id,
                    task.task_id,
                    TaskStatus.AVAILABLE,
                    "system",
                    "Task became runnable",
                )
                actions.append(f"available:{task.title}")
                changed = True
        return changed

    async def _agent_pull_tasks(self, workspace_id: str, actions: list[str]) -> bool:
        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return False

        organization = snapshot.organization
        active_counts: dict[str, int] = {}
        for task in snapshot.tasks:
            if task.assigned_agent_id and task.status in {TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS, TaskStatus.REVIEW}:
                active_counts[task.assigned_agent_id] = active_counts.get(task.assigned_agent_id, 0) + 1

        changed = False
        available_tasks = [
            task for task in snapshot.tasks
            if task.status == TaskStatus.AVAILABLE and not task.assigned_agent_id and self._dependencies_completed(task)
        ]
        available_tasks.sort(key=lambda task: (task.priority.value, task.created_at), reverse=True)

        for agent_record in self.memory.agents.values():
            if active_counts.get(agent_record.agent_id, 0) > 0:
                continue
            runtime_agent = self._agents.get(agent_record.name.lower())
            if runtime_agent is None:
                continue
            chosen_task: Optional[Task] = None
            for task in available_tasks:
                if task.assigned_agent_id:
                    continue
                if task.assigned_role_id and task.assigned_role_id != agent_record.role_id:
                    continue
                if not runtime_agent.can_accept_task(task, snapshot):
                    continue
                candidates = await self._eligible_agents_for_task(task, organization, snapshot)
                if candidates and candidates[0].agent_id == agent_record.agent_id:
                    chosen_task = task
                    break
            if chosen_task is None:
                continue
            await self.assign_task(workspace_id, chosen_task.task_id, agent_record.agent_id, agent_record.agent_id)
            chosen_task.assigned_agent_id = agent_record.agent_id
            actions.append(f"claimed:{chosen_task.title}:{agent_record.name}")
            changed = True
        return changed

    async def _auto_execute_tasks(self, workspace_id: str, actions: list[str]) -> bool:
        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return False

        changed = False
        for task in snapshot.tasks:
            if task.status not in {TaskStatus.CLAIMED, TaskStatus.ASSIGNED}:
                continue
            if not task.assigned_agent_id or not self._dependencies_completed(task):
                continue
            await self.execute_task(workspace_id, task.task_id)
            actions.append(f"executed:{task.title}")
            changed = True
        return changed

    async def _advance_review_state(self, workspace_id: str, actions: list[str]) -> bool:
        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return False

        changed = False
        for task in snapshot.tasks:
            if task.status not in {TaskStatus.REVIEW, TaskStatus.IN_REVIEW}:
                continue
            reviews = [review for review in snapshot.reviews if review.review_id in task.review_ids]
            if any(review.status == ReviewStatus.CHANGES_REQUESTED for review in reviews):
                await self._transition_task(workspace_id, task.task_id, TaskStatus.BLOCKED, "system", "Review requested changes")
                actions.append(f"blocked:{task.title}")
                changed = True
                continue
            approvals = sum(1 for review in reviews if review.status == ReviewStatus.APPROVED)
            required = task.required_approvals or (1 if reviews else 0)
            if approvals >= required and required > 0:
                await self._transition_task(workspace_id, task.task_id, TaskStatus.COMPLETED, "system", "Required reviews completed")
                actions.append(f"completed:{task.title}")
                changed = True
        return changed

    async def _auto_process_reviews(self, workspace_id: str, actions: list[str]) -> bool:
        snapshot = await self.memory.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return False

        changed = False
        tasks_by_id = {task.task_id: task for task in snapshot.tasks}
        for review in snapshot.reviews:
            if review.status != ReviewStatus.PENDING:
                continue
            task = tasks_by_id.get(review.task_id)
            if task is None or task.status not in {TaskStatus.REVIEW, TaskStatus.IN_REVIEW}:
                continue
            reviewer = self.memory.agents.get(review.reviewer_id)
            status = ReviewStatus.APPROVED
            summary = "Autonomous review completed."
            if task.context.get("force_review_changes"):
                status = ReviewStatus.CHANGES_REQUESTED
                summary = "Autonomous review flagged required changes."
            await self.update_review(
                workspace_id,
                review.review_id,
                status,
                review.reviewer_id or "system",
                summary=summary if reviewer is None else f"{reviewer.name}: {summary}",
            )
            actions.append(
                f"reviewed:{task.title}:{reviewer.name if reviewer else 'system'}:{status.value}"
            )
            changed = True
        return changed

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
        self.stop_autonomy_service()
        for agent in self._agents.values():
            await agent.stop()

    async def run(self) -> None:
        await self.start()
        self.start_autonomy_service()

    async def _apply_execution_result(
        self,
        workspace: Workspace,
        task: Task,
        agent_id: str,
        result: TaskExecutionResult,
    ) -> None:
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

        for draft in result.spawned_tasks:
            task_type = draft.get("task_type", TaskType.GENERAL)
            priority = draft.get("priority", Priority.NORMAL)
            if isinstance(task_type, str):
                task_type = TaskType(task_type)
            if isinstance(priority, str):
                priority = Priority(priority)
            created_task = await self.create_task(
                workspace.workspace_id,
                draft["title"],
                description=draft.get("description", ""),
                task_type=task_type,
                priority=priority,
                project_id=draft.get("project_id", task.project_id),
                assigned_role_id=draft.get("assigned_role_id"),
                created_by=agent_id,
                dependencies=draft.get("dependencies"),
                required_approvals=draft.get("required_approvals", 0),
                context=draft.get("context"),
            )
            task.context.setdefault("spawned_task_ids", []).append(created_task.task_id)

        for draft in result.decision_drafts:
            state = draft.get("state", DecisionState.PROPOSED)
            if isinstance(state, str):
                state = DecisionState(state)
            decision = await self.record_decision(
                workspace.workspace_id,
                draft["title"],
                summary=draft.get("summary", ""),
                made_by=agent_id,
                state=state,
                related_task_ids=draft.get("related_task_ids", [task.task_id]),
                related_artifact_ids=draft.get("related_artifact_ids", []),
                rationale=draft.get("rationale", []),
            )
            task.decision_ids.append(decision.decision_id)

        for escalation in result.escalations:
            task.notes.append(f"Escalation: {escalation}")

        final_status = result.status
        if result.escalations and final_status not in {TaskStatus.BLOCKED, TaskStatus.CANCELED}:
            final_status = TaskStatus.BLOCKED
        elif result.requested_reviews or task.required_approvals > 0:
            if final_status not in {TaskStatus.BLOCKED, TaskStatus.CANCELED, TaskStatus.REVIEW, TaskStatus.IN_REVIEW}:
                final_status = TaskStatus.REVIEW
            elif final_status == TaskStatus.IN_REVIEW:
                final_status = TaskStatus.REVIEW
        elif final_status == TaskStatus.APPROVED:
            final_status = TaskStatus.COMPLETED

        await self.memory.save_task(task)
        await self._transition_task(workspace.workspace_id, task.task_id, final_status, agent_id, result.summary)
        await self._record_event(
            workspace,
            EventType.TASK_EXECUTION_RECORDED,
            actor_id=agent_id,
            actor_type=ActorType.AGENT,
            entity_type="task",
            entity_id=task.task_id,
            summary=result.summary,
            payload={
                "status": final_status.value,
                "requested_reviews": result.requested_reviews,
                "escalations": result.escalations,
                "spawned_tasks": len(result.spawned_tasks),
                "decision_drafts": len(result.decision_drafts),
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
