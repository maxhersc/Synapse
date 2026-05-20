"""
Persistent in-memory workspace store for Synapse.

The store tracks organizations, participants, workspaces, and all
workspace entities so runtime operations can mutate shared state while
keeping an auditable event history.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

from protocols.message import (
    Artifact,
    Decision,
    HumanParticipant,
    Organization,
    OrganizationalAgent,
    Project,
    Review,
    Task,
    Workspace,
    WorkspaceEvent,
    WorkspaceSnapshot,
)


class SharedMemory:
    """Async-safe state store for organizational workspaces."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.organizations: dict[str, Organization] = {}
        self.workspaces: dict[str, Workspace] = {}
        self.projects: dict[str, Project] = {}
        self.tasks: dict[str, Task] = {}
        self.artifacts: dict[str, Artifact] = {}
        self.decisions: dict[str, Decision] = {}
        self.reviews: dict[str, Review] = {}
        self.events: dict[str, WorkspaceEvent] = {}
        self.agents: dict[str, OrganizationalAgent] = {}
        self.humans: dict[str, HumanParticipant] = {}
        self._store: dict[str, Any] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = value

    async def append(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store.setdefault(key, []).append(value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())

    async def all(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._store)

    async def clear(self) -> None:
        async with self._lock:
            self.organizations.clear()
            self.workspaces.clear()
            self.projects.clear()
            self.tasks.clear()
            self.artifacts.clear()
            self.decisions.clear()
            self.reviews.clear()
            self.events.clear()
            self.agents.clear()
            self.humans.clear()
            self._store.clear()

    async def has(self, key: str) -> bool:
        async with self._lock:
            return key in self._store

    async def save_organization(self, organization: Organization) -> Organization:
        async with self._lock:
            organization.updated_at = time.time()
            self.organizations[organization.organization_id] = organization
            return organization

    async def save_workspace(self, workspace: Workspace) -> Workspace:
        async with self._lock:
            workspace.updated_at = time.time()
            self.workspaces[workspace.workspace_id] = workspace
            return workspace

    async def save_project(self, project: Project) -> Project:
        async with self._lock:
            self.projects[project.project_id] = project
            return project

    async def save_task(self, task: Task) -> Task:
        async with self._lock:
            task.updated_at = time.time()
            self.tasks[task.task_id] = task
            return task

    async def save_artifact(self, artifact: Artifact) -> Artifact:
        async with self._lock:
            artifact.updated_at = time.time()
            self.artifacts[artifact.artifact_id] = artifact
            return artifact

    async def save_decision(self, decision: Decision) -> Decision:
        async with self._lock:
            decision.updated_at = time.time()
            self.decisions[decision.decision_id] = decision
            return decision

    async def save_review(self, review: Review) -> Review:
        async with self._lock:
            review.updated_at = time.time()
            self.reviews[review.review_id] = review
            return review

    async def save_event(self, event: WorkspaceEvent) -> WorkspaceEvent:
        async with self._lock:
            self.events[event.event_id] = event
            return event

    async def save_agent(self, agent: OrganizationalAgent) -> OrganizationalAgent:
        async with self._lock:
            self.agents[agent.agent_id] = agent
            return agent

    async def save_human(self, human: HumanParticipant) -> HumanParticipant:
        async with self._lock:
            self.humans[human.participant_id] = human
            return human

    async def get_workspace_snapshot(
        self,
        workspace_id: str,
    ) -> Optional[WorkspaceSnapshot]:
        async with self._lock:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return None

            organization = self.organizations[workspace.organization_id]
            return WorkspaceSnapshot(
                organization=organization,
                workspace=workspace,
                projects=[
                    self.projects[project_id]
                    for project_id in workspace.project_ids
                    if project_id in self.projects
                ],
                tasks=[
                    self.tasks[task_id]
                    for task_id in workspace.task_ids
                    if task_id in self.tasks
                ],
                artifacts=[
                    self.artifacts[artifact_id]
                    for artifact_id in workspace.artifact_ids
                    if artifact_id in self.artifacts
                ],
                decisions=[
                    self.decisions[decision_id]
                    for decision_id in workspace.decision_ids
                    if decision_id in self.decisions
                ],
                reviews=[
                    self.reviews[review_id]
                    for review_id in workspace.review_ids
                    if review_id in self.reviews
                ],
                events=[
                    self.events[event_id]
                    for event_id in workspace.event_ids
                    if event_id in self.events
                ],
                agents=list(self.agents.values()),
                humans=list(self.humans.values()),
            )

    async def export_workspace(self, workspace_id: str) -> dict[str, Any]:
        snapshot = await self.get_workspace_snapshot(workspace_id)
        if snapshot is None:
            return {}
        return self._json_safe(asdict(snapshot))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, set):
            return sorted(self._json_safe(item) for item in value)
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        return value
