#!/usr/bin/env python3
"""
Basic organizational workspace example for Synapse.

Run:
    PYTHONPATH=. python3 examples/basic_team.py
"""

from __future__ import annotations

import asyncio

from synapse import (
    AgentProfile,
    ArtifactType,
    CompanyContext,
    Permission,
    Priority,
    Runtime,
    SynapseAgent,
    TaskExecutionResult,
    TaskStatus,
    TaskType,
)


class ProductEngineer(SynapseAgent):
    profile = AgentProfile(
        name="Builder",
        role="Product Engineer",
        department="Engineering",
        strengths=["implementation", "delivery"],
        permissions={Permission.EXECUTE_TASKS, Permission.MANAGE_ARTIFACTS},
        description="Implements scoped product work and records outcomes.",
    )

    async def execute_task(self, task, snapshot) -> TaskExecutionResult:
        artifact_name = f"{task.title} Spec"
        summary = (
            f"Completed '{task.title}' for workspace '{snapshot.workspace.name}'. "
            f"Prepared an implementation artifact aligned with company goals."
        )
        return TaskExecutionResult(
            status=TaskStatus.IN_REVIEW,
            summary=summary,
            artifact_drafts=[
                {
                    "name": artifact_name,
                    "artifact_type": ArtifactType.DOCUMENT,
                    "content": (
                        f"# {artifact_name}\n\n"
                        f"Mission alignment: {snapshot.organization.context.mission}\n\n"
                        f"Task summary:\n{task.description}\n"
                    ),
                }
            ],
        )


async def main() -> None:
    runtime = Runtime()
    runtime.add_agent(ProductEngineer())
    await runtime.run()

    organization = await runtime.create_organization(
        "Acme Systems",
        context=CompanyContext(
            mission="Build reliable internal AI operations infrastructure.",
            goals=[
                "Ship structured workflows for mixed human and AI teams",
                "Keep organizational state observable and auditable",
            ],
            writing_style="Direct and technical",
            communication_style="State changes over free-form chat",
            product_standards=["Traceable execution", "Explicit ownership"],
            design_standards=["Clarity over ornament"],
            domain_knowledge=["Developer tools", "AI operations", "Workflow systems"],
            brand_voice="Precise and operational",
        ),
    )

    engineering = await runtime.add_department(
        organization.organization_id,
        "Engineering",
        "Builds product and workflow systems.",
    )
    engineer_role = await runtime.add_role(
        organization.organization_id,
        "Product Engineer",
        department_id=engineering.department_id,
        description="Owns implementation work inside the workspace.",
        permissions={
            Permission.EXECUTE_TASKS,
            Permission.MANAGE_ARTIFACTS,
            Permission.VIEW_WORKSPACE,
        },
    )
    worker = await runtime.register_worker(
        organization.organization_id,
        "Builder",
        engineer_role.role_id,
        model="gemma3:1b",
        skills=["implementation", "delivery"],
    )

    workspace = await runtime.create_workspace(
        organization.organization_id,
        "Operating System Buildout",
        "Persistent company workspace for Synapse platform work.",
    )
    project = await runtime.create_project(
        workspace.workspace_id,
        "Workspace Foundation",
        description="First pass at the organization-native runtime model.",
        owner_id=worker.agent_id,
    )
    task = await runtime.create_task(
        workspace.workspace_id,
        "Define workspace execution model",
        description=(
            "Create the initial implementation shape for organization, workspace, "
            "task, artifact, review, and decision primitives."
        ),
        task_type=TaskType.IMPLEMENTATION,
        priority=Priority.HIGH,
        project_id=project.project_id,
        assigned_role_id=engineer_role.role_id,
        created_by="system",
    )

    await runtime.assign_task(workspace.workspace_id, task.task_id, worker.agent_id, "system")
    await runtime.execute_task(workspace.workspace_id, task.task_id)

    snapshot = await runtime.memory.get_workspace_snapshot(workspace.workspace_id)
    if snapshot is None:
        raise RuntimeError("workspace snapshot unavailable")

    print(f"Organization: {snapshot.organization.name}")
    print(f"Workspace: {snapshot.workspace.name}")
    print(f"Projects: {[project.name for project in snapshot.projects]}")
    print(f"Tasks: {[(item.title, item.status.value) for item in snapshot.tasks]}")
    print(f"Artifacts: {[(artifact.name, artifact.artifact_type.value) for artifact in snapshot.artifacts]}")
    print("Recent events:")
    for event in snapshot.events[-5:]:
        print(f" - {event.event_type.value}: {event.summary}")

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
