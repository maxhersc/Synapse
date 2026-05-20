"""
Minimal Flask server exposing the new workspace-oriented runtime.

Endpoints:
  GET  /health
  POST /demo/bootstrap
  GET  /workspaces/<workspace_id>
"""

print("LOADING SERVER FILE:", __file__)

from __future__ import annotations

import asyncio

from flask import Flask, jsonify

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

app = Flask(__name__)

runtime = Runtime()


class DemoBuilder(SynapseAgent):
    profile = AgentProfile(
        name="Builder",
        role="Product Engineer",
        department="Engineering",
        strengths=["implementation"],
        permissions={Permission.EXECUTE_TASKS, Permission.MANAGE_ARTIFACTS},
        description="Creates implementation artifacts for demo tasks.",
    )

    async def execute_task(self, task, snapshot) -> TaskExecutionResult:
        return TaskExecutionResult(
            status=TaskStatus.IN_REVIEW,
            summary=f"Executed task '{task.title}' in workspace '{snapshot.workspace.name}'.",
            artifact_drafts=[
                {
                    "name": f"{task.title} Output",
                    "artifact_type": ArtifactType.DOCUMENT,
                    "content": f"Workspace task result for: {task.description}",
                }
            ],
        )


runtime.add_agent(DemoBuilder())


def run(coro):
    return asyncio.run(coro)


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Synapse Workspace Server",
        "endpoints": ["/health", "/demo/bootstrap", "/workspaces/<workspace_id>", "/research (legacy if present)"]
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/demo/bootstrap", methods=["POST"])
def bootstrap_demo():
    async def _bootstrap() -> dict:
        await runtime.run()
        organization = await runtime.create_organization(
            "Demo Company",
            context=CompanyContext(
                mission="Run structured organizational workflows with AI workers.",
                goals=["Create visible AI work execution", "Keep state persistent"],
                writing_style="Direct",
                communication_style="Operational",
            ),
        )
        department = await runtime.add_department(
            organization.organization_id,
            "Engineering",
            "Builds workflow capabilities.",
        )
        role = await runtime.add_role(
            organization.organization_id,
            "Product Engineer",
            department_id=department.department_id,
            permissions={Permission.EXECUTE_TASKS, Permission.MANAGE_ARTIFACTS},
        )
        worker = await runtime.register_worker(
            organization.organization_id,
            "Builder",
            role.role_id,
            model="gemma3:1b",
        )
        workspace = await runtime.create_workspace(
            organization.organization_id,
            "Demo Workspace",
            "Seed workspace for Synapse's organizational model.",
        )
        project = await runtime.create_project(
            workspace.workspace_id,
            "Operating System Demo",
            owner_id=worker.agent_id,
        )
        task = await runtime.create_task(
            workspace.workspace_id,
            "Establish execution trace",
            description="Generate the first structured output for the workspace.",
            task_type=TaskType.IMPLEMENTATION,
            priority=Priority.HIGH,
            project_id=project.project_id,
            assigned_role_id=role.role_id,
        )
        await runtime.assign_task(workspace.workspace_id, task.task_id, worker.agent_id, "system")
        await runtime.execute_task(workspace.workspace_id, task.task_id)
        snapshot = await runtime.memory.export_workspace(workspace.workspace_id)
        return snapshot

    return jsonify(run(_bootstrap()))


@app.route("/workspaces/<workspace_id>")
def get_workspace(workspace_id: str):
    return jsonify(run(runtime.memory.export_workspace(workspace_id)))


if __name__ == "__main__":
    print("Synapse workspace server running on http://localhost:5001")
    print(app.url_map)
    app.run(host="0.0.0.0", port=5001)
