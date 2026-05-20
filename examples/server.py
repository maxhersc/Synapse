"""
Minimal Flask server exposing the workspace-oriented Synapse demo.

Endpoints:
  GET  /
  GET  /health
  GET  /demo/bootstrap
  POST /demo/bootstrap
  GET  /workspaces/<workspace_id>
  POST /workspaces/<workspace_id>/tasks
  POST /workspaces/<workspace_id>/tasks/<task_id>/assign
  POST /workspaces/<workspace_id>/tasks/<task_id>/execute
  POST /workspaces/<workspace_id>/tasks/<task_id>/reviews
  POST /workspaces/<workspace_id>/reviews/<review_id>
  POST /workspaces/<workspace_id>/decisions
"""

from __future__ import annotations

import asyncio
import os

from flask import Flask, jsonify, request

from synapse import (
    AgentProfile,
    ArtifactType,
    CompanyContext,
    DecisionState,
    Permission,
    Priority,
    ReviewStatus,
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


class DemoReviewer(SynapseAgent):
    profile = AgentProfile(
        name="Reviewer",
        role="Designated Reviewer",
        department="Operations",
        strengths=["review", "governance"],
        permissions={Permission.REVIEW_WORK, Permission.APPROVE_WORK, Permission.VIEW_WORKSPACE},
        description="Reviews artifacts and approves or requests changes.",
    )


class DemoOperator(SynapseAgent):
    profile = AgentProfile(
        name="Operator",
        role="Workflow Operator",
        department="Operations",
        strengths=["coordination", "workflow"],
        permissions={Permission.MANAGE_TASKS, Permission.ASSIGN_TASKS, Permission.RECORD_DECISIONS},
        description="Coordinates assignments and records operational decisions.",
    )


runtime.add_agent(DemoBuilder())
runtime.add_agent(DemoReviewer())
runtime.add_agent(DemoOperator())


def run(coro):
    return asyncio.run(coro)


def _snapshot_or_404(workspace_id: str):
    snapshot = run(runtime.memory.export_workspace(workspace_id))
    if not snapshot:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


async def _seed_demo_workspace() -> dict:
    await runtime.memory.clear()
    await runtime.run()

    organization = await runtime.create_organization(
        "Demo Company",
        context=CompanyContext(
            mission="Run structured organizational workflows with AI workers.",
            goals=[
                "Create visible AI work execution",
                "Keep state persistent",
                "Make approvals and decisions inspectable",
            ],
            writing_style="Direct",
            communication_style="Operational",
            product_standards=["Explicit ownership", "Traceable execution"],
            design_standards=["Readable state", "Low-friction review"],
            domain_knowledge=["Workflow systems", "Developer operations"],
            brand_voice="Clear and accountable",
        ),
        decision_rules=["High-impact changes require explicit review", "Decisions should cite affected tasks"],
        communication_standards=["Prefer state changes over chat", "Log operational decisions"],
        brand_identity={"voice": "Operational", "posture": "Calm control room"},
    )
    engineering = await runtime.add_department(
        organization.organization_id,
        "Engineering",
        "Builds workflow capabilities.",
    )
    operations = await runtime.add_department(
        organization.organization_id,
        "Operations",
        "Oversees review and workflow control.",
    )

    builder_role = await runtime.add_role(
        organization.organization_id,
        "Product Engineer",
        department_id=engineering.department_id,
        description="Owns implementation and artifact delivery.",
        permissions={Permission.EXECUTE_TASKS, Permission.MANAGE_ARTIFACTS, Permission.VIEW_WORKSPACE},
    )
    reviewer_role = await runtime.add_role(
        organization.organization_id,
        "Reviewer",
        department_id=operations.department_id,
        description="Reviews output and marks approval state.",
        permissions={Permission.REVIEW_WORK, Permission.APPROVE_WORK, Permission.VIEW_WORKSPACE},
    )
    operator_role = await runtime.add_role(
        organization.organization_id,
        "Operator",
        department_id=operations.department_id,
        description="Coordinates tasks and records decisions.",
        permissions={Permission.MANAGE_TASKS, Permission.ASSIGN_TASKS, Permission.RECORD_DECISIONS},
    )

    builder = await runtime.register_worker(
        organization.organization_id,
        "Builder",
        builder_role.role_id,
        model="gemma3:1b",
        skills=["implementation", "delivery", "artifact drafting"],
        responsibilities=["Execute scoped tasks", "Produce shared artifacts"],
    )
    reviewer = await runtime.register_worker(
        organization.organization_id,
        "Reviewer",
        reviewer_role.role_id,
        model="gemma3:1b",
        skills=["review", "approval"],
        responsibilities=["Review artifacts", "Approve or request changes"],
    )
    operator = await runtime.register_worker(
        organization.organization_id,
        "Operator",
        operator_role.role_id,
        model="gemma3:1b",
        skills=["coordination", "issue routing"],
        responsibilities=["Assign work", "Record decisions", "Track execution state"],
    )
    human = await runtime.register_human(
        "Max",
        "Workspace Owner",
        permissions={
            Permission.MANAGE_ORGANIZATION,
            Permission.MANAGE_TASKS,
            Permission.ASSIGN_TASKS,
            Permission.APPROVE_WORK,
            Permission.RECORD_DECISIONS,
        },
    )

    workspace = await runtime.create_workspace(
        organization.organization_id,
        "Demo Workspace",
        "Seed workspace for Synapse's organizational operating system.",
        metadata={"owner_human_id": human.participant_id},
    )
    project = await runtime.create_project(
        workspace.workspace_id,
        "Operating System Demo",
        description="Demonstrates task execution, review, and decision capture.",
        owner_id=operator.agent_id,
    )
    task = await runtime.create_task(
        workspace.workspace_id,
        "Establish execution trace",
        description="Generate the first structured output for the workspace.",
        task_type=TaskType.IMPLEMENTATION,
        priority=Priority.HIGH,
        project_id=project.project_id,
        assigned_role_id=builder_role.role_id,
    )
    await runtime.assign_task(workspace.workspace_id, task.task_id, builder.agent_id, operator.agent_id)
    await runtime.execute_task(workspace.workspace_id, task.task_id)
    artifact_id = runtime.memory.tasks[task.task_id].artifact_ids[0]
    await runtime.create_review(
        workspace.workspace_id,
        task.task_id,
        reviewer.agent_id,
        artifact_id=artifact_id,
        summary="Initial implementation ready for reviewer inspection.",
    )
    await runtime.record_decision(
        workspace.workspace_id,
        "Adopt workspace-state demo flow",
        summary="The demo environment will emphasize task execution, reviews, and decisions instead of chat.",
        made_by=operator.agent_id,
        state=DecisionState.ACCEPTED,
        related_task_ids=[task.task_id],
        related_artifact_ids=[artifact_id],
        rationale=["Aligns the product surface with the operating-system direction."],
    )

    return await runtime.memory.export_workspace(workspace.workspace_id)


@app.route("/")
def index():
    index_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "index.html")
    with open(index_path, "r", encoding="utf-8") as handle:
        return handle.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/demo/bootstrap", methods=["GET", "POST"])
def bootstrap_demo():
    return jsonify(run(_seed_demo_workspace()))


@app.route("/workspaces/<workspace_id>")
def get_workspace(workspace_id: str):
    return _snapshot_or_404(workspace_id)


@app.route("/workspaces/<workspace_id>/tasks", methods=["POST"])
def create_task_route(workspace_id: str):
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    async def _create():
        workspace = runtime.memory.workspaces.get(workspace_id)
        if workspace is None:
            return None
        organization = runtime.memory.organizations[workspace.organization_id]
        assigned_role_id = payload.get("assigned_role_id")
        if not assigned_role_id and organization.roles:
            assigned_role_id = next(iter(organization.roles.keys()))
        project_id = payload.get("project_id") or (workspace.project_ids[0] if workspace.project_ids else None)
        task_type = TaskType(payload.get("task_type", TaskType.GENERAL.value))
        priority = Priority(payload.get("priority", Priority.NORMAL.value))
        await runtime.create_task(
            workspace_id,
            title,
            description=str(payload.get("description", "")),
            task_type=task_type,
            priority=priority,
            project_id=project_id,
            assigned_role_id=assigned_role_id,
            created_by=str(payload.get("created_by", "system")),
        )
        return await runtime.memory.export_workspace(workspace_id)

    snapshot = run(_create())
    if snapshot is None:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/tasks/<task_id>/assign", methods=["POST"])
def assign_task_route(workspace_id: str, task_id: str):
    payload = request.get_json(silent=True) or {}
    agent_id = str(payload.get("agent_id", "")).strip()
    if not agent_id:
        return jsonify({"error": "agent_id is required"}), 400

    async def _assign():
        if workspace_id not in runtime.memory.workspaces or task_id not in runtime.memory.tasks or agent_id not in runtime.memory.agents:
            return None
        await runtime.assign_task(workspace_id, task_id, agent_id, str(payload.get("actor_id", "system")))
        return await runtime.memory.export_workspace(workspace_id)

    snapshot = run(_assign())
    if snapshot is None:
        return jsonify({"error": "workspace, task, or agent not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/tasks/<task_id>/execute", methods=["POST"])
def execute_task_route(workspace_id: str, task_id: str):
    async def _execute():
        if workspace_id not in runtime.memory.workspaces or task_id not in runtime.memory.tasks:
            return None
        await runtime.execute_task(workspace_id, task_id)
        return await runtime.memory.export_workspace(workspace_id)

    try:
        snapshot = run(_execute())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if snapshot is None:
        return jsonify({"error": "workspace or task not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/tasks/<task_id>/reviews", methods=["POST"])
def create_review_route(workspace_id: str, task_id: str):
    payload = request.get_json(silent=True) or {}
    reviewer_id = str(payload.get("reviewer_id", "")).strip()
    if not reviewer_id:
        return jsonify({"error": "reviewer_id is required"}), 400

    async def _create_review():
        if workspace_id not in runtime.memory.workspaces or task_id not in runtime.memory.tasks or reviewer_id not in runtime.memory.agents:
            return None
        artifact_id = payload.get("artifact_id")
        if not artifact_id:
            task = runtime.memory.tasks[task_id]
            artifact_id = task.artifact_ids[-1] if task.artifact_ids else None
        await runtime.create_review(
            workspace_id,
            task_id,
            reviewer_id,
            artifact_id=artifact_id,
            summary=str(payload.get("summary", "Review requested from workspace UI.")),
        )
        return await runtime.memory.export_workspace(workspace_id)

    snapshot = run(_create_review())
    if snapshot is None:
        return jsonify({"error": "workspace, task, or reviewer not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/reviews/<review_id>", methods=["POST"])
def update_review_route(workspace_id: str, review_id: str):
    payload = request.get_json(silent=True) or {}
    status_raw = str(payload.get("status", "")).strip()
    if not status_raw:
        return jsonify({"error": "status is required"}), 400

    async def _update_review():
        if workspace_id not in runtime.memory.workspaces or review_id not in runtime.memory.reviews:
            return None
        status = ReviewStatus(status_raw)
        review = await runtime.update_review(
            workspace_id,
            review_id,
            status,
            str(payload.get("actor_id", "system")),
            summary=str(payload.get("summary", "")),
            requested_changes=payload.get("requested_changes"),
        )
        task = runtime.memory.tasks[review.task_id]
        if status == ReviewStatus.APPROVED:
            await runtime.update_task_status(workspace_id, task.task_id, TaskStatus.APPROVED, str(payload.get("actor_id", "system")), "Task approved in review.")
        elif status == ReviewStatus.CHANGES_REQUESTED:
            await runtime.update_task_status(workspace_id, task.task_id, TaskStatus.BLOCKED, str(payload.get("actor_id", "system")), "Changes requested in review.")
        return await runtime.memory.export_workspace(workspace_id)

    try:
        snapshot = run(_update_review())
    except ValueError:
        return jsonify({"error": "invalid review status"}), 400
    if snapshot is None:
        return jsonify({"error": "workspace or review not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/decisions", methods=["POST"])
def record_decision_route(workspace_id: str):
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    if not title or not summary:
        return jsonify({"error": "title and summary are required"}), 400

    async def _record():
        if workspace_id not in runtime.memory.workspaces:
            return None
        state = DecisionState(str(payload.get("state", DecisionState.PROPOSED.value)))
        await runtime.record_decision(
            workspace_id,
            title,
            summary=summary,
            made_by=str(payload.get("made_by", "system")),
            state=state,
            related_task_ids=payload.get("related_task_ids") or [],
            related_artifact_ids=payload.get("related_artifact_ids") or [],
            rationale=payload.get("rationale") or [],
        )
        return await runtime.memory.export_workspace(workspace_id)

    try:
        snapshot = run(_record())
    except ValueError:
        return jsonify({"error": "invalid decision state"}), 400
    if snapshot is None:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


if __name__ == "__main__":
    print("Synapse workspace server running on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001)
