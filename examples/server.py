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
import time

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


def _conversation_store_key(workspace_id: str) -> str:
    return f"workspace:{workspace_id}:conversations"


def _message_groups(snapshot: dict) -> list[dict[str, str]]:
    agents = snapshot.get("agents", [])
    groups = [{"id": "all", "label": "All Agents"}]
    departments = snapshot.get("organization", {}).get("departments", {})
    for department_id, department in departments.items():
        if any(agent.get("department_id") == department_id for agent in agents):
            groups.append({"id": f"department:{department_id}", "label": department.get("name", "Department")})
    for agent in agents:
        if "review" in agent.get("skills", []):
            groups.append({"id": "skill:review", "label": "Reviewers"})
            break
    return groups


def _recipient_agent_ids(snapshot: dict, target: str) -> list[str]:
    agents = snapshot.get("agents", [])
    if target == "all":
        return [agent["agent_id"] for agent in agents]
    if target.startswith("department:"):
        department_id = target.split(":", 1)[1]
        return [agent["agent_id"] for agent in agents if agent.get("department_id") == department_id]
    if target == "skill:review":
        return [agent["agent_id"] for agent in agents if "review" in agent.get("skills", [])]
    return [target]


def _conversation_title(snapshot: dict, target: str, participant_agent_ids: list[str]) -> str:
    departments = snapshot.get("organization", {}).get("departments", {})
    agents = {agent["agent_id"]: agent for agent in snapshot.get("agents", [])}
    if target == "all":
        return "All Agents"
    if target.startswith("department:"):
        department_id = target.split(":", 1)[1]
        return departments.get(department_id, {}).get("name", "Department Chat")
    if target == "skill:review":
        return "Reviewers"
    if len(participant_agent_ids) == 1 and participant_agent_ids[0] in agents:
        return agents[participant_agent_ids[0]]["name"]
    return "Group Chat"


def _new_conversation(snapshot: dict, conversation_id: str, target: str, participant_agent_ids: list[str]) -> dict:
    agents = {agent["agent_id"]: agent for agent in snapshot.get("agents", [])}
    participant_names = [agents[agent_id]["name"] for agent_id in participant_agent_ids if agent_id in agents]
    return {
        "id": conversation_id,
        "target": target,
        "title": _conversation_title(snapshot, target, participant_agent_ids),
        "type": "direct" if len(participant_agent_ids) == 1 else "group",
        "participant_agent_ids": participant_agent_ids,
        "participant_names": participant_names,
        "messages": [],
        "updated_at": time.time(),
    }


def _append_message_entry(conversation: dict, *, sender: str, recipients: list[str], body: str, kind: str = "message", audience: str = "direct") -> None:
    conversation["messages"].append(
        {
            "id": f"msg_{len(conversation['messages']) + 1}_{int(time.time() * 1000)}",
            "sender": sender,
            "recipients": recipients,
            "body": body,
            "kind": kind,
            "audience": audience,
            "timestamp": time.time(),
        }
    )
    conversation["updated_at"] = time.time()


def _agent_reply(agent: dict, organization_context: dict, user_message: str) -> str:
    mission = organization_context.get("mission") or "the company mission"
    goals = organization_context.get("goals") or []
    goal_fragment = goals[0] if goals else "the current workspace goals"
    role_note = ", ".join(agent.get("responsibilities", [])[:2]) or "my assigned responsibilities"
    return (
        f"I've received that and I will handle it as {agent['name']}. "
        f"I'll keep the response aligned with {mission.lower()} and focus on {goal_fragment.lower()}. "
        f"My next step is to work through {role_note.lower()}."
    )


async def _read_conversations(workspace_id: str) -> list[dict]:
    return await runtime.memory.get(_conversation_store_key(workspace_id), [])


async def _write_conversations(workspace_id: str, conversations: list[dict]) -> None:
    await runtime.memory.set(_conversation_store_key(workspace_id), conversations)


def _sorted_conversations(conversations: list[dict]) -> list[dict]:
    return sorted(conversations, key=lambda item: item.get("updated_at", 0), reverse=True)


def _snapshot_or_404(workspace_id: str):
    snapshot = run(_enrich_snapshot(workspace_id))
    if not snapshot:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


async def _enrich_snapshot(workspace_id: str) -> dict:
    snapshot = await runtime.memory.export_workspace(workspace_id)
    if not snapshot:
        return {}
    conversations = _sorted_conversations(await _read_conversations(workspace_id))
    snapshot["message_groups"] = _message_groups(snapshot)
    snapshot["conversations"] = conversations
    snapshot["messages"] = [message for conversation in conversations for message in conversation.get("messages", [])]
    return snapshot


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

    conversation_seed_snapshot = {
        "agents": [
            {"agent_id": builder.agent_id, "name": builder.name, "department_id": builder.department_id},
            {"agent_id": reviewer.agent_id, "name": reviewer.name, "department_id": reviewer.department_id},
            {"agent_id": operator.agent_id, "name": operator.name, "department_id": operator.department_id},
        ],
        "organization": {"departments": {}},
    }

    conversations = [
        _new_conversation(
            conversation_seed_snapshot,
            "conv_all_agents",
            "all",
            [builder.agent_id, reviewer.agent_id, operator.agent_id],
        ),
        _new_conversation(
            conversation_seed_snapshot,
            "conv_builder_direct",
            builder.agent_id,
            [builder.agent_id],
        ),
        _new_conversation(
            conversation_seed_snapshot,
            "conv_reviewer_direct",
            reviewer.agent_id,
            [reviewer.agent_id],
        ),
        _new_conversation(
            conversation_seed_snapshot,
            "conv_operator_direct",
            operator.agent_id,
            [operator.agent_id],
        ),
    ]

    _append_message_entry(
        conversations[0],
        sender="Operator",
        recipients=["Builder", "Reviewer"],
        body="We are bootstrapping the company workspace. Let's align on how we surface execution, reviews, and decisions clearly for the user.",
        kind="planning",
        audience="group",
    )
    _append_message_entry(
        conversations[0],
        sender="Builder",
        recipients=["Operator", "Reviewer"],
        body="I'll take the first pass on the implementation surface and keep the task and artifact flow simple enough to scan quickly.",
        kind="planning",
        audience="group",
    )
    _append_message_entry(
        conversations[0],
        sender="Reviewer",
        recipients=["Operator", "Builder"],
        body="I'll focus on whether the workflow is understandable and whether approvals and changes requested are obvious at a glance.",
        kind="planning",
        audience="group",
    )
    _append_message_entry(
        conversations[1],
        sender="Builder",
        recipients=["You"],
        body="My default focus is implementation and artifact delivery. If you need a concrete build pass, message me directly here.",
        kind="intro",
        audience="direct",
    )
    _append_message_entry(
        conversations[2],
        sender="Reviewer",
        recipients=["You"],
        body="I handle review, approvals, and clarity checks. Use this chat if you want feedback on the team's output.",
        kind="intro",
        audience="direct",
    )
    _append_message_entry(
        conversations[3],
        sender="Operator",
        recipients=["You"],
        body="I coordinate the team and keep the workspace organized. Ask me to route work or align the agents.",
        kind="intro",
        audience="direct",
    )
    await _write_conversations(workspace.workspace_id, conversations)

    return await _enrich_snapshot(workspace.workspace_id)


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


@app.route("/workspaces/<workspace_id>/context", methods=["POST"])
def update_context_route(workspace_id: str):
    payload = request.get_json(silent=True) or {}

    async def _update():
        workspace = runtime.memory.workspaces.get(workspace_id)
        if workspace is None:
            return None
        organization = runtime.memory.organizations[workspace.organization_id]
        context = organization.context
        if "mission" in payload:
            context.mission = str(payload.get("mission", "")).strip()
        if "goals" in payload:
            raw_goals = payload.get("goals", [])
            if isinstance(raw_goals, str):
                context.goals = [goal.strip() for goal in raw_goals.splitlines() if goal.strip()]
            else:
                context.goals = [str(goal).strip() for goal in raw_goals if str(goal).strip()]
        if "writing_style" in payload:
            context.writing_style = str(payload.get("writing_style", "")).strip()
        if "communication_style" in payload:
            context.communication_style = str(payload.get("communication_style", "")).strip()
        if "brand_voice" in payload:
            context.brand_voice = str(payload.get("brand_voice", "")).strip()
        if "domain_knowledge" in payload:
            raw_knowledge = payload.get("domain_knowledge", [])
            if isinstance(raw_knowledge, str):
                context.domain_knowledge = [item.strip() for item in raw_knowledge.splitlines() if item.strip()]
            else:
                context.domain_knowledge = [str(item).strip() for item in raw_knowledge if str(item).strip()]
        await runtime.memory.save_organization(organization)
        conversations = await _read_conversations(workspace_id)
        if conversations:
            _append_message_entry(
                conversations[0],
                sender="System",
                recipients=["All Agents"],
                body="Company context was updated. All agents should use the new mission, goals, and communication guidance immediately.",
                kind="context_update",
                audience="group",
            )
        await _write_conversations(workspace_id, conversations)
        return await _enrich_snapshot(workspace_id)

    snapshot = run(_update())
    if snapshot is None:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


@app.route("/workspaces/<workspace_id>/messages", methods=["GET"])
def get_messages_route(workspace_id: str):
    if workspace_id not in runtime.memory.workspaces:
        return jsonify({"error": "workspace not found"}), 404
    snapshot = run(_enrich_snapshot(workspace_id))
    return jsonify(
        {
            "conversations": snapshot.get("conversations", []),
            "agents": snapshot.get("agents", []),
            "groups": _message_groups(snapshot),
        }
    )


@app.route("/workspaces/<workspace_id>/messages", methods=["POST"])
def send_message_route(workspace_id: str):
    payload = request.get_json(silent=True) or {}
    body = str(payload.get("body", "")).strip()
    target = str(payload.get("target", "")).strip()
    conversation_id = str(payload.get("conversation_id", "")).strip()
    sender = str(payload.get("sender", "You")).strip() or "You"
    if not body or (not target and not conversation_id):
        return jsonify({"error": "body and target or conversation_id are required"}), 400

    async def _send():
        snapshot = await _enrich_snapshot(workspace_id)
        if not snapshot:
            return None
        conversations = await _read_conversations(workspace_id)
        agents = {agent["agent_id"]: agent for agent in snapshot.get("agents", [])}

        conversation = next((item for item in conversations if item["id"] == conversation_id), None) if conversation_id else None
        if conversation is None:
            recipient_agent_ids = _recipient_agent_ids(snapshot, target)
            if not recipient_agent_ids:
                return None
            resolved_target = target or (recipient_agent_ids[0] if len(recipient_agent_ids) == 1 else "all")
            conversation = _new_conversation(
                snapshot,
                f"conv_{len(conversations) + 1}_{int(time.time() * 1000)}",
                resolved_target,
                recipient_agent_ids,
            )
            conversations.append(conversation)
        else:
            recipient_agent_ids = conversation.get("participant_agent_ids", [])
            target = conversation.get("target", target)

        recipient_names = [agents[agent_id]["name"] for agent_id in recipient_agent_ids if agent_id in agents]
        audience = "group" if conversation.get("type") == "group" else "direct"
        _append_message_entry(
            conversation,
            sender=sender,
            recipients=recipient_names or [target],
            body=body,
            kind="message",
            audience=audience,
        )

        organization_context = snapshot.get("organization", {}).get("context", {})
        for agent_id in recipient_agent_ids:
            agent = agents.get(agent_id)
            if agent is None:
                continue
            _append_message_entry(
                conversation,
                sender=agent["name"],
                recipients=[sender],
                body=_agent_reply(agent, organization_context, body),
                kind="reply",
                audience="direct",
            )

        if audience == "group" and len(recipient_names) >= 2:
            _append_message_entry(
                conversation,
                sender=recipient_names[0],
                recipients=recipient_names[1:],
                body="Let's split this cleanly: I will take first ownership and keep the others informed with explicit updates.",
                kind="planning",
                audience="group",
            )

        await _write_conversations(workspace_id, conversations)
        return await _enrich_snapshot(workspace_id)

    snapshot = run(_send())
    if snapshot is None:
        return jsonify({"error": "workspace not found"}), 404
    return jsonify(snapshot)


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
        return await _enrich_snapshot(workspace_id)

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
        return await _enrich_snapshot(workspace_id)

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
        return await _enrich_snapshot(workspace_id)

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
        return await _enrich_snapshot(workspace_id)

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
        return await _enrich_snapshot(workspace_id)

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
