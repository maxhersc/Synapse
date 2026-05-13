from __future__ import annotations

import json
from typing import Any

from synapse.protocols.message import Goal, HelpRequest, Message, Task, TaskStatus


async def _call_ollama(prompt: str, model: str = "gemma3:1b") -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )
        return response.json()["response"]


class Coordinator:
    """Orchestration brain that decomposes goals, assigns tasks, and tracks execution state."""

    def __init__(self, bus: Any, memory: Any) -> None:
        self._bus: Any = bus
        self._memory: Any = memory
        self._agents: dict[str, Any] = {}
        self._tasks: dict[str, Task] = {}
        self._goals: dict[str, Goal] = {}

    def register_agent(self, agent: Any) -> None:
        """Register an agent so it can receive task assignments."""

        self._agents[agent.id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the coordinator registry."""

        self._agents.pop(agent_id, None)

    async def submit_goal(self, goal: Goal) -> list[Task]:
        """Store a goal, decompose it into tasks, and return the created tasks."""

        self._goals[goal.id] = goal
        await self._memory.set(f"goal:{goal.id}", goal)
        await self._memory.set(f"goal:{goal.id}:status", "submitted")

        tasks = await self._decompose_goal(goal)
        for task in tasks:
            self._tasks[task.id] = task
            await self._memory.set(f"task:{task.id}", task)
            await self._memory.set(f"task:{task.id}:status", task.status.value)

        if not tasks:
            await self._memory.set(f"goal:{goal.id}:status", "complete")
        else:
            await self._memory.set(f"goal:{goal.id}:status", "pending")

        return tasks

    async def assign_task(self, task: Task) -> None:
        """Assign a task to the best-fit agent, deliver it, and persist its state."""

        agent_id = task.assigned_to
        if agent_id is None:
            agent_id = self._keyword_match_agent(task)
            task.assigned_to = agent_id
        if agent_id is None:
            await self._memory.set(f"task:{task.id}:status", task.status.value)
            return

        task.status = TaskStatus.ASSIGNED
        await self._memory.set(f"task:{task.id}", task)
        await self._memory.set(f"task:{task.id}:status", task.status.value)
        await self._memory.set(f"goal:{task.goal_id}:status", "in_progress")

        await self._bus.send(
            Message(
                sender_id="coordinator",
                recipient_id=agent_id,
                content=task.description,
                metadata={"task": task},
            )
        )

    async def handle_help_request(self, request: HelpRequest) -> None:
        """Handle a help request by reassigning the task to an agent with the needed capability."""

        task = self._tasks.get(request.task_id)
        if task is None:
            return

        target_agent_id: str | None = None
        if request.capability_needed is not None:
            for agent_id, agent in self._agents.items():
                if agent_id == request.from_agent_id:
                    continue
                if request.capability_needed in agent.profile.capabilities:
                    target_agent_id = agent_id
                    break

        if target_agent_id is None:
            target_agent_id = self._keyword_match_agent(task)

        if target_agent_id is None:
            return

        task.assigned_to = target_agent_id
        task.status = TaskStatus.ASSIGNED
        await self._memory.set(f"task:{task.id}", task)
        await self._memory.set(f"task:{task.id}:status", task.status.value)
        await self._memory.set(f"goal:{task.goal_id}:status", "in_progress")

        await self._bus.send(
            Message(
                sender_id="coordinator",
                recipient_id=target_agent_id,
                content=task.description,
                metadata={"task": task},
            )
        )

    async def complete_task(self, task_id: str, result: Any) -> None:
        """Mark a task complete, persist its result, and update goal status if all tasks are done."""

        task = self._tasks.get(task_id)
        if task is None:
            return

        task.complete(result)
        await self._memory.set(f"task:{task.id}", task)
        await self._memory.set(f"task:{task.id}:status", task.status.value)
        await self._memory.set(f"task:{task.id}:result", result)

        goal_tasks = [item for item in self._tasks.values() if item.goal_id == task.goal_id]
        if goal_tasks and all(item.status == TaskStatus.COMPLETE for item in goal_tasks):
            await self._memory.set(f"goal:{task.goal_id}:status", "complete")
        else:
            await self._memory.set(f"goal:{task.goal_id}:status", "in_progress")

    async def fail_task(self, task_id: str, reason: str) -> None:
        """Mark a task failed and persist the failure reason."""

        task = self._tasks.get(task_id)
        if task is None:
            return

        task.fail(reason)
        await self._memory.set(f"task:{task.id}", task)
        await self._memory.set(f"task:{task.id}:status", task.status.value)
        await self._memory.set(f"task:{task.id}:result", reason)
        await self._memory.set(f"goal:{task.goal_id}:status", "failed")

    def progress(self, goal_id: str) -> dict[str, int]:
        """Return task counts by status for the given goal."""

        counts: dict[str, int] = {
            "pending": 0,
            "assigned": 0,
            "in_progress": 0,
            "complete": 0,
            "failed": 0,
        }
        for task in self._tasks.values():
            if task.goal_id != goal_id:
                continue
            if task.status == TaskStatus.PENDING:
                counts["pending"] += 1
            elif task.status == TaskStatus.ASSIGNED:
                counts["assigned"] += 1
            elif task.status == TaskStatus.IN_PROGRESS:
                counts["in_progress"] += 1
            elif task.status == TaskStatus.COMPLETE:
                counts["complete"] += 1
            elif task.status == TaskStatus.FAILED:
                counts["failed"] += 1
        return counts

    async def _decompose_goal(self, goal: Goal) -> list[Task]:
        """Create one simple task per registered agent based on the goal and agent strengths."""

        if not self._agents:
            return []

        prompt = "\n".join(
            [
                "You are a task coordinator managing a team of AI agents.",
                "",
                "Your job is to break the following goal into distinct subtasks — one per agent — where each subtask is a genuinely different piece of work. The output of each agent should feed into the next.",
                "",
                f"Goal: {goal.description}",
                "",
                "Available agents:",
                *[
                    (
                        f"- {agent.profile.name} (id: {agent.id}): "
                        f"{agent.profile.description}. Strengths: {agent.profile.strengths}"
                    )
                    for agent in self._agents.values()
                ],
                "",
                "Rules:",
                "- Each agent gets a DIFFERENT piece of the work, not a rephrasing of the same task",
                "- Tasks should be sequential — later agents build on earlier agents' results",
                "- Each task description should be specific and actionable, not generic",
                "- Assign tasks that match each agent's strengths and description",
                "",
                "Return ONLY a valid JSON array, no explanation, no markdown:",
                "[",
                '  {"agent_id": "researcher", "task": "specific actionable task for this agent"},',
                '  {"agent_id": "writer", "task": "specific actionable task that builds on researcher output"},',
                '  {"agent_id": "reviewer", "task": "specific actionable task that reviews writer output"}',
                "]",
            ]
        )

        try:
            response = await _call_ollama(prompt)
            assignments = json.loads(response)
            if not isinstance(assignments, list):
                raise ValueError("Expected a JSON list of assignments.")

            tasks: list[Task] = []
            seen_agents: set[str] = set()
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    raise ValueError("Each assignment must be an object.")
                agent_id = assignment.get("agent_id")
                description = assignment.get("task")
                if not isinstance(agent_id, str) or not isinstance(description, str):
                    raise ValueError("Each assignment must include string agent_id and task fields.")
                if agent_id not in self._agents:
                    raise ValueError(f"Unknown agent_id '{agent_id}' returned by coordinator.")
                seen_agents.add(agent_id)
                tasks.append(
                    Task(
                        description=description,
                        goal_id=goal.id,
                        assigned_to=agent_id,
                    )
                )

            if seen_agents != set(self._agents):
                raise ValueError("Coordinator did not assign exactly one task per registered agent.")

            return tasks
        except (json.JSONDecodeError, ValueError):
            print("[coordinator] LLM response parsing failed, falling back to keyword matching.")
            return self._fallback_tasks(goal)
        except Exception:
            return self._fallback_tasks(goal)

    def _fallback_tasks(self, goal: Goal) -> list[Task]:
        """Create one simple task per registered agent using local keyword matching."""

        tasks: list[Task] = []
        for agent in self._agents.values():
            strength = agent.profile.strengths[0] if agent.profile.strengths else agent.profile.name
            task = Task(
                description=f"{strength}: {goal.description}",
                goal_id=goal.id,
                assigned_to=agent.id,
            )
            tasks.append(task)
        return tasks

    def _keyword_match_agent(self, task: Task) -> str | None:
        """Return the registered agent whose strengths best match the task description."""

        if not self._agents:
            return None

        task_description = task.description.lower()
        best_agent_id: str | None = None
        best_score = -1

        for agent_id, agent in self._agents.items():
            score = 0
            for strength in agent.profile.strengths:
                if strength.lower() in task_description:
                    score += 1
            if score > best_score:
                best_score = score
                best_agent_id = agent_id

        return best_agent_id
