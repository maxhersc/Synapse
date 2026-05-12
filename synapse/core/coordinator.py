from __future__ import annotations

from typing import Any

from synapse.protocols.message import Goal, HelpRequest, Message, Task, TaskStatus


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
        """Store a goal, decompose it into tasks, assign them, and return the created tasks."""

        self._goals[goal.id] = goal
        await self._memory.set(f"goal:{goal.id}", goal)
        await self._memory.set(f"goal:{goal.id}:status", "submitted")

        tasks = await self._decompose_goal(goal)
        for task in tasks:
            self._tasks[task.id] = task
            await self.assign_task(task)

        if not tasks:
            await self._memory.set(f"goal:{goal.id}:status", "complete")
        else:
            await self._memory.set(f"goal:{goal.id}:status", "in_progress")

        return tasks

    async def assign_task(self, task: Task) -> None:
        """Assign a task to the best-fit agent, deliver it, and persist its state."""

        agent_id = await self._best_agent_for(task)
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
            target_agent_id = await self._best_agent_for(task)

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

        tasks: list[Task] = []
        for agent in self._agents.values():
            strength = agent.profile.strengths[0] if agent.profile.strengths else agent.profile.name
            task = Task(
                description=f"{strength}: {goal.description}",
                goal_id=goal.id,
            )
            tasks.append(task)
        return tasks

    async def _best_agent_for(self, task: Task) -> str | None:
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
