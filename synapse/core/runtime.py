from __future__ import annotations

import asyncio
from typing import Any

from synapse.core.bus import Bus
from synapse.core.coordinator import Coordinator
from synapse.core.memory import SharedMemory
from synapse.protocols.message import Goal, Task
from synapse.result import AgentResult


class Runtime:
    """Entry point that wires together Synapse core components and agent lifecycle management."""

    def __init__(self) -> None:
        self.memory = SharedMemory()
        self.bus = Bus()
        self.coordinator = Coordinator(self.bus, self.memory)
        self.results: dict[str, AgentResult] = {}
        self._agents: list[Any] = []
        self._pending_tasks: dict[str, Task] = {}

    def add(self, agent: Any) -> Runtime:
        """Register an agent with the runtime, bus, and coordinator."""

        agent._inject(self.bus, self.memory, self.coordinator, self)
        self._agents.append(agent)
        self.bus.attach(agent)
        self.coordinator.register_agent(agent)
        return self

    async def start(self) -> None:
        """Start all registered agents concurrently."""

        await asyncio.gather(*(agent.start() for agent in self._agents))

    async def stop(self) -> None:
        """Stop all registered agents concurrently."""

        await asyncio.gather(*(agent.stop() for agent in self._agents))

    async def run(self, until: asyncio.Future[Any] | None = None) -> None:
        """Start agents, wait for completion or indefinitely, and always stop cleanly."""

        await self.start()
        try:
            if until is None:
                await asyncio.Future()
            else:
                await until
        finally:
            await self.stop()

    async def submit_goal(self, goal: Goal) -> list[Task]:
        """Submit a goal to the coordinator and return the created tasks."""

        self._validate_dependencies()
        tasks = await self.coordinator.submit_goal(goal)
        for task in tasks:
            self._pending_tasks[task.id] = task
        await self._schedule_ready_tasks()
        return tasks

    def progress(self, goal_id: str) -> dict[str, int]:
        """Return progress information for the given goal."""

        return self.coordinator.progress(goal_id)

    @property
    def agents(self) -> list[Any]:
        """Return a copy of the registered agents."""

        return list(self._agents)

    def _context(self) -> dict[str, dict[str, AgentResult]]:
        """Return the execution context passed to agent task handlers."""

        return {"results": self.results}

    async def _store_result(
        self,
        agent_id: str,
        output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Store the latest result produced by an agent."""

        result = AgentResult(
            agent_id=agent_id,
            output=output,
            metadata={} if metadata is None else metadata,
        )
        self.results[result.agent_id] = result
        await self._schedule_ready_tasks()
        return result

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency refers to a registered agent."""

        known_agent_ids = {agent.id for agent in self._agents}
        for agent in self._agents:
            for dependency in getattr(agent, "depends_on", []):
                if dependency not in known_agent_ids:
                    raise ValueError(
                        f"Agent '{agent.id}' depends on missing agent '{dependency}'."
                    )

    async def _schedule_ready_tasks(self) -> None:
        """Assign all pending tasks whose agent dependencies have completed."""

        ready_tasks: list[Task] = []
        for task in self._pending_tasks.values():
            if task.assigned_to is None:
                ready_tasks.append(task)
                continue

            agent = self.coordinator._agents.get(task.assigned_to)
            if agent is None:
                raise ValueError(f"Task '{task.id}' targets missing agent '{task.assigned_to}'.")

            dependencies = getattr(agent, "depends_on", [])
            if all(dependency in self.results for dependency in dependencies):
                ready_tasks.append(task)

        for task in ready_tasks:
            self._pending_tasks.pop(task.id, None)
            await self.coordinator.assign_task(task)
