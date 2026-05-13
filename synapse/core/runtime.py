from __future__ import annotations

import asyncio
from typing import Any

from synapse.core.bus import Bus
from synapse.core.coordinator import Coordinator
from synapse.core.memory import SharedMemory
from synapse.protocols.message import Goal, Task


class Runtime:
    """Entry point that wires together Synapse core components and agent lifecycle management."""

    def __init__(self) -> None:
        self.memory = SharedMemory()
        self.bus = Bus()
        self.coordinator = Coordinator(self.bus, self.memory)
        self._agents: list[Any] = []

    def add(self, agent: Any) -> Runtime:
        """Register an agent with the runtime, bus, and coordinator."""

        agent._inject(self.bus, self.memory, self.coordinator)
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

        return await self.coordinator.submit_goal(goal)

    def progress(self, goal_id: str) -> dict[str, int]:
        """Return progress information for the given goal."""

        return self.coordinator.progress(goal_id)

    @property
    def agents(self) -> list[Any]:
        """Return a copy of the registered agents."""

        return list(self._agents)
