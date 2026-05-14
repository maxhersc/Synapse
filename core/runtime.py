"""
Synapse v0.2 — Runtime.

The top-level orchestrator that owns the bus, memory, coordinator,
and all registered agents. Provides submit_goal() and run().
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from protocols.message import Goal, Message
from agents.base import SynapseAgent
from core.bus import MessageBus
from core.memory import SharedMemory
from core.coordinator import Coordinator


class Runtime:
    """Owns everything. Start here."""

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.memory = SharedMemory()
        self.coordinator = Coordinator(self.bus, self.memory)
        self._agents: list[SynapseAgent] = []
        self._goal_queue: asyncio.Queue[Goal] = asyncio.Queue()
        self._display: Callable[[str, str], Awaitable[None]] | None = None

    # ──────────────────────────────────────────────
    #  Agent management
    # ──────────────────────────────────────────────

    def add_agent(self, agent: SynapseAgent) -> None:
        """Register an agent with the runtime."""
        self._agents.append(agent)
        self.bus.register(agent)

    # ──────────────────────────────────────────────
    #  Goal submission
    # ──────────────────────────────────────────────

    async def submit_goal(self, description: str, conversation_context: str = "") -> str:
        """Submit a high-level goal and wait for the result."""
        goal = Goal(description=description)
        result = await self.coordinator.execute(goal, conversation_context)
        return result

    # ──────────────────────────────────────────────
    #  Display hook
    # ──────────────────────────────────────────────

    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a display callback: async def callback(sender, content)."""
        self._display = callback

        async def _observer(msg: Message) -> None:
            # Don't show internal planner inbox messages
            if msg.metadata.get("silent"):
                return
            await callback(
                f"{msg.sender} → {msg.recipient}" if msg.recipient else msg.sender,
                msg.content,
            )

        self.bus.add_observer(_observer)

    # ──────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start all agents."""
        for agent in self._agents:
            await agent.start()

    async def stop(self) -> None:
        """Stop all agents."""
        for agent in self._agents:
            await agent.stop()

    async def run(self) -> None:
        """Start the runtime — agents are ready, waiting for goals."""
        await self.start()
