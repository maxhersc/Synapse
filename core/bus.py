"""
Synapse v0.2 — Message bus.

Routes messages between agents with observer support for real-time display.
All dispatching is async; observers see every message that flows through the system.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, TYPE_CHECKING

from protocols.message import Message

if TYPE_CHECKING:
    from agents.base import SynapseAgent

# Observer signature: async def observer(message: Message) -> None
Observer = Callable[[Message], Awaitable[None]]


class MessageBus:
    """Central message router for the agent swarm."""

    def __init__(self) -> None:
        self._agents: dict[str, "SynapseAgent"] = {}
        self._observers: list[Observer] = []

    def register(self, agent: "SynapseAgent") -> None:
        """Register an agent so it can receive messages."""
        self._agents[agent.name] = agent
        agent.bus = self

    def unregister(self, name: str) -> None:
        agent = self._agents.pop(name, None)
        if agent:
            agent.bus = None

    def add_observer(self, observer: Observer) -> None:
        """Add a function that sees every message (for logging / UI)."""
        self._observers.append(observer)

    async def dispatch(self, message: Message) -> None:
        """Route a message to its recipient and notify all observers."""
        # Notify observers first (so the UI sees it immediately)
        for obs in self._observers:
            try:
                await obs(message)
            except Exception:
                pass  # observers should never break the pipeline

        # Deliver to recipient
        recipient = self._agents.get(message.recipient)
        if recipient:
            await recipient._handle_incoming(message)

    async def broadcast(self, sender: str, content: str, exclude: set[str] | None = None) -> None:
        """Send a message to every registered agent (except those in exclude)."""
        exclude = exclude or set()
        for name, agent in self._agents.items():
            if name != sender and name not in exclude:
                msg = Message(sender=sender, recipient=name, content=content)
                await self.dispatch(msg)

    @property
    def agents(self) -> dict[str, "SynapseAgent"]:
        return dict(self._agents)
