from __future__ import annotations

import asyncio
from typing import Any, Callable

from synapse.protocols.message import Message


class Bus:
    """Central router that delivers messages between attached Synapse agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._observers: list[Callable[..., Any]] = []
        self._log: list[Message] = []

    def attach(self, agent: Any) -> None:
        """Register an agent with the bus."""

        self._agents[agent.id] = agent

    def detach(self, agent_id: str) -> None:
        """Unregister an agent from the bus."""

        self._agents.pop(agent_id, None)

    async def send(self, message: Message) -> None:
        """Route a message, log it, and notify recipients and observers in parallel."""

        self._log.append(message)

        deliveries: list[Any] = []
        observers = [callback(message) for callback in self._observers]

        if message.recipient_id == "broadcast":
            for agent in self._agents.values():
                if agent.id != message.sender_id:
                    deliveries.append(agent._inbox.put(message))
        elif message.recipient_id.startswith("role:"):
            role = message.recipient_id.split(":", 1)[1]
            for agent in self._agents.values():
                agent_role = getattr(agent.profile, "role", None)
                if agent_role == role or agent.profile.description == role:
                    deliveries.append(agent._inbox.put(message))
        else:
            agent = self._agents.get(message.recipient_id)
            if agent is not None:
                deliveries.append(agent._inbox.put(message))

        await asyncio.gather(*deliveries, *observers, return_exceptions=True)

    async def broadcast(self, sender_id: str, content: str | dict[str, Any]) -> None:
        """Send a broadcast message to all attached agents except the sender."""

        await self.send(
            Message(
                sender_id=sender_id,
                recipient_id="broadcast",
                content=content,
            )
        )

    def observe(self, callback: Callable[..., Any]) -> None:
        """Attach an async observer that is called for every message."""

        self._observers.append(callback)

    def history(
        self,
        thread_id: str | None = None,
        sender_id: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Return logged messages filtered by thread, sender, and optional limit."""

        messages = self._log
        if thread_id is not None:
            messages = [message for message in messages if message.thread_id == thread_id]
        if sender_id is not None:
            messages = [message for message in messages if message.sender_id == sender_id]
        if limit is not None:
            messages = messages[-limit:]
        return list(messages)
