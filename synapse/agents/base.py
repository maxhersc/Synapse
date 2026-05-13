from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from synapse.protocols.message import HelpRequest, Message, Priority, Task, TaskStatus


@dataclass
class AgentProfile:
    """Describes an agent's identity, strengths, and capabilities."""

    name: str
    model: str
    strengths: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    description: str = ""


class SynapseAgent(ABC):
    """Base class for Synapse agents that manages lifecycle, messaging, and shared memory access."""

    def __init__(
        self,
        agent_id: str | None = None,
        profile: AgentProfile | None = None,
    ) -> None:
        resolved_profile = profile if profile is not None else getattr(self.__class__, "profile", None)
        if resolved_profile is None:
            raise ValueError("SynapseAgent requires a profile or class-level profile attribute.")

        self.id: str = agent_id if agent_id is not None else self.__class__.__name__.lower()
        self.profile: AgentProfile = resolved_profile
        self._bus: Any = None
        self._memory: Any = None
        self._coordinator: Any = None
        self._inbox: asyncio.PriorityQueue[Message] = asyncio.PriorityQueue()
        self._running: bool = False
        self._current_task: Task | None = None
        self._runner_task: asyncio.Task[None] | None = None

    def _inject(self, bus: Any, memory: Any, coordinator: Any) -> None:
        """Inject runtime-managed infrastructure into the agent."""

        self._bus = bus
        self._memory = memory
        self._coordinator = coordinator

    async def start(self) -> None:
        """Start the agent's background processing loop."""

        if self._running:
            return
        self._running = True
        self._runner_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the agent's background processing loop and cancel it cleanly."""

        self._running = False
        if self._runner_task is None:
            return
        self._runner_task.cancel()
        try:
            await self._runner_task
        except asyncio.CancelledError:
            pass
        finally:
            self._runner_task = None

    async def _run(self) -> None:
        try:
            await self.on_start()
            while self._running:
                batch: list[Message] = []
                first_message = await self._inbox.get()
                batch.append(first_message)

                for _ in range(9):
                    try:
                        batch.append(self._inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                for message in batch:
                    try:
                        task: Task | None = None
                        if isinstance(message, Task):
                            task = message
                        elif isinstance(message.content, Task):
                            task = message.content
                        else:
                            task_candidate = message.metadata.get("task")
                            if isinstance(task_candidate, Task):
                                task = task_candidate

                        if task is not None:
                            self._current_task = task
                            task.status = TaskStatus.IN_PROGRESS
                            result = await self.handle_task(task)
                            task.complete(result)
                            self._current_task = None
                        else:
                            await self.handle_message(message)
                    except Exception as error:
                        if self._current_task is not None:
                            self._current_task.fail(str(error))
                            self._current_task = None
                        await self.on_error(error)
        except asyncio.CancelledError:
            raise
        finally:
            await self.on_stop()

    async def send_to(
        self,
        recipient_id: str,
        content: str | dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """Send a message to a specific recipient through the bus."""

        message = Message(
            sender_id=self.id,
            recipient_id=recipient_id,
            content=content,
            priority=priority,
        )
        await self._bus.send(message)

    async def broadcast(self, content: str | dict[str, Any]) -> None:
        """Send a broadcast message to all agents through the bus."""

        await self.send_to("broadcast", content)

    async def reply(self, original_message: Message, content: str | dict[str, Any]) -> None:
        """Reply to a message while preserving its thread and priority."""

        await self._bus.send(original_message.reply(sender_id=self.id, content=content))

    async def request_help(
        self,
        task: Task,
        reason: str,
        capability_needed: str | None = None,
    ) -> None:
        """Submit a formal help request to the coordinator for the current task."""

        help_request = HelpRequest(
            from_agent_id=self.id,
            task_id=task.id,
            reason=reason,
            capability_needed=capability_needed,
        )
        await self._coordinator.handle_help_request(help_request)

    async def remember(self, key: str, value: Any) -> None:
        """Store a value in shared memory."""

        await self._memory.set(key, value)

    async def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from shared memory."""

        return await self._memory.get(key, default)

    @abstractmethod
    async def handle_task(self, task: Task) -> str:
        """Handle an assigned task and return its result."""

    async def handle_message(self, message: Message) -> None:
        """Handle a non-task message sent to this agent."""

    async def on_start(self) -> None:
        """Run optional setup logic before the processing loop starts."""

    async def on_stop(self) -> None:
        """Run optional cleanup logic after the processing loop stops."""

    async def on_error(self, error: Exception) -> None:
        """Handle an exception raised during agent execution."""

        print(f"Agent {self.id} error: {error}")
