"""
Base agent abstraction for Synapse.

Agents operate as role-bound workers inside a persistent workspace.
They may still send messages for coordination, but their primary unit
of work is a structured task plus organization and workspace context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import httpx

from protocols.message import (
    HelpRequest,
    Message,
    Permission,
    ResearchOperation,
    Task,
    TaskExecutionResult,
    TaskStatus,
    WorkspaceSnapshot,
)

if TYPE_CHECKING:
    from core.bus import MessageBus

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class AgentProfile:
    """Declares a worker's role, capabilities, and execution guardrails."""

    name: str
    model: str = "gemma3:1b"
    role: str = ""
    department: str = ""
    strengths: list[str] = field(default_factory=list)
    permissions: set[Permission] = field(default_factory=set)
    description: str = ""


class SynapseAgent:
    """Base class for executable Synapse workers."""

    profile: AgentProfile

    def __init__(self) -> None:
        self.bus: Optional[MessageBus] = None
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._help_futures: dict[str, asyncio.Future] = {}
        self._running = False

    async def llm(self, prompt: str, model: str | None = None) -> str:
        target_model = model or self.profile.model
        payload = {"model": target_model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    async def send(self, recipient: str, content: str) -> None:
        msg = Message(
            sender=self.profile.name.lower(),
            recipient=recipient.lower(),
            content=content,
        )
        if self.bus:
            await self.bus.dispatch(msg)

    async def ask(self, recipient: str, question: str, context: str = "") -> str:
        req = HelpRequest(
            from_agent=self.profile.name.lower(),
            to_agent=recipient.lower(),
            question=question,
            context=context,
        )
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._help_futures[req.request_id] = future

        msg = Message(
            sender=self.profile.name.lower(),
            recipient=recipient.lower(),
            content=question,
            metadata={"help_request_id": req.request_id, "context": context},
        )
        if self.bus:
            await self.bus.dispatch(msg)
        return await future

    async def _handle_incoming(self, message: Message) -> None:
        reply_id = message.metadata.get("help_response_id")
        if reply_id and reply_id in self._help_futures:
            self._help_futures[reply_id].set_result(message.content)
            del self._help_futures[reply_id]
            return

        help_id = message.metadata.get("help_request_id")
        if help_id:
            answer = await self.handle_help(message.content, message.sender)
            reply = Message(
                sender=self.profile.name.lower(),
                recipient=message.sender,
                content=answer,
                reply_to=message.message_id,
                metadata={"help_response_id": help_id},
            )
            if self.bus:
                await self.bus.dispatch(reply)
            return

        await self.inbox.put(message)

    async def execute_task(self, task: Task, snapshot: WorkspaceSnapshot) -> TaskExecutionResult:
        """
        Override this to implement structured work.

        The default implementation degrades to a generic LLM summary so the
        runtime remains usable during the transition away from the old model.
        """
        summary = await self.llm(self.build_task_prompt(task, snapshot))
        return TaskExecutionResult(
            status=TaskStatus.COMPLETED,
            summary=summary.strip(),
        )

    def can_accept_task(self, task: Task, snapshot: WorkspaceSnapshot) -> bool:
        required_permission = Permission.REVIEW_WORK if task.task_type.value == "review" else (
            Permission.APPROVE_WORK if task.task_type.value == "approval" else Permission.EXECUTE_TASKS
        )
        if required_permission not in self.profile.permissions:
            return False
        if task.assigned_role_id:
            role = snapshot.organization.roles.get(task.assigned_role_id)
            if role and self.profile.role and role.name.lower() != self.profile.role.lower():
                return False
        return True

    def build_task_prompt(self, task: Task, snapshot: WorkspaceSnapshot) -> str:
        context = snapshot.organization.context
        return (
            f"You are {self.profile.name}, acting as {self.profile.role or self.profile.name} "
            f"inside the organization '{snapshot.organization.name}'.\n"
            f"Mission: {context.mission}\n"
            f"Goals: {', '.join(context.goals) or 'None provided'}\n"
            f"Writing style: {context.writing_style or 'Not specified'}\n"
            f"Communication style: {context.communication_style or 'Not specified'}\n"
            f"Task: {task.title}\n"
            f"Description: {task.description}\n"
            f"Task type: {task.task_type.value}\n"
            f"Deliver a concise execution update suitable for the workspace record."
        )

    async def handle_task(self, task: ResearchOperation) -> str:
        """Legacy compatibility hook for older research-oriented code paths."""
        raise NotImplementedError(
            f"{self.profile.name} has no legacy handle_task implementation"
        )

    async def handle_help(self, question: str, from_agent: str) -> str:
        return await self.llm(
            f"You are {self.profile.name}. {self.profile.description}\n\n"
            f"Agent '{from_agent}' is asking for help:\n{question}\n\n"
            f"Reply concisely and stay within your role permissions."
        )

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for fut in self._help_futures.values():
            if not fut.done():
                fut.cancel()
        self._help_futures.clear()

    @property
    def name(self) -> str:
        return self.profile.name.lower()
