"""
Synapse v0.2 — Base agent class.

Every agent inherits from SynapseAgent. Provides:
  - self.llm(prompt)  — call the local Ollama model
  - self.send(recipient, content) — send a message to another agent
  - self.ask(recipient, question)  — ask another agent for help and wait
  - handle_task(task) — override this to define agent behaviour
"""

from __future__ import annotations

import asyncio
import httpx
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

from protocols.message import Message, ResearchOperation, HelpRequest

if TYPE_CHECKING:
    from core.bus import MessageBus

OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class AgentProfile:
    """Declares who an agent is and what it's good at."""

    name: str
    model: str = "gemma3:1b"
    strengths: list[str] = field(default_factory=list)
    description: str = ""


class SynapseAgent:
    """Base class for all Synapse agents."""

    profile: AgentProfile  # subclasses set this as a class attribute

    def __init__(self) -> None:
        self.bus: Optional[MessageBus] = None
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._help_futures: dict[str, asyncio.Future] = {}
        self._running = False

    # ──────────────────────────────────────────────
    #  LLM helper
    # ──────────────────────────────────────────────

    async def llm(self, prompt: str, model: str | None = None) -> str:
        """Call the local Ollama instance with a prompt and return the response text."""
        target_model = model or self.profile.model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    # ──────────────────────────────────────────────
    #  Messaging helpers
    # ──────────────────────────────────────────────

    async def send(self, recipient: str, content: str) -> None:
        """Send a natural-language message to another agent."""
        msg = Message(
            sender=self.profile.name.lower(),
            recipient=recipient.lower(),
            content=content,
        )
        if self.bus:
            await self.bus.dispatch(msg)

    async def ask(self, recipient: str, question: str, context: str = "") -> str:
        """Ask another agent for help and wait for their response."""
        req = HelpRequest(
            from_agent=self.profile.name.lower(),
            to_agent=recipient.lower(),
            question=question,
            context=context,
        )
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._help_futures[req.request_id] = future

        # Deliver the question as a message
        msg = Message(
            sender=self.profile.name.lower(),
            recipient=recipient.lower(),
            content=f"[help-request:{req.request_id}] {question}",
            metadata={"help_request_id": req.request_id, "context": context},
        )
        if self.bus:
            await self.bus.dispatch(msg)

        # Wait for the response
        return await future

    async def _handle_incoming(self, message: Message) -> None:
        """Process an incoming message. If it's a help-request, handle and reply."""
        # Check if this is a response to one of our help requests
        reply_id = message.metadata.get("help_response_id")
        if reply_id and reply_id in self._help_futures:
            self._help_futures[reply_id].set_result(message.content)
            del self._help_futures[reply_id]
            return

        # Check if this is a help request we need to answer
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

        # Normal message — put it in the inbox for handle_task to consume
        await self.inbox.put(message)

    # ──────────────────────────────────────────────
    #  Override points
    # ──────────────────────────────────────────────

    async def handle_task(self, task: ResearchOperation) -> str:
        """Override this to define what the agent does with a task."""
        raise NotImplementedError(
            f"{self.profile.name} has no handle_task implementation"
        )

    async def handle_help(self, question: str, from_agent: str) -> str:
        """Override to customise how this agent responds to help requests."""
        return await self.llm(
            f"You are {self.profile.name}. {self.profile.description}\n\n"
            f"Agent '{from_agent}' is asking for your help:\n{question}\n\n"
            f"Provide a helpful, concise response."
        )

    # ──────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        # Cancel pending help futures
        for fut in self._help_futures.values():
            if not fut.done():
                fut.cancel()
        self._help_futures.clear()

    @property
    def name(self) -> str:
        return self.profile.name.lower()
