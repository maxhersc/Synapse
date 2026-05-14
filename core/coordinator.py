"""
Synapse v0.2 — Coordinator.

LLM-powered brain that:
  1. Takes a Goal (user's high-level intent)
  2. Uses gemma3:1b to break it into ordered subtasks
  3. Assigns each subtask to the best-fit agent based on strengths
  4. Executes subtasks sequentially, passing results downstream
  5. Synthesises a final answer from all results
"""

from __future__ import annotations

import asyncio
import httpx
from typing import TYPE_CHECKING

from protocols.message import Goal, Task, TaskStatus, Message

if TYPE_CHECKING:
    from core.bus import MessageBus
    from core.memory import SharedMemory
    from agents.base import SynapseAgent

OLLAMA_URL = "http://localhost:11434/api/generate"
COORDINATOR_MODEL = "gemma3:1b"


class Coordinator:
    """Breaks goals into tasks, assigns them, and orchestrates execution."""

    def __init__(self, bus: "MessageBus", memory: "SharedMemory") -> None:
        self.bus = bus
        self.memory = memory

    # ──────────────────────────────────────────────
    #  LLM call (coordinator uses the lightweight model)
    # ──────────────────────────────────────────────

    async def _llm(self, prompt: str) -> str:
        payload = {"model": COORDINATOR_MODEL, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    # ──────────────────────────────────────────────
    #  Task breakdown
    # ──────────────────────────────────────────────

    async def plan(self, goal: Goal, conversation_context: str = "") -> list[Task]:
        """Use the LLM to decompose a goal into sequential subtasks."""
        available = self.bus.agents
        agent_list = "\n".join(
            f"- {name}: {a.profile.description} (strengths: {', '.join(a.profile.strengths)})"
            for name, a in available.items()
            if name != "planner"  # planner is the user-facing agent
        )

        context_block = ""
        if conversation_context:
            context_block = f"\nContext from conversation with user:\n{conversation_context}\n"

        prompt = (
            f"You are a task coordinator. Break the following goal into 2-4 sequential subtasks.\n"
            f"Each subtask should be assigned to one of the available agents.\n\n"
            f"Available agents:\n{agent_list}\n"
            f"{context_block}\n"
            f"Goal: {goal.description}\n\n"
            f"Reply with ONLY a numbered list in this exact format (no other text):\n"
            f"1. [agent_name] | [task description]\n"
            f"2. [agent_name] | [task description]\n"
            f"...\n"
        )

        raw = await self._llm(prompt)
        return self._parse_plan(raw, goal)

    def _parse_plan(self, raw: str, goal: Goal) -> list[Task]:
        """Parse the LLM's numbered list into Task objects."""
        tasks: list[Task] = []
        available = {name.lower() for name in self.bus.agents if name != "planner"}

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading number and punctuation: "1. researcher | ..."
            # Find the pipe separator
            if "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue

            agent_part = parts[0].strip().lower()
            description = parts[1].strip()

            # Extract agent name — strip leading digits/punctuation
            agent_name = ""
            for token in agent_part.replace(".", " ").replace(")", " ").split():
                cleaned = token.strip().lower()
                if cleaned in available:
                    agent_name = cleaned
                    break

            if not agent_name:
                # Try fuzzy match — pick first agent whose name appears in the text
                for name in available:
                    if name in agent_part:
                        agent_name = name
                        break

            if not agent_name:
                # Fallback: assign to first available agent
                agent_name = next(iter(available)) if available else "unknown"

            task = Task(
                description=description,
                assigned_to=agent_name,
                created_by="coordinator",
            )
            tasks.append(task)

        # If parsing failed completely, create a single task assigned to anyone
        if not tasks and available:
            tasks.append(
                Task(
                    description=goal.description,
                    assigned_to=next(iter(available)),
                    created_by="coordinator",
                )
            )

        return tasks

    # ──────────────────────────────────────────────
    #  Execution
    # ──────────────────────────────────────────────

    async def execute(self, goal: Goal, conversation_context: str = "") -> str:
        """Plan, execute subtasks sequentially, and return clean output."""

        # Announce
        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="planner",
                content=f"Breaking down task: {goal.description}",
                metadata={"system": True},
            )
        )

        # Step 1 — Plan (with conversation context for better subtask descriptions)
        tasks = await self.plan(goal, conversation_context)
        goal.tasks = tasks

        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="planner",
                content=f"Created {len(tasks)} subtasks. Starting execution...",
                metadata={"system": True},
            )
        )

        # Step 2 — Execute each task sequentially, passing context forward
        previous_result = ""
        agent_results: dict[str, str] = {}  # agent_name -> raw result

        for i, task in enumerate(tasks):
            task.status = TaskStatus.IN_PROGRESS

            # Every agent gets the original user goal so they stay on topic
            task.context["goal"] = goal.description

            # Every agent gets the conversation context so they know the user's background
            if conversation_context:
                task.context["conversation"] = conversation_context

            # Give the task agent the previous result as context
            if previous_result:
                task.context["previous_results"] = previous_result

            agent = self.bus.agents.get(task.assigned_to)
            if not agent:
                task.fail(f"Agent '{task.assigned_to}' not found")
                previous_result = f"[{task.assigned_to}] FAILED: agent not found"
                continue

            # Announce the handoff
            if i > 0:
                prev_agent = tasks[i - 1].assigned_to
                await self.bus.dispatch(
                    Message(
                        sender=prev_agent,
                        recipient=task.assigned_to,
                        content=f"Here's what I produced:\n\n{previous_result}",
                        metadata={"handoff": True},
                    )
                )

            try:
                result = await agent.handle_task(task)
                task.complete(result)
                previous_result = result
                agent_results[task.assigned_to] = result

                # Store in shared memory
                await self.memory.set(f"task_{task.task_id}_result", result)

            except Exception as e:
                task.fail(str(e))
                previous_result = f"[{task.assigned_to}] FAILED: {e}"

        # Step 3 — Assemble final output (no extra LLM call)
        # Use the writer's paragraph + reviewer's verdict directly
        goal.status = TaskStatus.COMPLETED

        writer_output = agent_results.get("writer", "")
        reviewer_output = agent_results.get("reviewer", "")

        if writer_output and reviewer_output:
            final = f"{writer_output}\n\n{reviewer_output}"
        elif writer_output:
            final = writer_output
        else:
            # Fallback: use whatever we got
            final = previous_result if previous_result else "No results produced."

        goal.final_result = final
        await self.memory.set(f"goal_{goal.goal_id}_result", final)

        return final
