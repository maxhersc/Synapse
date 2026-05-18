#!/usr/bin/env python3
"""
Synapse v0.2 — Basic Team Example

A conversational planning agent that chats with the user.
When the user types /task <description>, the coordinator breaks it
into subtasks and dispatches them to a team of specialist agents:

  - Researcher : finds information and summarises findings
  - Writer     : drafts polished documents from research
  - Reviewer   : reviews work and provides feedback

Agents talk to each other — the researcher feeds the writer,
the writer sends drafts to the reviewer, and the reviewer
reports back to the planner.

Run:
    PYTHONPATH=. python3 examples/basic_team.py
"""

from __future__ import annotations

import asyncio
import sys
import json

from synapse import Runtime, SynapseAgent, AgentProfile
from protocols.message import Research, NodeStatus

INPUT_MODE = False
DEVELOPER_MODE = False


# ═══════════════════════════════════════════════════
#  ANSI colours for the terminal
# ═══════════════════════════════════════════════════

COLORS = {
    "planner":     "\033[96m",   # cyan
    "coordinator": "\033[93m",   # yellow
    "researcher":  "\033[92m",   # green
    "writer":      "\033[95m",   # magenta
    "reviewer":    "\033[94m",   # blue
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"

BANNER = f"""
{BOLD}\033[96m
 ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
 ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
 ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
 ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
 ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
{RESET}
{DIM}  v0.3 — Always-On Research Engine{RESET}
{DIM}  Type any question to automatically start a research pipeline.{RESET}
{DIM}  Type /quit to exit. Commands: /settings, /debug{RESET}
"""


def cprint(label: str, text: str) -> None:
    """Coloured print for agent messages."""
    # Extract just the agent name (before →)
    agent = label.split("→")[0].strip().split()[-1] if "→" in label else label
    name_lower = agent.lower()
    
    if name_lower in COLORS:
        color = COLORS[name_lower]
    else:
        # Generate a stable pseudo-random color for dynamic agents (e.g. 91-96)
        color_code = 91 + (hash(name_lower) % 6)
        color = f"\033[{color_code}m"
        
    prefix = f"{color}{BOLD}[{label}]{RESET}"
    # Indent multi-line output
    lines = text.strip().splitlines()
    if len(lines) == 1:
        print(f"{prefix} {lines[0]}")
    else:
        print(prefix)
        for line in lines:
            print(f"  {DIM}│{RESET} {line}")
    print()


# ═══════════════════════════════════════════════════
#  Planning Agent — conversational, user-facing
# ═══════════════════════════════════════════════════

class PlannerAgent(SynapseAgent):
    profile = AgentProfile(
        name="Planner",
        model="gemma3:1b",
        strengths=["planning", "conversation", "clarification"],
        description=(
            "Conversational planning agent. Chats with the user to understand "
            "their needs before any task is triggered. Friendly, curious, concise."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, str]] = []

    async def chat(self, user_input: str) -> str:
        """Have a conversation turn with the user."""
        self.history.append({"role": "user", "content": user_input})

        # Build conversation context
        conv = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Planner'}: {h['content']}"
            for h in self.history[-10:]  # keep last 10 turns
        )

        prompt = (
            f"You are Synapse Planner — a friendly, helpful planning assistant.\n"
            f"You're chatting with a user to understand what they want to build or accomplish.\n"
            f"Be concise (2-3 sentences max). Ask clarifying questions.\n"
            f"Do NOT execute tasks — just have a conversation.\n\n"
            f"Conversation so far:\n{conv}\n\n"
            f"Planner:"
        )

        response = await self.llm(prompt)
        response = response.strip()
        self.history.append({"role": "assistant", "content": response})
        return response

    async def summarize_context(self) -> str:
        """Summarize the conversation history to give context to the team."""
        if not self.history:
            return ""
        
        conv = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Planner'}: {h['content']}"
            for h in self.history
        )
        
        prompt = (
            f"Summarize the following conversation into a brief context string "
            f"that captures the user's goals, preferences, and any constraints.\n\n"
            f"Conversation:\n{conv}\n\n"
            f"Summary:"
        )
        return (await self.llm(prompt)).strip()

    async def handle_task(self, operation) -> str:
        return await self.llm(
            f"You are the Planner. Summarise this result for the user:\n\n{operation.description}"
        )

# ═══════════════════════════════════════════════════
#  Specialist Agents
# ═══════════════════════════════════════════════════

# ═══════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════

async def main() -> None:
    global INPUT_MODE
    print(BANNER)

    # Build the team
    runtime = Runtime()

    planner = PlannerAgent()
    runtime.add_agent(planner)

    # Wire up real-time display
    output_allowed_event = asyncio.Event()
    output_allowed_event.set()

    async def display(label: str, content: str) -> None:
        await output_allowed_event.wait()
        cprint(label, content)

    runtime.on_message(display)

    # Start
    await runtime.run()

    print(f"{COLORS['planner']}{BOLD}[system]{RESET} Synapse Research Engine initialized.\n"
          f"Awaiting queries...\n")

    # Conversation loop
    while True:
        # Check active research first
        if runtime.active_research:
            paused_research = None
            for research in runtime.active_research:
                if research.status == NodeStatus.BLOCKED_PENDING_INPUT:
                    paused_research = research
                    break
                    
            if paused_research:
                # USER_INPUT_MODE
                INPUT_MODE = True
                output_allowed_event.clear()
                
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(f"{BOLD}> {RESET}")
                    )
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{DIM}Goodbye!{RESET}")
                    break
                finally:
                    INPUT_MODE = False
                    output_allowed_event.set()
                    
                user_input = user_input.strip()
                if user_input:
                    print(f"\n{DIM}[runtime]{RESET}\nResuming all blocked agents...\n")
                    
                    existing = paused_research.context.get("shared_input", "")
                    paused_research.context["shared_input"] = f"{existing}\nUser provided: {user_input}".strip()
                    
                    paused_research.status = NodeStatus.RUNNING
                    for op in paused_research.plan:
                        if op.status == NodeStatus.BLOCKED_PENDING_INPUT:
                            op.context["resume_event"].set()
            else:
                # System is executing, do not show prompt
                await asyncio.sleep(0.5)
            continue

        # Top-level conversation (no active goals)
        INPUT_MODE = True
        output_allowed_event.clear()
        
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(f"{BOLD}> {RESET}")
            )
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break
        finally:
            INPUT_MODE = False
            output_allowed_event.set()
            
        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"\n{DIM}Goodbye!{RESET}")
            break
            
        if user_input.lower() == "/settings":
            print(f"\n{DIM}[settings] Configuration mode not yet implemented.{RESET}\n")
            continue
            
        if user_input.lower() == "/debug":
            print(f"\n{DIM}[debug] Diagnostic mode not yet implemented.{RESET}\n")
            continue

        # ─── Implicit Research Trigger ───
        research_question = user_input

        print(f"\n{COLORS['coordinator']}{BOLD}[coordinator]{RESET} "
              f"Received query. Initiating Research Pipeline...\n")

        try:
            # 1. Summarize conversation context
            conv_context = await planner.summarize_context()
            
            # 2. Submit research as a background process
            active_research_obj = await runtime.start_research(research_question, conv_context)
            
            async def wait_for_research(research_obj, desc):
                while research_obj.status not in (NodeStatus.COMPLETED, NodeStatus.FAILED):
                    await asyncio.sleep(0.5)
                    
                if research_obj.status == NodeStatus.COMPLETED:
                    print(f"\n{'─' * 60}")
                    cprint("system", f"Research Complete:\n\n{research_obj.final_output}")
                    print(f"{'─' * 60}\n")
                    planner.history.append({
                        "role": "assistant",
                        "content": f"[Research output for '{desc}']\n{research_obj.final_output}"
                    })
                elif research_obj.status == NodeStatus.FAILED:
                    print(f"\n{BOLD}\033[91m[error]{RESET} Research Pipeline failed.\n")
                    
                if research_obj in runtime.active_research:
                    runtime.active_research.remove(research_obj)

            asyncio.create_task(wait_for_research(active_research_obj, research_question))
            
        except Exception as e:
            print(f"\n{BOLD}\033[91m[error]{RESET} Research initiation failed: {e}\n")

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
