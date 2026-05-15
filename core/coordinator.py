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

from protocols.message import Goal, Task, TaskStatus, Message, ScopeContract
from agents.base import SynapseAgent, AgentProfile

if TYPE_CHECKING:
    from core.bus import MessageBus
    from core.memory import SharedMemory

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
        """Use the LLM to decompose a goal into non-overlapping subtasks."""
        context_block = ""
        if conversation_context:
            context_block = f"\nContext from conversation with user:\n{conversation_context}\n"

        prompt = (
            f"You are a task planner. Break the following goal into 2-4 distinct, parallel subtasks.\n"
            f"Assign each subtask a CONCRETE, TASK-SPECIFIC agent name (e.g., 'Flight Agent', 'Database Architect', 'UI Designer').\n"
            f"DO NOT use generic roles like 'Researcher', 'Writer', 'Reviewer', 'Execution Agent', or 'Verification Agent'.\n"
            f"ALSO, define a REQUIRED INPUT SCHEMA for this entire goal.\n"
            f"List any data fields that the agents will need to complete their tasks.\n"
            f"{context_block}\n"
            f"Goal: {goal.description}\n\n"
            f"Reply with ONLY this exact format (no other text):\n"
            f"SCHEMA:\n"
            f"- [field_name] (required)\n"
            f"- [field_name] (optional)\n"
            f"TASKS:\n"
            f"1. [Specific Agent Name] | [task description]\n"
            f"   ALLOWED: [only these outputs]\n"
            f"   FORBIDDEN: [explicit list of forbidden outputs]\n"
            f"   MAX_RESPONSIBILITY: [what it is allowed to decide]\n"
            f"   FORMAT: [strict output schema]\n"
            f"2. [Specific Agent Name] | [task description]\n"
            f"...\n"
        )

        raw = await self._llm(prompt)
        return self._parse_plan(raw, goal)

    def _parse_plan(self, raw: str, goal: Goal) -> list[Task]:
        """Parse the LLM's output into Task objects and extract schema."""
        tasks: list[Task] = []
        mode = "tasks"
        current_task = None

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("SCHEMA:"):
                mode = "schema"
                continue
            elif line.startswith("TASKS:"):
                mode = "tasks"
                continue
                
            if mode == "schema" and line.startswith("- "):
                field_name = line[2:].strip()
                goal.schema.append(field_name)
                continue
                
            if mode == "tasks":
                if line.startswith("ALLOWED:"):
                    if current_task and current_task.contract: current_task.contract.allowed_outputs = line[8:].strip()
                elif line.startswith("FORBIDDEN:"):
                    if current_task and current_task.contract: current_task.contract.forbidden_outputs = line[10:].strip()
                elif line.startswith("MAX_RESPONSIBILITY:"):
                    if current_task and current_task.contract: current_task.contract.max_responsibility = line[19:].strip()
                elif line.startswith("FORMAT:"):
                    if current_task and current_task.contract: current_task.contract.output_format = line[7:].strip()
                elif "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) != 2:
                        continue

                    agent_name = parts[0].strip()
                    # Strip leading number and punctuation
                    while agent_name and not agent_name[0].isalpha():
                        agent_name = agent_name[1:].strip()

                    description = parts[1].strip()

                    if not agent_name:
                        agent_name = "Execution Agent"

                    task = Task(
                        description=description,
                        assigned_to=agent_name,
                        created_by="coordinator",
                        contract=ScopeContract()
                    )
                    current_task = task
                    tasks.append(task)

        # If parsing failed completely, create a single generic task
        if not tasks:
            tasks.append(
                Task(
                    description=goal.description,
                    assigned_to="Execution Agent",
                    created_by="coordinator",
                )
            )

        return tasks

    # ──────────────────────────────────────────────
    #  Execution
    # ──────────────────────────────────────────────

    async def execute(self, goal: Goal, conversation_context: str = "") -> str:
        """Plan, execute subtasks independently, and return merged output."""

        # Announce
        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="planner",
                content=f"Breaking down task: {goal.description}",
                metadata={"system": True},
            )
        )

        # Step 1 — Plan
        tasks = await self.plan(goal, conversation_context)
        goal.tasks = tasks

        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="planner",
                content=f"Created {len(tasks)} non-overlapping subtasks. Starting independent execution...",
                metadata={"system": True},
            )
        )

        # Define DynamicAgent class once
        class DynamicAgent(SynapseAgent):
            def __init__(self, role_name: str, description: str):
                super().__init__()
                self.profile = AgentProfile(
                    name=role_name,
                    model="gemma3:4b",
                    strengths=["execution"],
                    description=description
                )

        # Step 2 — Execute all tasks in parallel
        agent_results: dict[str, str] = {}

        async def run_task(task: Task) -> tuple[str, str]:
            task.status = TaskStatus.RUNNING
            task.context["goal"] = goal.description
            if conversation_context:
                task.context["conversation"] = conversation_context

            # Create an event for pausing
            task.context["resume_event"] = asyncio.Event()

            agent = DynamicAgent(task.assigned_to, f"Task-specific agent: {task.assigned_to}")
            self.bus.register(agent)

            await self.bus.dispatch(
                Message(
                    sender="coordinator",
                    recipient=agent.name,
                    content=f"Started parallel execution: {task.description}",
                    metadata={"system": True},
                )
            )

            try:
                g = task.context.get("goal", task.description)
                c = task.context.get("conversation", "")
                
                while True:
                    schema_str = "\n- ".join(goal.schema) if goal.schema else "None"
                    
                    contract_text = ""
                    if task.contract:
                        contract_text = (
                            f"HARD SCOPE CONTRACT:\n"
                            f"ALLOWED OUTPUTS: {task.contract.allowed_outputs}\n"
                            f"FORBIDDEN OUTPUTS: {task.contract.forbidden_outputs}\n"
                            f"MAX RESPONSIBILITY: {task.contract.max_responsibility}\n"
                            f"OUTPUT FORMAT: {task.contract.output_format}\n\n"
                            f"CRITICAL RULE: You must act as a bounded function. Do not act as a full assistant. You MUST NOT exceed your max responsibility or produce forbidden outputs.\n\n"
                        )
                        
                    prompt = (
                        f"You are the {agent.profile.name}. Your task is: {task.description}\n"
                        f"Overall goal: {g}\n"
                        f"Context: {c}\n"
                        f"Global Input Schema:\n- {schema_str}\n\n"
                        f"{contract_text}"
                        f"You MUST output your final result or block request as a strict JSON object (NO markdown formatting, just raw JSON):\n"
                        f"{{\n"
                        f"  \"agent\": \"{agent.profile.name}\",\n"
                        f"  \"type\": \"artifact\",\n"
                        f"  \"status\": \"complete | blocked | failed\",\n"
                        f"  \"data\": {{ ... structured data ... }}\n"
                        f"}}\n\n"
                        f"If you are missing REQUIRED schema fields, use 'status': 'blocked', and inside 'data' include 'missing_fields': [list of missing field names].\n"
                        f"If you have enough information, use 'status': 'complete' and put your specific structured artifact in 'data'.\n"
                    )
                    result = await agent.llm(prompt)
                    
                    try:
                        import json
                        cleaned_result = result.strip()
                        if cleaned_result.startswith("```json"):
                            cleaned_result = cleaned_result[7:]
                        elif cleaned_result.startswith("```"):
                            cleaned_result = cleaned_result[3:]
                        if cleaned_result.endswith("```"):
                            cleaned_result = cleaned_result[:-3]
                            
                        parsed = json.loads(cleaned_result.strip())
                        
                        if not all(k in parsed for k in ["agent", "type", "status", "data"]):
                            raise ValueError("Missing required JSON fields: agent, type, status, data.")
                            
                        status = parsed.get("status")
                        
                        if status == "blocked":
                            missing = parsed.get("data", {}).get("missing_fields", [])
                            if not missing:
                                raise ValueError("Blocked status requires 'missing_fields' array inside 'data'.")
                                
                            task.status = TaskStatus.BLOCKED_PENDING_INPUT
                            task.context["missing_fields"] = missing
                            
                            task.context["resume_event"].clear()
                            await task.context["resume_event"].wait()
                            
                            task.status = TaskStatus.RUNNING
                            c += "\n\nShared Input State Updated:\n" + goal.context.get("shared_input", "")
                            continue
                            
                        elif status == "complete":
                            # Output validation layer
                            if task.contract:
                                validation_prompt = (
                                    f"You are a strict Output Validator.\n"
                                    f"Analyze this output against the following contract:\n"
                                    f"ALLOWED: {task.contract.allowed_outputs}\n"
                                    f"FORBIDDEN: {task.contract.forbidden_outputs}\n"
                                    f"MAX RESPONSIBILITY: {task.contract.max_responsibility}\n\n"
                                    f"Output to analyze (JSON data):\n{json.dumps(parsed['data'], indent=2)}\n\n"
                                    f"Did the output violate the contract (e.g. by producing forbidden content, exceeding responsibility, or hallucinating outside scope)?\n"
                                    f"Reply 'VALID' if it is strictly within scope.\n"
                                    f"Reply 'VIOLATION: [reason]' if it breached the contract."
                                )
                                validation_result = await self._llm(validation_prompt)
                                if validation_result.strip().startswith("VIOLATION:"):
                                    c += f"\n\nSystem Error: Your previous output violated the contract.\nValidator notes: {validation_result.strip()}\nPlease rewrite and STRICTLY obey your FORBIDDEN OUTPUTS and MAX RESPONSIBILITY."
                                    continue
                                    
                            # Persistent execution log
                            goal.context.setdefault("execution_log", []).append(parsed)
                            
                            final_output_str = json.dumps(parsed, indent=2)
                            task.complete(final_output_str)
                            await self.memory.set(f"task_{task.task_id}_result", final_output_str)
                            
                            await self.bus.dispatch(
                                Message(
                                    sender=agent.name,
                                    recipient="coordinator",
                                    content=f"Artifact completed:\n\n{final_output_str}",
                                )
                            )

                            return (task.assigned_to, final_output_str)
                            
                        elif status == "failed":
                            task.fail("Agent self-reported failure.")
                            return (task.assigned_to, "FAILED: Agent self-reported failure.")
                            
                        else:
                            raise ValueError(f"Invalid status: {status}")
                            
                    except json.JSONDecodeError as e:
                        c += f"\n\nSystem Error: Output was not valid JSON. You MUST output strict JSON.\nError: {e}\nRewrite your response."
                        continue
                    except ValueError as e:
                        c += f"\n\nSystem Error: JSON violated schema.\nError: {e}\nRewrite your response."
                        continue
            except Exception as e:
                task.fail(str(e))
                return (task.assigned_to, f"FAILED: {e}")
            finally:
                self.bus.unregister(agent.name)

        async def monitor_merge() -> None:
            printed_waiting = False
            while not all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in tasks):
                # Check if all non-finished tasks are BLOCKED_PENDING_INPUT
                non_finished = [t for t in tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]
                
                if non_finished and all(t.status == TaskStatus.BLOCKED_PENDING_INPUT for t in non_finished):
                    if not printed_waiting:
                        # Aggregate and deduplicate missing fields
                        missing = set()
                        for t in non_finished:
                            missing.update(t.context.get("missing_fields", []))
                            
                        missing_str = ", ".join(missing)
                        
                        # Generate consolidated question
                        q_prompt = (
                            f"You are the Coordinator. The following required fields are missing from the user: {missing_str}\n"
                            f"Ask ONE consolidated, natural-language question to get this information."
                        )
                        consolidated_q = await self._llm(q_prompt)
                        
                        await self.bus.dispatch(
                            Message(
                                sender="coordinator", 
                                recipient="user", 
                                content=f"{consolidated_q.strip()}"
                            )
                        )
                        goal.status = TaskStatus.BLOCKED_PENDING_INPUT
                        printed_waiting = True
                else:
                    goal.status = TaskStatus.RUNNING
                    printed_waiting = False
                    
                await asyncio.sleep(0.5)

        monitor = asyncio.create_task(monitor_merge())
        results = await asyncio.gather(*(run_task(t) for t in tasks))
        monitor.cancel()

        for name, res in results:
            agent_results[name] = res

        # Step 3 — Assemble final output
        import json
        execution_log = goal.context.get("execution_log", [])
        valid_artifacts = [item for item in execution_log if item.get("status") == "complete"]

        if not valid_artifacts:
            merged_data = "No valid completed artifacts available."
        else:
            all_outputs = []
            for artifact in valid_artifacts:
                agent_name = artifact.get("agent", "Unknown")
                data_str = json.dumps(artifact.get("data", {}), indent=2)
                all_outputs.append(f"--- Output from {agent_name} ---\n{data_str}\n")

            merge_prompt = (
                f"You are the Final Planner. The user's original goal was: {goal.description}\n"
                f"The execution agents have produced the following distinct outputs:\n\n"
                f"{''.join(all_outputs)}\n\n"
                f"Combine these outputs into a single cohesive structured JSON object (or raw structured data).\n"
                f"CRITICAL RULE: You may ONLY combine the outputs as-is. Do NOT modify the content. Do NOT add new information, commentary, or summaries.\n"
                f"Do not include the raw individual agent tags in the final output."
            )
            
            await self.bus.dispatch(
                Message(
                    sender="coordinator",
                    recipient="planner",
                    content="All subtasks completed. Merging outputs...",
                    metadata={"system": True},
                )
            )
            merged_data = await self._llm(merge_prompt)

        # Step 4 — Render Agent
        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="render_agent",
                content="Formatting final output for user...",
                metadata={"system": True},
            )
        )
        
        render_prompt = (
            f"You are the Render Agent.\n"
            f"Your responsibility is to take the following merged structured data and convert it into a highly readable, organized, and beautiful Markdown format for the user.\n"
            f"CRITICAL RULES:\n"
            f"- Organize content into clear sections with headers.\n"
            f"- Ensure clarity and structure.\n"
            f"- You MUST NOT run new tasks, modify the underlying data's meaning, or add hallucinated information.\n"
            f"- You MUST NOT ask user questions.\n"
            f"- You MUST NOT output raw JSON or metadata.\n"
            f"\n"
            f"Merged Data from Planner:\n"
            f"{merged_data}\n\n"
            f"Output ONLY the final Markdown formatted result."
        )
        
        final_rendered = None
        render_agent = DynamicAgent("Render Agent", "Formats merged structured data into user-facing output.")
        self.bus.register(render_agent)
        
        try:
            final_rendered = await render_agent.llm(render_prompt)
            if not final_rendered or not final_rendered.strip():
                final_rendered = "RENDER STEP MISSING — CANNOT FINALIZE OUTPUT"
        except Exception:
            final_rendered = "RENDER STEP MISSING — CANNOT FINALIZE OUTPUT"
        finally:
            self.bus.unregister(render_agent.name)
            
        await self.bus.dispatch(
            Message(
                sender="render_agent",
                recipient="coordinator",
                content="Render complete.",
            )
        )

        goal.final_result = final_rendered
        await self.memory.set(f"goal_{goal.goal_id}_result", final_rendered)
        
        goal.status = TaskStatus.COMPLETED

        return final_rendered
