# Synapse

**A coordination layer for AI agent teams.**

Synapse is not a framework for building individual agents — it is the operating layer that lets multiple LLM-powered agents work together as a structured system.

Instead of manually chaining prompts or building brittle workflows, Synapse treats agents like members of an organization: each one has a role, a set of capabilities, and access to shared context. You define the team and the goal. Synapse handles the rest.

---

## Install

```bash
pip install synapse-agents
```

Or run from source:

```bash
git clone https://github.com/maxhersc/synapse.git
cd synapse
PYTHONPATH=. python3 examples/basic_team.py
```

---

## Quickstart

```python
from synapse import Runtime, Goal, SynapseAgent, AgentProfile
import asyncio

class ResearcherAgent(SynapseAgent):
    profile = AgentProfile(
        name="Researcher",
        model="claude-sonnet-4-20250514",
        strengths=["research", "summarization"],
        capabilities=["web_search"],
    )

    async def handle_task(self, task):
        # Replace with a real LLM call
        return "Amadeus, Skyscanner, and Booking.com are the top travel APIs."

class WriterAgent(SynapseAgent):
    profile = AgentProfile(
        name="Writer",
        model="claude-sonnet-4-20250514",
        strengths=["writing", "summarization"],
        capabilities=["document_writing"],
    )

    async def handle_task(self, task):
        return "Summary written based on research findings."

async def main():
    runtime = Runtime()
    runtime.add(ResearcherAgent("researcher", ResearcherAgent.profile, runtime.bus, runtime.memory, runtime.coordinator))
    runtime.add(WriterAgent("writer", WriterAgent.profile, runtime.bus, runtime.memory, runtime.coordinator))

    goal = Goal(description="Research and summarize the top travel APIs")
    tasks = await runtime.submit_goal(goal)

    done = asyncio.get_event_loop().create_future()
    asyncio.get_event_loop().call_later(3, done.set_result, None)
    await runtime.run(until=done)

    print(runtime.progress(goal.id))

asyncio.run(main())
```

---

## How It Works

1. Developer defines a **Goal** and a **team of agents**
2. The **Coordinator** breaks the goal into **Tasks**
3. Tasks are assigned based on each agent's **strengths and capabilities**
4. Agents execute tasks and return results
5. Agents can **request help** if they are stuck
6. The system runs until all tasks are complete

---

## Core Components

| Component | Responsibility |
|---|---|
| `SynapseAgent` | Base class for all agents — subclass and implement `handle_task()` |
| `AgentProfile` | Declares an agent's model, strengths, and capabilities |
| `Coordinator` | Assigns tasks, tracks progress, handles help requests |
| `Bus` | Async message routing between agents |
| `SharedMemory` | Global key/value store all agents can read and write |
| `Runtime` | Entry point — owns all components, starts and stops the system |
| `Goal` | The top-level objective submitted to the team |
| `Task` | A unit of work with an owner, status, and result |

---

## Design Principles

- **Framework-agnostic** — works with Claude, GPT, local models, or any LLM
- **Zero required dependencies** — pure Python stdlib core
- **Agents as team members** — roles, capabilities, and shared context built in
- **Observable by default** — every message and task state change is logged
- **Developer-first API** — subclass, implement one method, and go

---

## Use Cases

- Multi-step software development pipelines
- Research and synthesis systems
- Automated content workflows
- Complex planning and execution tasks
- Any system where multiple AI agents need to collaborate

---

## Project Status

**v0.1 — Early access.** Core architecture is stable and running. APIs may change before v1.0.

---

## License

MIT © 2026 Max Herscovitch