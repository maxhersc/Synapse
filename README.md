# Synapse

**The coordination layer for AI agent teams.**

Synapse is not a framework for building individual agents. It is the operating layer that lets multiple LLM-powered agents work together as a structured team — with shared goals, task assignment, result passing, and shared memory.

Instead of manually chaining prompts or wiring brittle workflows, you define a team and a goal. Synapse handles the rest.

---

## Why Synapse

Every existing tool for multi-agent AI — LangChain, CrewAI, AutoGen — bundles communication and coordination inside their own framework. You're locked in the moment you pick one.

Synapse is different. It is the coordination layer that works with any framework, any LLM, and any agent you already have. Bring your own models. Bring your own logic. Synapse handles how they work together.

---

## How It Works

1. You define agents — each with a role, strengths, and capabilities
2. You give the team a goal
3. Synapse breaks the goal into tasks and assigns them based on each agent's profile
4. Agents execute in sequence — each one receiving the previous agent's output
5. Results flow through the team until the goal is complete

---

## Quickstart

**Clone the repo:**
```bash
git clone https://github.com/maxhersc/synapse.git
cd synapse
```

**Install dependencies:**
```bash
pip install httpx
```

**Run the example with Ollama (local, no API key needed):**

First make sure Ollama is running with a model pulled:
```bash
ollama pull gemma3:4b
ollama serve
```

Then in a new terminal tab:
```bash
PYTHONPATH=. python3 examples/basic_team.py
```

**Or use Claude** by replacing the `call_ollama()` helper in your agent with an Anthropic API call:
```python
import anthropic

async def call_claude(prompt: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

---

## Defining Your Own Agents

Create a new project folder anywhere on your machine — no need to touch Synapse's source code:

```
my_project/
├── my_team.py
└── agents/
    ├── researcher.py
    ├── writer.py
    └── reviewer.py
```

**`agents/researcher.py`:**
```python
from synapse import SynapseAgent, AgentProfile

class ResearcherAgent(SynapseAgent):
    profile = AgentProfile(
        name="Researcher",
        model="gemma3:4b",
        strengths=["research", "summarization"],
        capabilities=["web_search"],
    )

    async def handle_task(self, task):
        # Call your LLM here
        return await call_ollama(task.description)
```

**`my_team.py`:**
```python
from synapse import Runtime, Goal
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent
import asyncio

async def main():
    runtime = Runtime()
    runtime.add(ResearcherAgent())
    runtime.add(WriterAgent())
    runtime.add(ReviewerAgent())

    goal = Goal(description="Research and write a report on the best travel APIs")
    await runtime.submit_goal(goal)
    await runtime.run()

asyncio.run(main())
```

**Run it:**
```bash
python3 my_team.py
```

---

## What You See

```
[synapse] Starting team...
[synapse] Goal: Research and write a report on the best travel APIs

[researcher] Working...
[researcher] Done: Amadeus, Skyscanner, and Expedia are the top travel APIs...

[writer] Received research. Writing...
[writer] Done: Here is a structured summary of the top travel APIs...

[reviewer] Reviewing...
[reviewer] Done: The summary is clear and accurate. Recommended improvements...

[synapse] Goal complete.
```

---

## Core Components

| Component | Responsibility |
|---|---|
| `SynapseAgent` | Base class — subclass and implement `handle_task()` |
| `AgentProfile` | Declares an agent's model, strengths, and capabilities |
| `Coordinator` | Assigns tasks and tracks progress |
| `Bus` | Async message routing between agents |
| `SharedMemory` | Global state any agent can read and write |
| `Runtime` | Entry point — owns all components, runs the system |
| `Goal` | The objective submitted to the team |
| `Task` | A unit of work with an owner, status, and result |

---

## Design Principles

- **Framework-agnostic** — works with Claude, Ollama, GPT, or any LLM
- **Zero required dependencies** for the core — pure Python stdlib
- **Agents as team members** — roles, capabilities, and shared context built in
- **Results flow automatically** — each agent receives the previous agent's output
- **DAG execution** — tasks run in the right order with parallel support where possible
- **Observable by default** — full execution trace available via `runtime.trace`

---

## Project Status

**v0.1 — Early access.** Core architecture is stable and running. APIs may change before v1.0.

This is an early release. Feedback, issues, and contributions are welcome.

---

## Contributing

Synapse is early stage and moving fast. If you find a bug, have a feature request, or want to contribute — open an issue or pull request. All contributions welcome.

---

## License

MIT © 2026 Max Herscovitch