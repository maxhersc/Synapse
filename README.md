# Synapse

**Synapse is a coordination layer for AI agent teams.**

It is not a framework for building individual agents — it is an operating layer that lets multiple LLM-powered agents work together as a structured system.

---

## Overview

Synapse enables developers to define a goal and a team of agents, and then delegates planning, task assignment, execution, and recovery automatically.

Instead of manually chaining prompts or building brittle workflows, Synapse treats agents like members of an organization with roles, capabilities, and shared context.

---

## Core Idea

* Agents are long-lived workers powered by LLMs
* A coordinator assigns and manages tasks
* A message bus handles communication
* Shared memory stores global state
* Tasks are dynamic and can be split, reassigned, or escalated

---

## Architecture

### Core Components

* **Agent** — LLM-powered worker that executes tasks
* **AgentProfile** — defines model, strengths, and capabilities
* **Coordinator** — assigns tasks and manages execution flow
* **Task** — unit of work with lifecycle and ownership
* **Goal** — high-level objective for the system
* **Message Bus** — async communication layer between agents
* **SharedMemory** — global state store for coordination
* **Runtime** — system entry point and execution engine

---

## Execution Model

1. Developer defines a **Goal** and a **Team of Agents**
2. Coordinator breaks goal into **Tasks**
3. Tasks are assigned based on **AgentProfile capabilities**
4. Agents execute tasks and emit results
5. Tasks may spawn sub-tasks or request help
6. System continues until goal completion criteria are met

---

## Key Design Principles

* Framework-agnostic core (pure Python)
* Minimal external dependencies
* Event-driven coordination
* Agents can request help dynamically
* Shared memory is explicit, not hidden in prompts
* Observability is built-in from the ground up

---

## Developer API (Conceptual)

```python
team = Team(
    agents=[
        ResearchAgent(),
        CodingAgent(),
        WriterAgent(),
    ]
)

team.run(
    goal="Build a REST API with documentation"
)
```

---

## Use Cases

* Multi-step software development
* Research and synthesis systems
* Automated content pipelines
* Complex planning tasks
* AI-driven workflows requiring collaboration

---

## Project Status

Early-stage architecture design (v0.1 redesign in progress).
Core abstractions are being finalized before full implementation.

---

## Vision

Synapse is the coordination layer for AI systems — enabling agents to operate not as isolated tools, but as structured teams with shared goals, memory, and communication.
