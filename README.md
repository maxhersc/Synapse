# Synapse

A research-first AI system that transforms queries into structured, verifiable research reports using multi-stage agent pipelines.

## Overview

Synapse is no longer a chat-based assistant. It is a **research engine** that:
- Searches for real web sources
- Extracts and verifies evidence
- Builds structured claims
- Produces auditable research reports

## Core Change

The system has transitioned from:
- `/ask` → simple LLM response

To:
- `/research` → full research pipeline execution

## Pipeline

Each query now runs through:

1. Query ingestion
2. Web search / retrieval
3. Evidence extraction
4. Claim generation
5. Verification step
6. Final structured report

## Output Format

Each research request returns:

- Summary
- Key points
- Evidence-backed claims
- Source list

## Frontend

A lightweight HTML interface (`index.html`) now:
- Sends requests to `/research`
- Renders structured research outputs
- Displays results as organized sections instead of chat messages

## Key Design Shift

Synapse is designed as:
- A transparent reasoning system
- A source-grounded research tool
- A verifiable alternative to chat-based LLMs

Not:
- A conversational assistant
- A prompt-response chatbot

## Current Limitations

- No real-time streaming (batch responses only)
- Source verification depends on retrieval quality
- UI is minimal and will evolve toward a research graph interface

## Future Direction

- Interactive source graphs
- Evidence-to-claim visual linking
- Real-time pipeline streaming
- Agent-level verification layers

## Status

Early research engine prototype (active development)