import asyncio

import httpx

from synapse import AgentProfile, Goal, Node, Runtime, SynapseAgent


async def call_ollama(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:1b",
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"].strip()
    except httpx.HTTPError as error:
        return f"Ollama request failed: {error}. Make sure Ollama is running locally on http://localhost:11434."


class ResearcherAgent(SynapseAgent):
    profile = AgentProfile(
        name="Researcher",
        model="gemma3:1b",
        strengths=["research", "summarization"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task):
        print("\n[researcher] Working...")
        prompt = (
            f"You are a researcher. {task.description}. "
            "List the top 3 results concisely."
        )
        result = await call_ollama(prompt)
        print(f"[researcher] Done: {result}")
        return result


class WriterAgent(SynapseAgent):
    profile = AgentProfile(
        name="Writer",
        model="gemma3:1b",
        strengths=["writing", "formatting"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task, context):
        research = context["results"]["researcher"].output
        print("\n[writer] Received research. Writing...")
        prompt = (
            "You are a writer. Using this research: "
            f"{research} "
            "Write a short clear summary that can be handed to a reviewer for final polishing."
        )
        result = await call_ollama(prompt)
        print(f"[writer] Done: {result}")
        return result


class ReviewerAgent(SynapseAgent):
    profile = AgentProfile(
        name="Reviewer",
        model="gemma3:1",
        strengths=["review", "fact-checking"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task, context):
        research = context["results"]["researcher"].output
        draft = context["results"]["writer"].output
        print("\n[reviewer] Reviewing...")
        prompt = (
            "You are a reviewer. Use the original research and the draft summary below to produce "
            "one final polished response that is accurate, clear, concise, and ready for the user.\n\n"
            f"Research:\n{research}\n\n"
            f"Draft:\n{draft}\n\n"
            "Return the improved final response only."
        )
        result = await call_ollama(prompt)
        print(f"[reviewer] Done: {result}")
        return result


async def main() -> None:
    # Build a runtime and register a team of local Ollama-backed agents.
    runtime = Runtime().add(ResearcherAgent()).add(WriterAgent()).add(ReviewerAgent())

    print("[synapse] Starting team...")
    print("[synapse] What do you want the team to work on?")
    user_input = input("> ").strip()

    if not user_input:
        print("[synapse] No goal provided. Exiting.")
        return

    # Define the shared goal for the team.
    goal = Goal(description=user_input)
    print(f"\n[synapse] Goal: {goal.description}")

    # Build a DAG so research flows into writing, then writing flows into review.
    nodes = [
        Node(id="researcher-task", agent="researcher"),
        Node(id="writer-task", agent="writer", depends_on=["researcher-task"]),
        Node(id="reviewer-task", agent="reviewer", depends_on=["writer-task"]),
    ]

    # Run the DAG and let each agent consume prior outputs through context["results"].
    results = await runtime.run_dag(goal, nodes)
    final_response = results["reviewer"].output
    print("\n[synapse] Goal complete.")
    print("\n[synapse] Final combined response:")
    print(final_response)


if __name__ == "__main__":
    asyncio.run(main())
