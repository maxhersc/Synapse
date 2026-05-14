import asyncio

from synapse import Runtime, Goal, Node
from synapse.agents.researcher import ResearcherAgent
from synapse.agents.writer import WriterAgent
from synapse.agents.reviewer import ReviewerAgent


async def main() -> None:
    runtime = (
        Runtime()
        .add(ResearcherAgent())
        .add(WriterAgent())
        .add(ReviewerAgent())
    )

    print("[synapse] Starting team...")
    print("[synapse] What do you want the team to work on?")

    user_input = input("> ").strip()

    if not user_input:
        print("[synapse] No goal provided.")
        return

    goal = Goal(description=user_input)

    print(f"\n[synapse] Goal: {goal.description}")

    nodes = [
        Node(id="research", agent="researcher"),
        Node(id="write", agent="writer", depends_on=["research"]),
        Node(id="review", agent="reviewer", depends_on=["write"]),
    ]

    print(runtime._agent_index)
    results = await runtime.run_dag(goal, nodes)

    print("\n[synapse] Goal complete.")
    print("\n[synapse] Final response:")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
