import asyncio

from synapse import AgentProfile, Goal, Runtime, SynapseAgent


class ResearcherAgent(SynapseAgent):
    profile = AgentProfile("Researcher", "demo-model", ["research", "web search"], ["web_search"])
    async def handle_task(self, task):
        print(f"[{self.id}] Received: {task.description}"); await asyncio.sleep(0.5)
        result = "Top travel APIs include Amadeus, Skyscanner, and Booking.com partner tools."
        print(f"[{self.id}] Completed: {result}"); return result


class WriterAgent(SynapseAgent):
    profile = AgentProfile("Writer", "demo-model", ["writing", "summarization"], ["document_writing"])
    depends_on = ["researcher"]
    async def handle_task(self, task, context):
        print(f"[{self.id}] Received: {task.description}"); await asyncio.sleep(0.3)
        research = context["results"]["researcher"].output
        result = f"Summary: {research}"
        print(f"[{self.id}] Completed: {result}"); return result


class ReviewerAgent(SynapseAgent):
    profile = AgentProfile("Reviewer", "demo-model", ["review", "fact-checking"], ["quality_review"])
    depends_on = ["writer"]
    async def handle_task(self, task, context):
        print(f"[{self.id}] Received: {task.description}"); await asyncio.sleep(0.2)
        writer_output = context["results"]["writer"].output
        print(f"[{self.id}] Completed: Approved.")
        return f"Approved. Reviewed: {writer_output}"


async def main() -> None:
    # Build the runtime and register three collaborating agents.
    runtime = Runtime().add(ResearcherAgent()).add(WriterAgent()).add(ReviewerAgent())

    # Submit a developer goal and inspect the generated assignments.
    goal = Goal(description="Research and summarize the top travel APIs")
    tasks = await runtime.submit_goal(goal)
    for task in tasks:
        print(f"Assigned task -> {task.assigned_to}: {task.description}")

    # Run the system for 3 seconds so the agents can finish their work.
    loop = asyncio.get_event_loop()
    done = loop.create_future()
    loop.call_later(3, done.set_result, None)
    await runtime.run(until=done)

    # Print the final goal progress summary.
    print("Final progress:", runtime.progress(goal.id))


if __name__ == "__main__":
    asyncio.run(main())
