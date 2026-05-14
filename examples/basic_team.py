async def main() -> None:
    # Build a runtime and register a team of local Ollama-backed agents.
    runtime = Runtime().add(PlannerAgent()).add(ResearcherAgent()).add(WriterAgent()).add(ReviewerAgent())

    agent_map = runtime._agent_index

    print("[synapse] Starting team...")
    print("[synapse] What do you want the team to work on?")

    user_input = input("> ").strip()

    if not user_input:
        print("[synapse] No goal provided.")
        return

    print(f"\n[synapse] Goal: {user_input}")

    context = runtime._context()
    context["state"] = user_input

    researcher = agent_map["researcher"]
    writer = agent_map["writer"]
    reviewer = agent_map["reviewer"]

    # =========================
    # STEP 1: RESEARCH
    # =========================
    research_task = Task(
        description=user_input,
        goal_id="local_goal",
        assigned_to="researcher",
    )

    research_result = await researcher.handle_task(research_task, context=context)
    context["research"] = research_result

    # =========================
    # STEP 2: WRITE
    # =========================
    write_task = Task(
        description=str(research_result),
        goal_id="local_goal",
        assigned_to="writer",
    )

    write_result = await writer.handle_task(write_task, context=context)
    context["draft"] = write_result

    # =========================
    # STEP 3: REVIEW
    # =========================
    review_task = Task(
        description=str(write_result),
        goal_id="local_goal",
        assigned_to="reviewer",
    )

    review_result = await reviewer.handle_task(review_task, context=context)
    context["review"] = review_result

    # =========================
    # STEP 4: OPTIONAL REVISION LOOP
    # =========================
    if isinstance(review_result, dict) and review_result.get("modify"):
        revise_task = Task(
            description=str(review_result),
            goal_id="local_goal",
            assigned_to="writer",
        )

        final_result = await writer.handle_task(revise_task, context=context)
    else:
        final_result = write_result

    # =========================
    # OUTPUT
    # =========================
    print("\n[synapse] Final response:\n")
    print(final_result)
