import asyncio
import json
import re

import httpx

from synapse import AgentProfile, Goal, Node, Runtime, SynapseAgent


async def call_ollama(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"].strip()
    except httpx.HTTPError as error:
        return f"Ollama request failed: {error}. Make sure Ollama is running locally on http://localhost:11434."


def parse_json_response(response: str) -> dict:
    response = response.strip()

    # Remove markdown code fences if present
    response = re.sub(r"```(json)?", "", response)
    response = response.replace("```", "")

    # Find first JSON object start
    start = response.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", response, 0)

    depth = 0
    for i in range(start, len(response)):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = response[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # second-pass repair for common model issues
                    candidate2 = candidate.replace("'", '"')
                    return json.loads(candidate2)

    raise json.JSONDecodeError("Unclosed JSON object", response, start)


def with_strict_json_suffix(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Return ONLY valid JSON. No extra text.\n"
        "Use double quotes for all keys and strings.\n"
        "Do not include markdown.\n"
        "Do not include trailing commas.\n"
        "Return exactly one JSON object.\n"
    )


async def call_ollama_json(prompt: str) -> dict:
    strict_prompt = with_strict_json_suffix(prompt)
    response = await call_ollama(strict_prompt)
    try:
        return parse_json_response(response)
    except json.JSONDecodeError:
        retry_prompt = (
            f"{strict_prompt}\n"
            "Your previous response was invalid. Try again and output only a single valid JSON object."
        )
        retry_response = await call_ollama(retry_prompt)
        return parse_json_response(retry_response)


def render_final_response(state: dict) -> str:
    items = state.get("items", [])
    notes = state.get("notes", [])
    lines = [
        "Travel API Recommendations",
        "",
        "This final recommendation was assembled from the team’s merged structured decisions.",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.get('name', f'Option {index}')}")
        lines.append(f"   Best for: {item.get('focus', 'General travel workflows')}")
        lines.append(f"   Why it stands out: {item.get('reason', 'Useful for travel product development.')}")
        if item.get("status"):
            lines.append(f"   Status: {item['status']}")

    if notes:
        lines.append("")
        lines.append("Team notes:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


class ResearcherAgent(SynapseAgent):
    profile = AgentProfile(
        name="Researcher",
        model="gemma3:4b",
        strengths=["research", "summarization"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task):
        print("\n[researcher] Working...")
        prompt = (
            "You are a researcher producing mutation deltas for a multi-agent system.\n"
            f"Goal: {task.description}\n\n"
            "Return JSON with this exact shape:\n"
            '{'
            '"add": ['
            '{"id": "string", "name": "string", "focus": "string", "reason": "string", "status": "candidate"}'
            '], '
            '"remove": [], '
            '"modify": [], '
            '"notes": ["string"]'
            '}\n'
            "Rules:\n"
            "- Return ONLY mutation deltas, never a full final state\n"
            "- add must contain exactly 3 candidate items\n"
            "- focus should describe the main use case for the API\n"
            "- reason should be concise and factual\n"
            "- notes should explain why these candidates were introduced\n"
            "\nCRITICAL CONSTRAINT:\n"
            "All items must be real existing iOS travel apps. Do NOT invent apps, features, or APIs.\n"
            "Only use known apps such as PackPoint, TripIt, Google Maps, Airbnb, Hopper, Rome2Rio, Booking.com.\n"
            "If uncertain, prefer widely known mainstream travel apps only.\n"
        )
        result = await call_ollama_json(prompt)
        print(f"[researcher] Done: {json.dumps(result, indent=2)}")
        return result


class WriterAgent(SynapseAgent):
    profile = AgentProfile(
        name="Writer",
        model="gemma3:4b",
        strengths=["writing", "formatting"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task, context):
        state = context["state"]
        print("\n[writer] Received research. Writing...")
        prompt = (
            "You are a writer producing mutation deltas for a multi-agent system.\n"
            f"Goal: {task.description}\n\n"
            f"Current shared state:\n{json.dumps(state)}\n\n"
            "Do NOT write a final answer. Output only changes relative to the current shared state.\n"
            "Return JSON with this exact shape:\n"
            '{'
            '"add": [], '
            '"remove": [], '
            '"modify": ['
            '{"id": "string", "focus": "string", "reason": "string", "status": "drafted"}'
            '], '
            '"notes": ["string"]'
            '}\n'
            "Rules:\n"
            "- Return ONLY mutation deltas, never a full state\n"
            "- Do not repeat unchanged items\n"
            "- modify existing items to improve focus and reasons for presentation\n"
            "- notes must justify why the modifications were necessary\n"
            "\nCRITICAL CONSTRAINT:\n"
            "All items must reference real existing iOS travel apps only. Do NOT invent new apps or abstract features.\n"
            "Allowed entities include PackPoint, TripIt, Google Maps, Airbnb, Hopper, Rome2Rio, Booking.com.\n"
            "Modification must stay within the same real app entity.\n"
        )
        result = await call_ollama_json(prompt)
        print(f"[writer] Done: {json.dumps(result, indent=2)}")
        return result


class ReviewerAgent(SynapseAgent):
    profile = AgentProfile(
        name="Reviewer",
        model="gemma3:1b",
        strengths=["review", "fact-checking"],
        capabilities=["local_llm"],
    )

    async def handle_task(self, task, context):
        state = context["state"]
        print("\n[reviewer] Reviewing...")
        prompt = (
            "You are a reviewer producing mutation deltas for a multi-agent system.\n"
            f"Goal: {task.description}\n\n"
            f"Current shared state:\n{json.dumps(state)}\n\n"
            "Do NOT write a final answer. Output only changes relative to the current shared state.\n"
            "Return JSON with this exact shape:\n"
            '{'
            '"add": [], '
            '"remove": ["string"], '
            '"modify": ['
            '{"id": "string", "reason": "string", "status": "approved"}'
            '], '
            '"notes": ["string"]'
            '}\n'
            "Rules:\n"
            "- Return ONLY mutation deltas, never a full state\n"
            "- Reviewer must make at least one non-trivial remove or modify decision when justified\n"
            "- remove weak, duplicated, or unsupported candidates if needed\n"
            "- modify surviving items to improve accuracy and decision quality\n"
            "- notes must explain every review decision\n"
            "\nCRITICAL CONSTRAINT:\n"
            "Review must only evaluate real existing iOS travel apps. Do NOT introduce or modify into fictional apps or abstract features.\n"
            "Use only known apps such as PackPoint, TripIt, Google Maps, Airbnb, Hopper, Rome2Rio, Booking.com.\n"
        )
        result = await call_ollama_json(prompt)
        print(f"[reviewer] Done: {json.dumps(result, indent=2)}")
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
    final_state = await runtime.run_dag(goal, nodes)
    final_response = render_final_response(final_state)
    print("\n[synapse] Goal complete.")
    print("\n[synapse] Final combined response:")
    print(final_response)


if __name__ == "__main__":
    asyncio.run(main())
