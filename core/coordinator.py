"""
Synapse v0.3 — Coordinator.

LLM-powered brain that:
  1. Takes a Research object (user's question)
  2. Runs a strict 5-stage pipeline:
     - Search (raw retrieval, valid JSON only)
     - Evidence Extraction (quote-level grounding)
     - Claim Generation (based ONLY on VALID_EVIDENCE)
     - Verification (cross-check claims)
     - Synthesis (final output)
"""

from __future__ import annotations

import asyncio
import httpx
import json
from typing import TYPE_CHECKING

from protocols.message import Research, ResearchOperation, NodeStatus, Message, ScopeContract, Evidence, Claim, ClaimStatus
from agents.base import SynapseAgent, AgentProfile

if TYPE_CHECKING:
    from core.bus import MessageBus
    from core.memory import SharedMemory

OLLAMA_URL = "http://localhost:11434/api/generate"
COORDINATOR_MODEL = "gemma3:1b"

class DynamicAgent(SynapseAgent):
    def __init__(self, role_name: str, description: str):
        super().__init__()
        self.profile = AgentProfile(
            name=role_name,
            model="gemma3:4b",
            strengths=["execution"],
            description=description
        )

class Coordinator:
    """Orchestrates the strict evidence-first research pipeline."""

    def __init__(self, bus: "MessageBus", memory: "SharedMemory") -> None:
        self.bus = bus
        self.memory = memory

    async def _llm(self, prompt: str) -> str:
        payload = {"model": COORDINATOR_MODEL, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    def _validate_raw_output(self, parsed: dict) -> None:
        """
        HARD INPUT VALIDATION GATE.
        Rules:
        - Must be a dictionary.
        - Must not contain system error strings, stack traces, or raw HTML.
        """
        if not isinstance(parsed, dict):
            raise ValueError("Output is not a valid JSON dictionary.")

        str_repr = json.dumps(parsed).lower()
        forbidden_strings = ["traceback", "exception", "<html", "<doctype", "internal server error"]
        for bad_str in forbidden_strings:
            if bad_str in str_repr:
                raise ValueError(f"Found forbidden failure/error/HTML string in output: {bad_str}")

    async def _run_agent(self, agent_name: str, desc: str, prompt: str, fallback_data: dict = None) -> dict:
        if fallback_data is None:
            fallback_data = {}
            
        agent = DynamicAgent(agent_name, desc)
        self.bus.register(agent)

        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient=agent.name,
                content=f"Starting stage: {desc}",
                metadata={"system": True},
            )
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await agent.llm(prompt)
                cleaned_result = result.strip()
                if cleaned_result.startswith("```json"):
                    cleaned_result = cleaned_result[7:]
                elif cleaned_result.startswith("```"):
                    cleaned_result = cleaned_result[3:]
                if cleaned_result.endswith("```"):
                    cleaned_result = cleaned_result[:-3]
                    
                parsed = json.loads(cleaned_result.strip())
                self._validate_raw_output(parsed)

                await self.bus.dispatch(
                    Message(
                        sender=agent.name,
                        recipient="coordinator",
                        content=f"Stage completed:\n\n{json.dumps(parsed, indent=2)}",
                        metadata={"system": True},
                    )
                )
                return parsed
            except Exception as e:
                if attempt < max_retries - 1:
                    await self.bus.dispatch(
                        Message(
                            sender="coordinator",
                            recipient="system",
                            content=f"[SYSTEM_LOG] {agent.name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying...",
                            metadata={"system": True},
                        )
                    )
                else:
                    await self.bus.dispatch(
                        Message(
                            sender="coordinator",
                            recipient="system",
                            content=f"[SYSTEM_LOG] {agent.name} permanently failed after {max_retries} attempts: {str(e)}. Returning empty fallback.",
                            metadata={"system": True},
                        )
                    )
        
        self.bus.unregister(agent.name)
        # FAILURE IS NOT DATA. Return structured empty fallback.
        return fallback_data

    async def _run_search_validation_layer(self, question: str, context: str) -> list[dict]:
        """Search pipeline with strict validation, retry, and fallback provider."""
        search_prompt = (
            f"You are the search_agent. ONLY retrieve raw sources relevant to the question.\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Output JSON EXACTLY matching this format:\n"
            f"{{\n"
            f"  \"results\": [\n"
            f"    {{\"title\": \"...\", \"url\": \"...\", \"snippet\": \"...\"}}\n"
            f"  ]\n"
            f"}}"
        )

        for provider in ["Primary Search Provider", "Secondary Search Provider"]:
            search_res = await self._run_agent("search_agent", f"Raw Source Retrieval ({provider})", search_prompt, fallback_data={"results": []})
            results = search_res.get("results", [])
            
            validated_results = []
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        title = str(r.get("title", "")).strip()
                        url = str(r.get("url", "")).strip()
                        snippet = str(r.get("snippet", "")).strip()
                        
                        if title and url and snippet:
                            snippet_lower = snippet.lower()
                            if "error" not in snippet_lower and "traceback" not in snippet_lower:
                                validated_results.append({
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet
                                })
            
            if validated_results:
                return validated_results
                
            await self.bus.dispatch(
                Message(
                    sender="coordinator",
                    recipient="system",
                    content=f"[SYSTEM_LOG] {provider} returned invalid or empty results. Escalating...",
                    metadata={"system": True},
                )
            )

        # Final fallback - Empty Result Set
        return []

    async def execute(self, research: Research, conversation_context: str = "") -> str:
        research.status = NodeStatus.RUNNING

        await self.bus.dispatch(
            Message(
                sender="coordinator",
                recipient="system",
                content=f"Starting strict research pipeline for: {research.question}",
                metadata={"system": True},
            )
        )

        ctx = conversation_context

        # STAGE 1: Search (Strict Validation)
        validated_search_results = await self._run_search_validation_layer(research.question, ctx)
        research.sources = [r["url"] for r in validated_search_results]

        # STAGE 2: Evidence Extraction (quote-level grounding)
        valid_evidence = []
        if validated_search_results:
            ext_prompt = (
                f"You are the quote_extractor_agent. ONLY extract verbatim evidence from provided sources.\n"
                f"Question: {research.question}\n"
                f"Sources: {json.dumps(validated_search_results)}\n"
                f"Output JSON EXACTLY matching this format:\n"
                f"{{\n"
                f"  \"evidence\": [\n"
                f"    {{\"quote\": \"verbatim text\", \"source\": \"URL\", \"location\": \"paragraph 1\", \"confidence_score\": 0.9}}\n"
                f"  ]\n"
                f"}}"
            )
            ext_res = await self._run_agent("quote_extractor_agent", "Evidence Extraction", ext_prompt, fallback_data={"evidence": []})
            raw_evidence = ext_res.get("evidence", [])
            
            # STRICT EVIDENCE ACCEPTANCE RULE
            if isinstance(raw_evidence, list):
                for e in raw_evidence:
                    if not isinstance(e, dict): continue
                    quote = str(e.get("quote", "")).strip()
                    source = str(e.get("source", "")).strip()
                    location = str(e.get("location", "")).strip()
                    
                    if quote and location and (source.startswith("http") or source.startswith("www")):
                        valid_evidence.append(e)

        # STAGE 3: Claim Generation (based ONLY on VALID_EVIDENCE)
        claims = []
        if valid_evidence:
            claim_prompt = (
                f"You are the claim_generation_agent. ONLY generate claims based STRICTLY on the provided VALID_EVIDENCE.\n"
                f"Embed the evidence directly inside the claim object.\n"
                f"Question: {research.question}\n"
                f"Evidence: {json.dumps(valid_evidence)}\n"
                f"Output JSON EXACTLY matching this format:\n"
                f"{{\n"
                f"  \"claims\": [\n"
                f"    {{\"claim\": \"...\", \"evidence\": [{{\"quote\": \"...\", \"source\": \"...\", \"location\": \"...\", \"confidence_score\": 0.9}}]}}\n"
                f"  ]\n"
                f"}}"
            )
            claim_res = await self._run_agent("claim_generation_agent", "Claim Generation", claim_prompt, fallback_data={"claims": []})
            raw_claims = claim_res.get("claims", [])
            
            if isinstance(raw_claims, list):
                for c in raw_claims:
                    if not isinstance(c, dict) or not c.get("claim"): continue
                    
                    ev_list = []
                    for e in c.get("evidence", []):
                        if not isinstance(e, dict): continue
                        ev_list.append(Evidence(
                            quote=e.get("quote", ""),
                            source=e.get("source", ""),
                            location=e.get("location", ""),
                            retrieved_by="quote_extractor_agent",
                            confidence_score=e.get("confidence_score", 0.5)
                        ))
                    claims.append(Claim(
                        claim=c.get("claim", ""),
                        evidence=ev_list
                    ))

        # STAGE 4: Verification (MANDATORY)
        verified_claims_list = []
        if claims:
            claims_to_check = []
            for c in claims:
                claims_to_check.append({
                    "claim": c.claim,
                    "evidence": [{"quote": e.quote, "source": e.source} for e in c.evidence]
                })
                
            verify_prompt = (
                f"You are the fact_check_agent. ONLY verify claims.\n"
                f"For each claim, assign a status: 'verified', 'disputed', 'weak', or 'unsupported'.\n"
                f"If evidence is missing or weak, status must be 'unsupported' or 'weak'.\n"
                f"Claims: {json.dumps(claims_to_check)}\n"
                f"Output JSON with a 'verified_claims' array.\n"
                f"{{\n"
                f"  \"verified_claims\": [\n"
                f"    {{\"claim\": \"...\", \"status\": \"verified\", \"supporting_sources\": [], \"contradicting_sources\": [], \"confidence_score\": 0.95}}\n"
                f"  ]\n"
                f"}}"
            )
            verify_res = await self._run_agent("fact_check_agent", "Mandatory Verification", verify_prompt, fallback_data={"verified_claims": []})
            verified_data = verify_res.get("verified_claims", [])
            
            if isinstance(verified_data, list):
                for i, c in enumerate(claims):
                    if not c.evidence:
                        c.status = ClaimStatus.UNSUPPORTED
                        c.confidence_score = 0.0
                        continue
                        
                    if i < len(verified_data) and isinstance(verified_data[i], dict):
                        vd = verified_data[i]
                        status_str = str(vd.get("status", "unsupported")).lower()
                        if status_str == "verified": c.status = ClaimStatus.VERIFIED
                        elif status_str == "disputed": c.status = ClaimStatus.DISPUTED
                        elif status_str == "weak": c.status = ClaimStatus.WEAK
                        else: c.status = ClaimStatus.UNSUPPORTED
                        
                        c.supporting_sources = vd.get("supporting_sources", [])
                        c.contradicting_sources = vd.get("contradicting_sources", [])
                        c.confidence_score = vd.get("confidence_score", 0.0)
                    else:
                        c.status = ClaimStatus.UNSUPPORTED
                        c.confidence_score = 0.0
                    
                    if c.status == ClaimStatus.VERIFIED:
                        verified_claims_list.append(c)

        research.claims = claims
        
        # STAGE 5: Synthesis
        if not verified_claims_list:
            # If no valid evidence/claims, DO NOT fabricate claims.
            synth_fallback = {
                "research_question": research.question,
                "final_summary": "No sufficient verified evidence available.",
                "key_points": [],
                "claims_used": []
            }
            final_output_str = json.dumps(synth_fallback, indent=2)
            
            await self.bus.dispatch(
                Message(
                    sender="coordinator",
                    recipient="system",
                    content="No verified claims available. Emitting insufficient evidence response.",
                    metadata={"system": True},
                )
            )
        else:
            synth_input = {
                "research_question": research.question,
                "verified_claims": [
                    {
                        "claim": c.claim,
                        "evidence": [{"quote": e.quote, "source": e.source, "location": e.location} for e in c.evidence]
                    }
                    for c in verified_claims_list
                ]
            }
            
            synth_prompt = (
                f"You are the synthesis_agent. ONLY build the final structured output from the provided verified claims.\n"
                f"You MUST format the output EXACTLY matching this JSON schema:\n"
                f"{{\n"
                f"  \"research_question\": \"...\",\n"
                f"  \"final_summary\": \"...\",\n"
                f"  \"key_points\": [\"...\"],\n"
                f"  \"claims_used\": [\"...\"]\n"
                f"}}\n"
                f"RULES:\n"
                f"1. Output ONLY valid JSON. No markdown, no text outside JSON.\n"
                f"2. Do not include explanations or debug text.\n"
                f"Input Data: {json.dumps(synth_input)}\n"
            )
            synth_fallback = {
                "research_question": research.question,
                "final_summary": "No sufficient verified evidence available.",
                "key_points": [],
                "claims_used": []
            }
            synth_res = await self._run_agent("synthesis_agent", "Final Synthesis", synth_prompt, fallback_data=synth_fallback)
            
            # Validate output matches schema loosely
            if not synth_res or not isinstance(synth_res, dict) or "final_summary" not in synth_res:
                synth_res = synth_fallback
                
            final_output_str = json.dumps(synth_res, indent=2)

        research.final_output = final_output_str
        await self.memory.set(f"research_{research.id}_result", final_output_str)
        
        research.status = NodeStatus.COMPLETED

        return final_output_str
