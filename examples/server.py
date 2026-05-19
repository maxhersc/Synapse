"""
Synapse v0.3 — Research Swarm Flask Backend Server.

Provides a POST /research API endpoint matching the contract expected by the MaxAI UI.
"""

from __future__ import annotations

import asyncio
import os
from flask import Flask, request, jsonify
import json
import traceback

from synapse import Runtime
from protocols.message import Research, NodeStatus

app = Flask(__name__)

@app.route('/')
def index():
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'index.html'), 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return f"index.html not found: {str(e)}", 404

# Manual CORS setup to avoid external dependencies
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/research', methods=['POST'])
def handle_research():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Runs the async synapse pipeline inside synchronous flask request thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_data = loop.run_until_complete(run_pipeline(query))
        loop.close()
        return jsonify(response_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/research', methods=['GET'])
def handle_research_get():
    return "Use POST /research", 200

async def run_pipeline(question: str) -> dict:
    runtime = Runtime()
    await runtime.run()
    
    try:
        # Starts Synapse Research orchestration
        research_obj = await runtime.start_research(question, "")
        
        # Block until the research status transitions out of RUNNING
        while research_obj.status not in (NodeStatus.COMPLETED, NodeStatus.FAILED):
            await asyncio.sleep(0.5)
            
        if research_obj.status == NodeStatus.COMPLETED:
            try:
                synth_data = json.loads(research_obj.final_output)
            except Exception:
                synth_data = {}
                
            summary = synth_data.get("final_summary", "No sufficient verified evidence available.")
            key_points = synth_data.get("key_points", [])
            
            # Extract list of evidence objects
            evidence_list = []
            if research_obj.claims:
                for c in research_obj.claims:
                    status_val = c.status.value if hasattr(c.status, "value") else str(c.status)
                    if "verified" in str(status_val).lower() and c.evidence:
                        for ev in c.evidence:
                            evidence_list.append({
                                "quote": ev.quote,
                                "source": ev.source,
                                "location": ev.location,
                                "confidence": ev.confidence_score
                            })
                            
            return {
                "question": question,
                "summary": summary,
                "key_points": key_points,
                "sources": list(set(research_obj.sources or [])),
                "evidence": evidence_list
            }
        else:
            return {
                "question": question,
                "summary": "Research Pipeline failed to resolve the question.",
                "key_points": [],
                "sources": [],
                "evidence": []
            }
    finally:
        await runtime.stop()

if __name__ == "__main__":
    print("Synapse Research Server running on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001)
