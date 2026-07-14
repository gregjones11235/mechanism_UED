#!/usr/bin/env python3
"""Stage L2: Export real candidate task summaries for LLM judgment.

Reads the task archive graph and exports compact summaries to JSONL.
Does NOT call LLM APIs.
"""

import json
import os
import sys

import networkx as nx

OUTPUT_PATH = "mechanism_logs/pending_llm_tasks.jsonl"


def export_tasks(graphml_path="task_graph.graphml", output_path=OUTPUT_PATH, max_tasks=20):
    """Export compact candidate task summaries from the archive graph."""
    if not os.path.exists(graphml_path):
        print(f"Graph file not found: {graphml_path}")
        print("Run a DiCode smoke first to generate the task graph.")
        return []

    g = nx.read_graphml(graphml_path)
    print(f"Loaded graph with {g.number_of_nodes()} nodes")

    tasks = []
    for node_id, data in g.nodes(data=True):
        # Extract compact summary
        description = data.get("description", "")
        if not description:
            # Try to extract from code docstring
            code = data.get("code", "")
            if code and '"""' in code:
                try:
                    description = code.split('"""')[1].strip()[:300]
                except Exception:
                    pass

        status = data.get("status", "unknown")
        source_type = data.get("type", status)

        # Extract skills from relevant achievement names in code
        skills = _extract_skills_from_code(data.get("code", ""))

        # Performance metrics
        perf_history = data.get("performance_history", [])
        if isinstance(perf_history, str):
            try:
                perf_history = json.loads(perf_history)
            except Exception:
                perf_history = []

        recent_sr = None
        best_sr = None
        if perf_history:
            srs = [h.get("sr", -1) for h in perf_history if isinstance(h, dict) and h.get("sr", -1) >= 0]
            if srs:
                recent_sr = srs[-1]
                best_sr = max(srs)

        priority = float(data.get("priority_score", 0.0))
        learnability = float(data.get("learnability_score", priority))

        summary = {
            "task_id": str(node_id),
            "source": str(source_type),
            "status": str(status),
            "description": str(description)[:500],
            "skills": skills[:5] if skills else "unknown",
            "recent_success": recent_sr,
            "best_success": best_sr,
            "priority_score": priority,
            "learnability_score": learnability,
            "session_created": int(data.get("session_created", 0)) if data.get("session_created") else 0,
        }
        tasks.append(summary)

    # Limit
    tasks = tasks[:max_tasks]

    # Write
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Exported {len(tasks)} tasks to {output_path}")
    for t in tasks:
        print(f"  {t['task_id']}: src={t['source']} sr={t['recent_success']} "
              f"best={t['best_success']} priority={t['priority_score']:.3f} "
              f"skills={t['skills'][:3] if isinstance(t['skills'], list) else t['skills'][:60]}")

    return tasks


def _extract_skills_from_code(code: str) -> list[str]:
    """Extract skill names from task code (achievement references)."""
    if not code:
        return []
    import re
    # Look for Achievement.X patterns
    achievements = re.findall(r'Achievement\.(\w+)', code)
    # Deduplicate and lowercase
    seen = set()
    skills = []
    for a in achievements:
        a_lower = a.lower()
        if a_lower not in seen:
            seen.add(a_lower)
            skills.append(a_lower)
    return skills


if __name__ == "__main__":
    export_tasks()
