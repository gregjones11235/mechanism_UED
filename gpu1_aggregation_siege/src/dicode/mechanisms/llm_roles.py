"""LLM role definitions for curriculum aggregation.

Each role provides a specific judgment perspective on candidate tasks:
  - Tutor (Qwen): progression, learnability, tech tree progress
  - Critic (DeepSeek): failure risk, too-hard/already-mastered detection
  - Explorer (GLM): novelty, diversity, skill coverage
"""

import json
import re
from typing import Optional

from dicode.mechanisms.llm_providers import (
    ROLE_PROVIDER_MAP,
    call_llm_api,
    get_provider_config,
)


ROLE_DEFINITIONS = {
    "tutor": {
        "provider": "qwen",
        "description": "Progression Judge — evaluates learnability and curriculum progress",
        "score_keys": ["progression_score", "learnability_score", "tech_tree_progress_score"],
        "flag_keys": ["too_easy", "too_hard"],
    },
    "critic": {
        "provider": "deepseek",
        "description": "Failure-Risk Judge — identifies too-hard, already-mastered, or invalid tasks",
        "score_keys": ["critic_penalty"],
        "flag_keys": ["too_hard", "already_mastered", "invalid_risk", "metric_hacking_risk"],
    },
    "explorer": {
        "provider": "glm",
        "description": "Novelty-Diversity Judge — evaluates novelty and skill diversity",
        "score_keys": ["novelty_score", "diversity_score"],
        "flag_keys": [],
    },
}


def build_role_prompt(role: str, task_summary: dict) -> str:
    """Build a strict JSON-only prompt for a specific role evaluating a task.

    Args:
        role: One of 'tutor', 'critic', 'explorer'.
        task_summary: Dict with task info (task_id, description, skills, etc.).

    Returns:
        Prompt string for the LLM.
    """
    task_id = task_summary.get("task_id", "unknown")
    description = task_summary.get("description", task_summary.get("summary", "No description"))
    source = task_summary.get("source", task_summary.get("status", "unknown"))
    skills = task_summary.get("skills", task_summary.get("skill_tag", "unknown"))
    recent_sr = task_summary.get("recent_success", task_summary.get("sr", "unknown"))
    best_sr = task_summary.get("best_success", task_summary.get("best_sr", "unknown"))

    base_info = (
        f"Task ID: {task_id}\n"
        f"Source: {source}\n"
        f"Description: {description}\n"
        f"Skills involved: {skills}\n"
        f"Recent success rate: {recent_sr}\n"
        f"Best success rate: {best_sr}\n"
    )

    if role == "tutor":
        prompt = (
            f"You are a curriculum progression judge for a reinforcement learning agent.\n"
            f"{base_info}\n"
            f"Evaluate whether this task is appropriate for the agent's current learning stage.\n"
            f"Return STRICT JSON only, no explanation, no markdown:\n"
            f'{{"task_id":"{task_id}","role":"tutor","provider":"qwen","model":"qwen-turbo",'
            f'"scores":{{"progression_score":0.XX,"learnability_score":0.XX,"tech_tree_progress_score":0.XX}},'
            f'"flags":{{"too_easy":false,"too_hard":false}},'
            f'"skill_tag":"...","decision":"accept|hold|reject","short_reason":"..."}}'
        )
    elif role == "critic":
        prompt = (
            f"You are a failure-risk judge for a reinforcement learning agent.\n"
            f"{base_info}\n"
            f"Evaluate risks: is this task too hard, already mastered, invalid, or prone to metric hacking?\n"
            f"Return STRICT JSON only, no explanation, no markdown:\n"
            f'{{"task_id":"{task_id}","role":"critic","provider":"deepseek","model":"deepseek-chat",'
            f'"scores":{{"critic_penalty":0.XX}},'
            f'"flags":{{"too_hard":false,"already_mastered":false,"invalid_risk":false,"metric_hacking_risk":false}},'
            f'"skill_tag":"...","decision":"accept|hold|reject","short_reason":"..."}}'
        )
    elif role == "explorer":
        prompt = (
            f"You are a novelty and diversity judge for a reinforcement learning agent's curriculum.\n"
            f"{base_info}\n"
            f"Evaluate: how novel and diverse is this task compared to typical training tasks?\n"
            f"Return STRICT JSON only, no explanation, no markdown:\n"
            f'{{"task_id":"{task_id}","role":"explorer","provider":"glm","model":"glm-4-flash",'
            f'"scores":{{"novelty_score":0.XX,"diversity_score":0.XX}},'
            f'"flags":{{}},'
            f'"skill_tag":"...","decision":"accept|hold|reject","short_reason":"..."}}'
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return prompt


def parse_role_response(content: str, task_id: str, role: str) -> Optional[dict]:
    """Parse LLM response into the standard role judgment schema.

    Attempts to extract JSON, with one retry for repair.

    Args:
        content: Raw response content from the LLM.
        task_id: Expected task ID.
        role: Expected role name.

    Returns:
        Parsed judgment dict, or None if parsing fails.
    """
    # Try to extract JSON from the response
    json_str = _extract_json(content)

    if json_str is None:
        return None

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt repair: fix common issues
        json_str = _repair_json(json_str)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            return None

    # Validate and normalize
    result.setdefault("task_id", task_id)
    result.setdefault("role", role)
    result.setdefault("provider", ROLE_PROVIDER_MAP.get(role, "unknown"))
    result.setdefault("model", "")
    result.setdefault("scores", {})
    result.setdefault("flags", {})
    result.setdefault("skill_tag", "")
    result.setdefault("decision", "hold")
    result.setdefault("short_reason", "")

    # Clamp scores to [0, 1]
    for key in result["scores"]:
        try:
            result["scores"][key] = max(0.0, min(1.0, float(result["scores"][key])))
        except (ValueError, TypeError):
            result["scores"][key] = 0.0

    # Validate decision
    if result["decision"] not in ("accept", "hold", "reject"):
        result["decision"] = "hold"

    return result


def _extract_json(content: str) -> Optional[str]:
    """Extract JSON object from LLM response content."""
    # Try to find JSON object boundaries
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return content[start:end + 1]


def _repair_json(json_str: str) -> str:
    """Attempt basic repairs on malformed JSON."""
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    # Quote unquoted keys
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    return json_str


def call_role_judge(
    role: str,
    task_summary: dict,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
) -> dict:
    """Call an LLM to get a role-specific judgment for a candidate task.

    Args:
        role: 'tutor', 'critic', or 'explorer'.
        task_summary: Dict with task info.
        provider_name: Override the default provider for this role.
        model: Override the default model.
        max_tokens: Max output tokens.

    Returns:
        Dict with judgment results, including 'success', 'error', and the parsed judgment.
    """
    if role not in ROLE_DEFINITIONS:
        return {"success": False, "error": f"Unknown role: {role}"}

    provider = provider_name or ROLE_PROVIDER_MAP.get(role, "qwen")

    prompt = build_role_prompt(role, task_summary)
    messages = [{"role": "user", "content": prompt}]

    response = call_llm_api(
        provider_name=provider,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    if not response["success"]:
        return {
            "success": False,
            "error": response.get("error", "Unknown API error"),
            "provider": provider,
            "role": role,
            "task_id": task_summary.get("task_id", "unknown"),
            "input_tokens_est": response.get("input_tokens_est", 0),
            "output_tokens_est": 0,
            "estimated_cost": response.get("estimated_cost", 0.0),
        }

    # Parse the response
    parsed = parse_role_response(
        response["content"],
        task_summary.get("task_id", "unknown"),
        role,
    )

    if parsed is None:
        return {
            "success": False,
            "error": "Failed to parse LLM response as JSON",
            "provider": provider,
            "role": role,
            "task_id": task_summary.get("task_id", "unknown"),
            "raw_content": response["content"][:500],
            "input_tokens_est": response.get("input_tokens_est", 0),
            "output_tokens_est": response.get("output_tokens_est", 0),
            "estimated_cost": response.get("estimated_cost", 0.0),
        }

    return {
        "success": True,
        "judgment": parsed,
        "provider": provider,
        "role": role,
        "task_id": task_summary.get("task_id", "unknown"),
        "input_tokens_est": response.get("input_tokens_est", 0),
        "output_tokens_est": response.get("output_tokens_est", 0),
        "estimated_cost": response.get("estimated_cost", 0.0),
    }
