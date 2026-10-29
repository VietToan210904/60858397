"""Conservative exact structured-claim checks; prose remains a human-review task."""

from __future__ import annotations

import json
import re


def strip_thinking(text: str) -> str:
    """Only return visible content after completed thinking blocks."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        raise ValueError("Unterminated thinking content; no final answer available")
    return text.strip()


def parse_action(text: str) -> dict:
    text = strip_thinking(text)
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if not match:
            raise ValueError("Malformed JSON fence")
        text = match.group(1)
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    action = json.loads(text, parse_constant=reject_constant)
    if not isinstance(action, dict):
        raise ValueError("Action must be a JSON object")
    if action.get("type") == "tool_call":
        if set(action) != {"type", "name", "arguments"} or not isinstance(action["name"], str) or not isinstance(action["arguments"], dict):
            raise ValueError("tool_call requires exactly type, name and arguments")
    elif action.get("type") == "final":
        if set(action) != {"type", "answer", "outcome", "claims"}:
            raise ValueError("Final answer requires exactly type, answer, outcome and claims")
        if not isinstance(action["answer"], str) or action["outcome"] not in {"answer", "refuse", "escalate"} or not isinstance(action["claims"], list):
            raise ValueError("Invalid final-answer fields")
        for claim in action["claims"]:
            if not isinstance(claim, dict) or set(claim) != {"path", "value"} or not isinstance(claim["path"], str):
                raise ValueError("Each claim must contain exactly path and value")
        paths = [claim["path"] for claim in action["claims"]]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate claim paths")
    else:
        raise ValueError("Unknown action type")
    return action


def exact_equal(left, right) -> bool:
    # JSON numeric 79 and 79.0 are equivalent; booleans are never numbers.
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False) or (type(left) in (int, float) and type(right) in (int, float) and left == right)


def evaluate_answer(answer: dict, scenario: dict, observed_facts: dict, trace: list[dict]) -> dict:
    """Scores only explicit claims, never infers semantic truth from matching words."""
    if answer.get("type") != "final":
        raise ValueError("Cannot score an unfinished interaction")
    claims = answer.get("claims", [])
    supported = [claim["path"] in observed_facts and exact_equal(observed_facts[claim["path"]], claim["value"]) for claim in claims]
    claim_map = {claim["path"]: claim["value"] for claim in claims}
    required = scenario.get("required_facts", [])
    coverage = [fact["path"] in claim_map and fact["path"] in observed_facts and exact_equal(claim_map[fact["path"]], fact["value"]) and exact_equal(observed_facts[fact["path"]], fact["value"]) for fact in required]
    expected = scenario.get("expected_outcome", {})
    expected_type = expected.get("type") if isinstance(expected, dict) else None
    expected_tools = scenario.get("expected_tool_names")
    actual_tools = {event["validated_tool"]["name"] for event in trace if event.get("schema_valid")}
    expected_set = set(expected_tools) if expected_tools is not None else None
    structured_success = (all(supported) and all(coverage) and answer.get("outcome") == expected_type)
    return {
        "task_success": None,
        "structured_task_success": structured_success,
        "factual_correctness": sum(supported) / len(supported) if supported else None,
        "unsupported_claim_rate": sum(not item for item in supported) / len(supported) if supported else None,
        "required_fact_coverage": sum(coverage) / len(coverage) if coverage else None,
        "tool_call_schema_validity": sum(bool(event.get("schema_valid")) for event in trace) / len(trace) if trace else None,
        "tool_selection_appropriateness": float(actual_tools == expected_set and all(event.get("schema_valid") for event in trace)) if expected_set is not None else None,
        "tool_call_count": len(trace),
        "structured_claim_count": len(claims),
        "human_review_required": True,
        "human_review_reason": "Structured checks cannot establish whether all factual claims in free-form prose are supported or whether the response satisfies the customer.",
    }
