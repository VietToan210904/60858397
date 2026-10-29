"""Recompute structured assistance evidence from frozen business data and actions."""

from __future__ import annotations

import json

from .tools import ToolController, ToolValidationError
from .validation import evaluate_answer, exact_equal, parse_action


def verify_trace(trace: dict, scenario: dict, kb: dict) -> list[str]:
    """Return integrity errors; tool replay is confined to the read-only controller.

    This verifies trace consistency and exact structured evidence. It makes no
    semantic judgment about free-form prose and cannot independently prove that
    a saved transcript originated from a particular model or GPU.
    """
    from stylesphere.common.io import scrub
    from .runner import initial_messages, observation_message

    errors: list[str] = []
    try:
        if trace.get("scenario_id") != scenario["scenario_id"]:
            errors.append("Trace scenario identifier differs from frozen scenario")
        protocol_version = trace.get("controller_protocol_version", 1)
        if type(protocol_version) is not int or protocol_version not in {1, 2}:
            return errors + ["Unsupported assistance controller protocol version"]
        controller = ToolController(kb, scenario["permitted_tools"], scenario["available_business_records"])
        messages = initial_messages(scenario, controller, protocol_version)
        turns = trace.get("turns")
        if not isinstance(turns, list) or not turns:
            return errors + ["Trace has no model turns"]
        replayed_final = None
        for number, turn in enumerate(turns, 1):
            if not isinstance(turn, dict):
                errors.append(f"Turn {number} is not an object")
                continue
            if turn.get("step") != number:
                errors.append(f"Turn {number} has an inconsistent step index")
            if not exact_equal(turn.get("request_messages"), scrub(messages)):
                errors.append(f"Turn {number} request differs from the replayed conversation")
            if replayed_final is not None:
                errors.append("Trace continues after a final response")
                break
            if "visible_model_output" not in turn:
                if not turn.get("failure"):
                    errors.append(f"Turn {number} has neither visible output nor a recorded failure")
                continue
            try:
                action = parse_action(turn["visible_model_output"])
            except (ValueError, TypeError) as exc:
                if not turn.get("failure"):
                    errors.append(f"Turn {number} contains invalid output without a recorded failure: {exc}")
                continue
            if not exact_equal(action, turn.get("proposed_action")):
                errors.append(f"Turn {number} proposed action differs from visible model output")
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            if action["type"] == "final":
                replayed_final = action
                if "validated_tool_call" in turn:
                    errors.append(f"Turn {number} records a tool execution for a final response")
                continue
            try:
                observation = controller.execute(action["name"], action["arguments"])
            except ToolValidationError as exc:
                observation = {"tool_error": str(exc), "action_executed": False}
            if not exact_equal(turn.get("validated_tool_call"), scrub(controller.trace[-1])):
                errors.append(f"Turn {number} tool validation/observation differs from read-only replay")
            observation = observation_message(observation, controller.observed_facts, protocol_version)
            if not exact_equal(turn.get("observation"), scrub(observation)):
                errors.append(f"Turn {number} observation differs from frozen knowledge base")
            messages.append({"role": "user", "content": "READ_ONLY_TOOL_OBSERVATION " + json.dumps(observation, ensure_ascii=False)})
        if not exact_equal(trace.get("tool_trace"), scrub(controller.trace)):
            errors.append("Stored tool trace differs from controller replay")
        if not exact_equal(trace.get("observed_facts"), scrub(controller.observed_facts)):
            errors.append("Stored observed facts differ from authorised frozen-record observations")
        if not exact_equal(trace.get("final_response"), replayed_final):
            errors.append("Stored final response differs from visible final model action")
        if replayed_final is not None:
            expected = evaluate_answer(replayed_final, scenario, controller.observed_facts, controller.trace)
            if not exact_equal(trace.get("metrics"), expected):
                errors.append("Stored assistance metrics differ from deterministic recomputation")
            if trace.get("error") is not None:
                errors.append("Completed trace unexpectedly contains an execution error")
        elif "metrics" in trace:
            errors.append("Unfinished interaction has final-answer metrics")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"Malformed assistance trace or frozen scenario: {type(exc).__name__}: {exc}")
    return errors
