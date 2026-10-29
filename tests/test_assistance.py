"""Hypothetical fixtures; scripted responses test the controller, never a real model."""

import copy
import json

import pytest

from stylesphere.assistance.runner import initial_messages, interact
from stylesphere.assistance.tools import ToolController, ToolValidationError, validate_knowledge_base
from stylesphere.assistance.validation import evaluate_answer, parse_action, strip_thinking


@pytest.fixture
def kb():
    return {"metadata": {"hypothetical": True}, "products": [{"product_id": "P001", "name": "Red Shoe", "category": "footwear", "description": "red leather shoe",
             "price": 89.0, "currency": "AUD", "sizes": ["38", "39"], "colours": ["red"],
             "inventory": [{"size": "38", "colour": "red", "quantity": 2}, {"size": "39", "colour": "red", "quantity": 0}]}],
            "delivery_policy": {"standard_days": "3-5"}, "return_policy": {"window_days": 30}}


@pytest.fixture
def scenario():
    return {"scenario_id": "TEST001", "customer_request": "What is the price of P001?", "available_business_records": ["P001"],
            "required_facts": [{"path": "products.P001.price", "value": 89.0}], "permitted_tools": ["get_product_details"],
            "expected_outcome": {"type": "answer"}, "expected_tool_names": ["get_product_details"], "evaluation_category": "unit_fixture", "hypothetical": True}


def final(price=89):
    return {"type": "final", "answer": f"It costs AUD {price}.", "outcome": "answer", "claims": [{"path": "products.P001.price", "value": price}]}


def test_tool_arguments_are_strict_and_read_only(kb):
    controller = ToolController(kb, ["get_product_details", "check_inventory", "search_products"])
    with pytest.raises(ToolValidationError, match="authorised"):
        controller.execute("purchase", {"product_id": "P001"})
    with pytest.raises(ToolValidationError, match="Unexpected"):
        controller.execute("get_product_details", {"product_id": "P001", "execute": "anything"})
    with pytest.raises(ToolValidationError, match="Invalid value"):
        controller.execute("search_products", {"query": "red", "limit": True})
    with pytest.raises(ToolValidationError, match="Unknown size"):
        controller.execute("check_inventory", {"product_id": "P001", "size": "99"})
    product = controller.execute("get_product_details", {"product_id": "P001"})
    product["product"]["price"] = 0
    assert kb["products"][0]["price"] == 89
    assert controller.execute("get_product_details", {"product_id": "P001"})["product"]["price"] == 89


def test_visibility_restricts_products_and_policies(kb):
    controller = ToolController(kb, ["get_product_details", "get_delivery_policy"], [])
    with pytest.raises(ToolValidationError, match="unavailable"):
        controller.execute("get_product_details", {"product_id": "P001"})
    with pytest.raises(ToolValidationError, match="unavailable"):
        controller.execute("get_delivery_policy", {})


def test_inventory_observation_exposes_exact_canonical_path(kb):
    controller = ToolController(kb, ["check_inventory"])
    observation = controller.execute("check_inventory", {"product_id": "P001", "size": "39", "colour": "red"})
    assert observation["inventory"][0]["quantity"] == 0
    assert observation["inventory"][0]["fact_path"] == "products.P001.inventory.1"
    assert controller.observed_facts["products.P001.inventory.1.quantity"] == 0
    assert "products.P001.price" not in controller.observed_facts


def test_grounded_claims_are_exact_and_prose_requires_review(kb, scenario):
    controller = ToolController(kb, scenario["permitted_tools"])
    # A correct-looking fact is unsupported until a permitted observation supplied it.
    unobserved = evaluate_answer(final(), scenario, {}, [])
    assert unobserved["unsupported_claim_rate"] == 1
    assert unobserved["structured_task_success"] is False
    controller.execute("get_product_details", {"product_id": "P001"})
    result = evaluate_answer(final(), scenario, controller.observed_facts, controller.trace)
    assert result["factual_correctness"] == 1
    assert result["required_fact_coverage"] == 1
    assert result["structured_task_success"] is True
    assert result["task_success"] is None and result["human_review_required"] is True
    assert evaluate_answer(final(90), scenario, controller.observed_facts, controller.trace)["factual_correctness"] == 0


def test_thinking_is_excluded_and_json_schema_rejects_extra_actions():
    assert parse_action('<think>private deliberation</think>\n' + json.dumps(final())) == final()
    with pytest.raises(ValueError, match="Unterminated"):
        strip_thinking("<think>unfinished")
    with pytest.raises(ValueError):
        parse_action('{"type":"tool_call","name":"purchase","arguments":{},"shell":"run"}')
    with pytest.raises(ValueError):
        parse_action('{"type":"tool_call","name":"search_products","arguments":{"max_price":NaN}}')


class ScriptedTestModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.last_timings = {}
    def generate(self, messages):
        return json.dumps(next(self.actions))


def test_shared_controller_records_complete_tool_sequence(kb, scenario):
    model = ScriptedTestModel([{"type": "tool_call", "name": "get_product_details", "arguments": {"product_id": "P001"}}, final()])
    result, trace = interact(model, scenario, kb)
    assert result == final()
    assert len(trace["turns"]) == 2
    assert trace["tool_trace"][0]["schema_valid"] is True
    assert trace["metrics"]["structured_task_success"] is True
    prompt = initial_messages(scenario, ToolController(kb, scenario["permitted_tools"], ["P001"]))
    assert "89.0" not in json.dumps(prompt)  # The gold answer was not leaked into the prompt.
    assert trace["controller_protocol_version"] == 2
    assert trace["turns"][0]["observation"]["observed_facts"]["products.P001.price"] == 89
    assert "delivery_policy.standard_days" not in trace["turns"][0]["observation"]["observed_facts"]


def test_step_limit_and_unauthorised_tool_are_logged(kb, scenario):
    model = ScriptedTestModel([{"type": "tool_call", "name": "purchase", "arguments": {"product_id": "P001"}}])
    result, trace = interact(model, scenario, kb, max_steps=1)
    assert result is None
    assert trace["tool_trace"][0]["schema_valid"] is False
    assert trace["error"]["type"] == "StepLimitExceeded"
    assert trace["turns"][0]["observation"]["action_executed"] is False


def test_knowledge_base_rejects_invalid_stock_and_duplicate_products(kb):
    damaged = copy.deepcopy(kb)
    damaged["products"][0]["inventory"][0]["quantity"] = -1
    with pytest.raises(ValueError, match="quantity"):
        validate_knowledge_base(damaged)
    damaged = copy.deepcopy(kb)
    damaged["products"].append(copy.deepcopy(damaged["products"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_knowledge_base(damaged)


def test_trace_verification_replays_tools_and_detects_tampering(kb, scenario):
    from stylesphere.assistance.verification import verify_trace
    model = ScriptedTestModel([{"type": "tool_call", "name": "get_product_details", "arguments": {"product_id": "P001"}}, final()])
    _, trace = interact(model, scenario, kb)
    assert verify_trace(trace, scenario, kb) == []
    modified = copy.deepcopy(trace)
    modified["observed_facts"]["products.P001.price"] = 1
    assert any("observed facts" in error for error in verify_trace(modified, scenario, kb))
    modified = copy.deepcopy(trace)
    modified["metrics"]["factual_correctness"] = 0.5
    assert any("recomputation" in error for error in verify_trace(modified, scenario, kb))
    modified = copy.deepcopy(trace)
    modified["turns"][0]["observation"]["product"]["price"] = 1
    assert any("frozen knowledge base" in error for error in verify_trace(modified, scenario, kb))
    modified = copy.deepcopy(trace)
    modified["turns"][0]["visible_model_output"] = json.dumps({"type": "tool_call", "name": "purchase", "arguments": {"product_id": "P001"}})
    assert verify_trace(modified, scenario, kb)
    modified = copy.deepcopy(trace)
    modified["turns"][0]["observation"]["observed_facts"]["products.P001.price"] = 1
    assert any("frozen knowledge base" in error for error in verify_trace(modified, scenario, kb))


def test_legacy_trace_replay_preserves_absent_protocol_version(kb, scenario):
    from stylesphere.assistance.verification import verify_trace
    actions = [{"type": "tool_call", "name": "get_product_details", "arguments": {"product_id": "P001"}}, final()]
    _, trace = interact(ScriptedTestModel(actions), scenario, kb, protocol_version=1)
    assert "observed_facts" not in trace["turns"][0]["observation"]
    trace.pop("controller_protocol_version")  # Traces recorded before v2 have no field.
    assert verify_trace(trace, scenario, kb) == []


def test_failed_attempts_keep_timings_and_distinct_hashed_traces(tmp_path, monkeypatch, kb, scenario):
    """Synthetic runtime only: retries must preserve evidence of failed output."""
    from stylesphere.assistance import runner
    from stylesphere.common import runtime
    from stylesphere.common.io import sha256_file

    class TestMeasurement:
        elapsed_s = 0.25
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class InvalidOutputTestModel:
        revision = "fixture-revision"
        metadata = {"unit_test_fixture": True}
        last_timings = {"preprocessing_s": 0.01, "inference_s": 0.2, "peak_allocated_mb": 1, "peak_reserved_mb": 2}
        def __init__(self, *args):
            pass
        def generate(self, messages):
            return '{"type":"tool_call"}}'
        def close(self):
            pass

    monkeypatch.setattr(runtime, "Measurement", TestMeasurement)
    monkeypatch.setattr(runtime, "require_cuda", lambda: None)
    monkeypatch.setattr(runner, "ChatModel", InvalidOutputTestModel)
    import torch
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "synthetic-unit-test-device")
    dataset = tmp_path / "datasets" / "assistance"
    dataset.mkdir(parents=True)
    (dataset / "knowledge_base.json").write_text(json.dumps(kb))
    (dataset / "scenarios.jsonl").write_text(json.dumps(scenario) + "\n")
    config = {"models": {key: {"revision": "fixture-revision"} for key in runner.MODEL_IDS},
              "evaluation": {"warmup_runs": 0, "assistance": {"controller_protocol_version": 2}}}
    for _ in range(2):
        result = runner.run(tmp_path, config, tmp_path / "results", 1, smoke=True, model_keys=["phi4_mini"])
        assert result["failed"] == 1 and result["success"] == 0
    records = [json.loads(line) for line in (tmp_path / "results/assistance/records.jsonl").read_text().splitlines()]
    assert records[0]["artifacts"]["trace"] != records[1]["artifacts"]["trace"]
    for record in records:
        assert record["timings"]["inference_s"] == 0.2
        assert record["timings"]["end_to_end_s"] == 0.25
        assert "metrics" not in record
        for key, location in record["artifacts"].items():
            assert sha256_file(tmp_path / location) == record["artifact_sha256"][key]
