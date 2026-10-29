"""Supplementary review validates authored labels; never regrades prose by keywords."""
import copy
import json

import pytest

from stylesphere.assistance.ai_assessment import extract_prose, materialize_decisions, validate_assessments


def packet(case="assist-001", prose="It costs AUD 79.", status="failed"):
    return {"assessment_id": "assessment-" + case, "case_id": case, "model": "phi4_mini",
            "source_record_sha256": "a" * 64, "trace_sha256": "b" * 64,
            "scenario": {"required_facts": [{"path": "products.P001.price", "value": 79}]},
            "knowledge_base": {"products": [{"product_id": "P001", "price": 79}],
                               "delivery_policy": {}, "return_policy": {}},
            "terminal_visible_output": '{"answer":"' + prose + '"',
            "response": {"text": prose}, "original_status": status}


def decisions(case="001", task="S", facts="S", checklist="E"):
    return f"[phi4_mini]\n{case}|{task}|{facts}|0|0|A|{checklist}|high||An explicitly authored judgement.\n"


def test_readable_answer_scalar_is_retained_without_accepting_outer_action():
    raw = json.dumps({"type": "final", "answer": 'Price is AUD 79; quote: "yes".'})[:-1]
    response = extract_prose(raw)
    assert response["text"] == 'Price is AUD 79; quote: "yes".'
    assert response["method"] == "retained_json_answer_scalar"
    assert raw.endswith('"')


def test_tool_only_output_has_no_customer_prose():
    assert extract_prose('{"type":"tool_call","tool":"get_product_details"}')["text"] is None


def test_explicit_semantic_decision_does_not_repair_protocol_failure():
    source = packet()
    before = copy.deepcopy(source)
    reviews = materialize_decisions([source], decisions())
    assert reviews[0]["ai_semantic_task_verdict"] == "satisfied"
    assert "original_protocol_failure" in reviews[0]["error_categories"]
    assert source == before


def test_every_failure_requires_explicit_authored_assessment():
    with pytest.raises(ValueError, match="Every selected execution"):
        materialize_decisions([packet(), packet("assist-002")], decisions())


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(trace_sha256="tampered"), "binding differs"),
    (lambda r: r["evidence"][0].update(excerpt="Invented quotation"), "exact retained-output"),
    (lambda r: r["evidence"][0].update(kb_paths=["products.P999.price"]), "nonexistent KB"),
    (lambda r: r.update(requested_fact_entailment={}), "checklist differs"),
])
def test_tampered_review_evidence_or_binding_rejected(mutation, match):
    source = packet()
    reviews = materialize_decisions([source], decisions())
    mutation(reviews[0])
    with pytest.raises(ValueError, match=match):
        validate_assessments([source], reviews)


def test_no_prose_cannot_be_assessed_as_delivered_answer():
    source = packet()
    source["response"]["text"] = None
    with pytest.raises(ValueError, match="prose score requires"):
        materialize_decisions([source], decisions())


def test_author_can_mark_truthful_looking_text_wrong_without_keyword_rule():
    source = packet(prose="Correct answer: AUD 79.")
    review = materialize_decisions([source], decisions(task="F", facts="C", checklist="C"))[0]
    # Validation binds the supplied semantic review; it does not infer correctness
    # from the words 'correct', the numeric token, or the structured source value.
    assert review["ai_semantic_task_verdict"] == "not_satisfied"


def test_full_authored_review_covers_all_160_attempts_and_73_failures():
    from pathlib import Path
    from stylesphere.assistance.ai_assessment import read_jsonl
    directory = Path(__file__).resolve().parents[1] / "results/assistance/ai_assessment/v1"
    packets = read_jsonl(directory / "evidence_packets.jsonl")
    reviews = materialize_decisions(packets, (directory / "authored_decisions.txt").read_text(encoding="utf-8"))
    assert len(reviews) == 160
    assert sum(p["original_status"] != "success" for p in packets) == 73
    assert sum("original_protocol_failure" in r["error_categories"] for r in reviews) == 73
    assert sum(r["ai_factual_prose_verdict"] == "no_customer_prose" for r in reviews) == 10
