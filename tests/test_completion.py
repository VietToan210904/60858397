"""Synthetic gating regressions; no model execution or fabricated study scores."""
import copy
import json
from pathlib import Path

import pytest

from stylesphere.reporting import completion


def synthetic_complete(monkeypatch, root):
    source = root / "src/stylesphere/reporting/completion.py"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic source fixture", encoding="utf-8")
    monkeypatch.setattr(completion, "__file__", str(source))
    def first(e, counts):
        counts.update(completion.REQUIRED_COUNTS)
        return []
    monkeypatch.setattr(completion, "_visual", first)
    monkeypatch.setattr(completion, "_assistance", lambda e, counts: [])
    monkeypatch.setattr(completion, "_retrieval", lambda e, counts: [])
    monkeypatch.setattr(completion, "_technical", lambda e, counts, selected: None)
    monkeypatch.setattr(completion, "_human_separation", lambda e, counts: counts.update(human_rating_values_created=0))
    monkeypatch.setattr(completion, "_reference", lambda e, counts: None)
    return source


def test_scope_completion_is_separate_from_legacy_human_primary(monkeypatch, tmp_path):
    synthetic_complete(monkeypatch, tmp_path)
    receipt = completion.finalize_completion(tmp_path)
    assert receipt["requested_component_complete"] is True
    assert receipt["legacy_benchmark_complete"] is False
    assert receipt["counts"]["technical_protocol_failures"] == 73
    assert completion.validate_completion_receipt(tmp_path) == []


def test_pending_reference_audit_never_becomes_complete(monkeypatch, tmp_path):
    synthetic_complete(monkeypatch, tmp_path)
    def pending(e, counts):
        raise FileNotFoundError("Independent reference audit is pending")
    monkeypatch.setattr(completion, "_reference", pending)
    receipt = completion.finalize_completion(tmp_path)
    assert receipt["requested_component_complete"] is False
    assert any("pending" in error for error in receipt["errors"])
    assert completion.validate_completion_receipt(tmp_path)


def test_missing_assistance_failures_reject_completion(monkeypatch, tmp_path):
    synthetic_complete(monkeypatch, tmp_path)
    def incomplete(e, counts):
        counts["assistance_assessments"] = 87
        counts["assistance_failed_attempts_assessed"] = 0
        return []
    monkeypatch.setattr(completion, "_assistance", incomplete)
    receipt = completion.inspect_completion(tmp_path)
    assert receipt["requested_component_complete"] is False
    assert any("assistance_assessments" in error for error in receipt["errors"])


def test_source_append_after_completion_is_detected(monkeypatch, tmp_path):
    source = synthetic_complete(monkeypatch, tmp_path)
    completion.finalize_completion(tmp_path)
    source.write_text(source.read_text() + "\nchanged", encoding="utf-8")
    errors = completion.validate_completion_receipt(tmp_path)
    assert any("SHA256 differs" in error for error in errors)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(requested_component_complete=False),
    lambda r: r.update(legacy_benchmark_complete=True),
    lambda r: r["counts"].update(reference_test_scored=39),
    lambda r: r.update(checks=[{"name": "normalized_reference_extension", "passed": True}]),
])
def test_receipt_booleans_counts_and_check_registry_cannot_be_forged(monkeypatch, tmp_path, mutate):
    synthetic_complete(monkeypatch, tmp_path)
    receipt = completion.finalize_completion(tmp_path)
    mutate(receipt)
    (tmp_path / completion.RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    assert completion.validate_completion_receipt(tmp_path)


def test_fresh_validation_rejects_forged_success_even_when_hashes_match(monkeypatch, tmp_path):
    synthetic_complete(monkeypatch, tmp_path)
    completion.finalize_completion(tmp_path)
    def missing(e, counts):
        raise ValueError("A current annotated case was removed")
    monkeypatch.setattr(completion, "_retrieval", missing)
    assert any("removed" in error for error in completion.validate_completion_receipt(tmp_path))


def test_evidence_paths_cannot_escape_workspace(tmp_path):
    evidence = completion.Evidence(tmp_path)
    with pytest.raises(ValueError, match="inside the project"):
        evidence.path(tmp_path.parent / "outside.json")


def test_byte_snapshot_detects_change_during_validation(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"case":1}\n', encoding="utf-8")
    evidence = completion.Evidence(tmp_path)
    assert evidence.lines("records.jsonl") == [{"case": 1}]
    path.write_text('{"case":1}\n{"case":2}\n', encoding="utf-8")
    assert evidence.stable()


def test_missing_receipt_is_rejected(tmp_path):
    assert completion.validate_completion_receipt(tmp_path)


def test_nested_extraction_keeps_current_absolute_paths_and_rebases_legacy_paths(tmp_path):
    original = tmp_path / "original-project"
    extracted = original / ".cache/package-validation/run/StyleSphere_Evaluation"
    source = extracted / "src/stylesphere/reporting/completion.py"
    source.parent.mkdir(parents=True)
    source.write_text("copied source", encoding="utf-8")
    legacy = original / "src/stylesphere/reporting/completion.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("different original source", encoding="utf-8")
    origins = extracted / "environment/path_origins.json"
    origins.parent.mkdir()
    origins.write_text(json.dumps({"original_project_roots": [str(original)]}), encoding="utf-8")
    evidence = completion.Evidence(extracted)
    # The extracted absolute path also starts with the original alias. It must
    # resolve directly, while a retained legacy absolute path must still rebase.
    assert evidence.path(source) == source.resolve()
    assert evidence.data(source) == "copied source"
    assert evidence.path(legacy) == source.resolve()
    assert evidence.bind(legacy) == source.resolve()
    assert evidence.hashes == {"src/stylesphere/reporting/completion.py": completion.sha256_file(source)}
