"""Synthetic snapshots test report provenance, never model observations."""

import csv
import hashlib
import json

import pytest

from stylesphere.common.io import append_jsonl, canonical_hash, sha256_file
from stylesphere.reporting import reports


def test_snapshot_hashes_exact_bom_bytes_and_preserves_list_default(tmp_path):
    source = tmp_path / "retrieval/records.jsonl"
    source.parent.mkdir()
    record = {"case_id": "synthetic-café"}
    data = b"\xef\xbb\xbf" + (json.dumps(record, ensure_ascii=False) + "\r\n\n").encode("utf-8")
    source.write_bytes(data)
    assert reports.load_records(tmp_path) == [record]
    records, hashes = reports.load_records(tmp_path, with_source_hashes=True)
    assert records == [record]
    assert hashes == {source: hashlib.sha256(data).hexdigest()}


def test_snapshot_preserves_malformed_json_line_diagnostic(tmp_path):
    source = tmp_path / "retrieval/records.jsonl"
    source.parent.mkdir()
    source.write_bytes(b'{}\n{malformed}\n')
    with pytest.raises(ValueError, match="Invalid JSON at records.jsonl:2"):
        reports.load_records(tmp_path, with_source_hashes=True)


def test_append_during_summary_keeps_manifest_bound_to_processed_snapshot(tmp_path, monkeypatch):
    output = tmp_path / "results"
    source = output / "retrieval/records.jsonl"
    record = {"pair": "retrieval", "model": "synthetic-fixture", "case_id": "case1", "seed": 1,
              "mode": "smoke", "status": "blocked", "provenance": "model_inference",
              "config_sha256": "fixture-config", "protocol_sha256": "fixture-protocol",
              "input_sha256": "fixture-input", "error": {"type": "SyntheticFixture"}}
    append_jsonl(source, record)
    captured_sha = sha256_file(source)
    original_summary = reports.summary
    appended = False

    def append_then_summarize(values):
        nonlocal appended
        if not appended:
            appended = True
            append_jsonl(source, {**record, "case_id": "case2"})
        return original_summary(values)

    monkeypatch.setattr(reports, "summary", append_then_summarize)
    result = reports.report(tmp_path, output, {"seed": 1})
    assert appended and result["records"] == 1
    manifest = json.loads((output / "reports/manifest.json").read_text())
    assert manifest["source_record_hashes"] == {"results/retrieval/records.jsonl": captured_sha}
    assert manifest["records_digest"] == canonical_hash([record])
    assert captured_sha != sha256_file(source)
    with (output / "reports/model_summary.csv").open(newline="") as stream:
        assert all(row["blocked_cases"] == "1" for row in csv.DictReader(stream))

    from stylesphere.reporting.verification import verify
    append_jsonl(output / 'generation/records.jsonl', {**record, 'pair':'generation'})
    verification = verify(tmp_path, output)
    assert "Report source changed; regenerate report: results/retrieval/records.jsonl" in verification["errors"]
    assert "Report source-file set changed; regenerate report" in verification["errors"]
