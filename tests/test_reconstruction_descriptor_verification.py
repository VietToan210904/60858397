"""Synthetic failure evidence tests descriptor binding, without model execution."""
import json

from stylesphere.common.io import canonical_hash, sha256_file
from stylesphere.reporting.verification import verify


def test_rehashed_failure_descriptor_cannot_replace_original_input_fingerprint(tmp_path):
    output = tmp_path / "results"
    descriptor_path = output / "reconstruction/test-case/input_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    original = {"manifest": {"object_id": "synthetic-unit-case", "source_image": "original.png"},
                "file_sha256": {"source_image": "1" * 64}}
    descriptor_path.write_text(json.dumps(original), encoding="utf-8")
    record = {"pair": "reconstruction", "model": "synthetic-test", "case_id": "synthetic-unit-case", "seed": 1,
              "status": "failed", "mode": "evaluation", "provenance": "model_inference",
              "run_id": "synthetic-test-run", "timestamp_utc": "2026-01-01T00:00:00Z",
              "checkpoint_revision": "a" * 40, "input_sha256": canonical_hash(original), "config_sha256": "test-config",
              "artifacts": {"input_descriptor": descriptor_path.relative_to(tmp_path).as_posix()},
              "artifact_sha256": {"input_descriptor": sha256_file(descriptor_path)}}
    records_path = output / "reconstruction/records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    before = verify(tmp_path, output)
    assert not any("reconstruction input" in error for error in before["errors"])

    replaced = {"manifest": {"object_id": "synthetic-unit-case", "source_image": "substitute.png"},
                "file_sha256": {"source_image": "2" * 64}}
    descriptor_path.write_text(json.dumps(replaced), encoding="utf-8")
    # The artifact-byte checksum is internally consistent; the immutable input
    # identity still describes the original case and must reject this replacement.
    record["artifact_sha256"]["input_descriptor"] = sha256_file(descriptor_path)
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    after = verify(tmp_path, output)
    assert not any("Changed artifact:" in error for error in after["errors"])
    assert "Frozen reconstruction input fingerprint mismatch: reconstruction/synthetic-test/synthetic-unit-case" in after["errors"]
