"""Synthetic transport fixtures only; no unit test makes network requests."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("acquire_retrieval_smoke", Path(__file__).resolve().parents[1] / "scripts/acquire_retrieval_smoke.py")
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)
REVISION = "a" * 40
PAYLOAD = b"SYNTHETIC UNIT TEST IMAGE BYTES; NOT BENCHMARK DATA"


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.headers = {}
        self.closed = False

    def iter_content(self, size):
        yield self.payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.fixture
def retained(tmp_path):
    out = tmp_path / "datasets/retrieval_smoke"
    out.mkdir(parents=True)
    card, licence = b"SYNTHETIC DATA CARD", b"SYNTHETIC LICENCE FIXTURE"
    (out / "SOURCE_DATASET_CARD.md").write_bytes(card)
    (out / "LICENCE.txt").write_bytes(licence)
    manifest = {"source_revision": REVISION, "source_card_sha256": acquisition.digest(card),
                "licence_record": {"filename": "LICENCE.txt", "sha256": acquisition.digest(licence)},
                "assets": [{"product_id": "p-synthetic-test", "source_id": "test-0", "row_index": 7,
                            "image_sha256": acquisition.digest(PAYLOAD), "bytes": len(PAYLOAD)}]}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=3), encoding="utf-8")
    return tmp_path, out, path.read_bytes()


def row_payload(revision=REVISION):
    url = f"https://datasets-server.huggingface.co/cached-assets/Marqo/fashion200k/--/{revision}/--/default/data/7/image/image.jpg?Signature=UNIT_TEST_ONLY"
    return json.dumps({"rows": [{"row_idx": 7, "row": {"item_ID": "test-0", "image": {"src": url}}}]}).encode()


def test_restore_exact_match_preserves_inputs_and_good_images(retained, monkeypatch):
    root, out, original_manifest = retained
    session = FakeSession([FakeResponse(row_payload()), FakeResponse(PAYLOAD)])
    monkeypatch.setattr(acquisition.requests, "Session", lambda: session)
    result = acquisition.restore_missing(root)
    assert result["restored_images"] == 1
    assert result["downloaded_bytes_this_invocation"] == len(row_payload()) + len(PAYLOAD)
    assert (out / "images/p-synthetic-test.jpg").read_bytes() == PAYLOAD
    assert (out / "manifest.json").read_bytes() == original_manifest
    assert session.calls[0][1]["params"]["revision"] == REVISION
    timestamp = (out / "images/p-synthetic-test.jpg").stat().st_mtime_ns
    second = acquisition.restore_missing(root, session=FakeSession([]))
    assert second["status"] == "verified_existing"
    assert second["downloaded_bytes_this_invocation"] == 0
    assert (out / "images/p-synthetic-test.jpg").stat().st_mtime_ns == timestamp


def test_restore_hash_mismatch_writes_no_image_or_manifest(retained):
    root, out, original_manifest = retained
    session = FakeSession([FakeResponse(row_payload()), FakeResponse(b"WRONG SYNTHETIC BYTES")])
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        acquisition.restore_missing(root, session=session)
    assert not (out / "images/p-synthetic-test.jpg").exists()
    assert (out / "manifest.json").read_bytes() == original_manifest


def test_restore_refuses_new_source_revision(retained):
    root, out, original_manifest = retained
    session = FakeSession([FakeResponse(row_payload("b" * 40))])
    with pytest.raises(RuntimeError, match="revision unavailable") as exc:
        acquisition.restore_missing(root, session=session)
    assert "Signature" not in str(exc.value)
    assert len(session.calls) == 1
    assert (out / "manifest.json").read_bytes() == original_manifest


def test_restore_enforces_transfer_cap_before_writing(retained):
    root, out, original_manifest = retained
    session = FakeSession([FakeResponse(row_payload()), FakeResponse(PAYLOAD)])
    with pytest.raises(RuntimeError, match="hard download limit"):
        acquisition.restore_missing(root, session=session, limit=len(row_payload()) + len(PAYLOAD) - 1)
    assert not (out / "images/p-synthetic-test.jpg").exists()
    assert (out / "manifest.json").read_bytes() == original_manifest


def test_default_existing_mode_remains_strict_about_missing_images(retained, monkeypatch):
    root, out, original_manifest = retained
    session = FakeSession([])
    monkeypatch.setattr(acquisition.requests, "Session", lambda: session)
    monkeypatch.setattr(sys, "argv", ["acquire_retrieval_smoke.py", "--root", str(root)])
    with pytest.raises(FileNotFoundError):
        acquisition.main()
    assert session.calls == []
    assert (out / "manifest.json").read_bytes() == original_manifest


def test_restore_never_overwrites_corrupted_existing_image(retained):
    root, out, original_manifest = retained
    image = out / "images/p-synthetic-test.jpg"
    image.parent.mkdir()
    image.write_bytes(b"CORRUPTED SYNTHETIC INPUT")
    with pytest.raises(RuntimeError, match="no overwrite"):
        acquisition.restore_missing(root, session=FakeSession([]))
    assert image.read_bytes() == b"CORRUPTED SYNTHETIC INPUT"
    assert (out / "manifest.json").read_bytes() == original_manifest


def test_pinned_source_unavailable_does_not_log_signed_url(retained):
    root, _, _ = retained
    session = FakeSession([FakeResponse(row_payload()), FakeResponse(b"", status=403)])
    with pytest.raises(RuntimeError, match="no replacement source") as exc:
        acquisition.restore_missing(root, session=session)
    assert "Signature" not in str(exc.value)
