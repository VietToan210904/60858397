"""Synthetic byte fixtures exercise transport integrity, never model observations."""

import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request

import pytest

from stylesphere.common import ranged_download


PAYLOAD = b"synthetic-bytes-for-transport-unit-test"


class Response:
    def __init__(self, status, content_range, data):
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.data = data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, size):
        result,self.data = self.data[:size],self.data[size:]
        return result


def plan(tmp_path, *, expected_hash=None):
    return {"model_id": "synthetic/unit-fixture", "resolved_revision": "a" * 40, "destination": str(tmp_path / "weights"),
            "files": [{"path": "model.safetensors", "size_bytes": len(PAYLOAD),
                       "expected_sha256": expected_hash or hashlib.sha256(PAYLOAD).hexdigest()}]}


def responder(request, timeout):
    start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
    return Response(206, f"bytes {start}-{end}/{len(PAYLOAD)}", PAYLOAD[start:end + 1])


def test_resumes_verified_ranges_and_preserves_source_prefix(tmp_path, monkeypatch):
    evidence = plan(tmp_path)
    source = tmp_path / "synthetic-prefix.part"
    source.write_bytes(PAYLOAD[:5])
    evidence["files"][0]["source_prefix"] = str(source)
    requests = []
    fail = [True]
    def fetch(request, timeout):
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        requests.append(start)
        if fail[0] and start >= 13:
            return Response(206, "bytes 0-0/1", b"x")
        return responder(request, timeout)
    monkeypatch.setattr(ranged_download, "urlopen", fetch)
    monkeypatch.setattr(ranged_download.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Bounded download failed"):
        ranged_download.download_plan(evidence, workers=1, chunk_bytes=8)
    assert not (tmp_path / "weights/model.safetensors").exists()
    fail[0] = False
    ranged_download.download_plan(evidence, workers=1, chunk_bytes=4)
    assert (tmp_path / "weights/model.safetensors").read_bytes() == PAYLOAD
    assert source.read_bytes() == PAYLOAD[:5]
    assert requests.count(5) == 1 and requests.count(9) == 0  # Original 8-byte range was retained as two 4-byte ranges.


def test_rejects_ignored_range_before_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr(ranged_download, "urlopen", lambda request, timeout: Response(200, "", PAYLOAD))
    monkeypatch.setattr(ranged_download.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Invalid bounded response"):
        ranged_download.download_plan(plan(tmp_path), workers=1, chunk_bytes=4)
    assert not (tmp_path / "weights/model.safetensors").exists()


def test_full_hash_mismatch_never_publishes_weight(tmp_path, monkeypatch):
    monkeypatch.setattr(ranged_download, "urlopen", responder)
    with pytest.raises(ValueError, match="differs from official size/SHA256"):
        ranged_download.download_plan(plan(tmp_path, expected_hash="0" * 64), workers=1, chunk_bytes=8)
    assert not (tmp_path / "weights/model.safetensors").exists()


def test_cross_origin_redirect_strips_credentials_but_keeps_range():
    handler = ranged_download.CredentialBoundRedirectHandler()
    initial = Request("https://huggingface.co/synthetic/repo/resolve/pin/model.safetensors",
                      headers={"Authorization": "Bearer synthetic-unit-fixture", "Cookie": "session=synthetic", "Range": "bytes=0-3"})
    redirected = handler.redirect_request(initial, None, 302, "Found", {}, "https://signed-storage.example/object?signature=synthetic")
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Cookie") is None
    assert redirected.get_header("Range") == "bytes=0-3"
    same_origin = handler.redirect_request(initial, None, 302, "Found", {}, "https://huggingface.co/another-path")
    assert same_origin.get_header("Authorization") == "Bearer synthetic-unit-fixture"
    with pytest.raises(ValueError, match="non-HTTPS"):
        handler.redirect_request(initial, None, 302, "Found", {}, "http://signed-storage.example/object")


def test_transient_journal_lock_retries_locally_without_duplicate_ranges(tmp_path, monkeypatch):
    calls, saved_states = [], []
    original_replace = Path.replace
    failures = [2]
    def fetch(request, timeout):
        calls.append(request.get_header("Range"))
        return responder(request, timeout)
    def replace(source, destination):
        if source.name.endswith(".json.tmp"):
            if failures[0]:
                failures[0] -= 1
                raise PermissionError("synthetic transient file scanner lock")
            saved_states.append(json.loads(source.read_text()))
        return original_replace(source, destination)
    monkeypatch.setattr(ranged_download, "urlopen", fetch)
    monkeypatch.setattr(ranged_download.time, "sleep", lambda _: None)
    monkeypatch.setattr(Path, "replace", replace)
    ranged_download.download_plan(plan(tmp_path), workers=1, chunk_bytes=8)
    assert len(calls) == len(set(calls)) == (len(PAYLOAD) + 7) // 8
    assert saved_states
    assert all(len(s["completed_starts"]) == len(set(s["completed_starts"])) for s in saved_states)
    assert (tmp_path / "weights/model.safetensors").read_bytes() == PAYLOAD
