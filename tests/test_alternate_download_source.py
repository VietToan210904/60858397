"""Synthetic transport-only checks; no checkpoint or network operations."""
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from stylesphere.common import ranged_download

PAYLOAD = b"SYNTHETIC SOURCE TRANSPORT UNIT TEST"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA}/fixture.pt"


def item(url=URL):
    return {"path": "fixture.pt", "size_bytes": len(PAYLOAD), "expected_sha256": SHA, "source_url": url}


@pytest.mark.parametrize("url", [
    URL.replace("https:", "http:"), URL.replace("openaipublic.azureedge.net", "unapproved.example"),
    URL + "?credential=synthetic", URL + "#fragment", URL.replace(SHA, "0" * 64),
    URL.replace("openaipublic.azureedge.net", "openaipublic.azureedge.net:443"),
])
def test_alternate_source_rejects_wrong_origin_or_identity(url):
    with pytest.raises(ValueError, match="exact approved HTTPS"):
        ranged_download._approved_source_url(item(url))


def test_alternate_source_keeps_default_journal_compatible_and_suppresses_auth(tmp_path, monkeypatch):
    assert ranged_download._approved_source_url({"path": "default.bin"}) is None
    assert ranged_download._approved_source_url(item()) == URL
    requests = []
    class Response:
        status = 206
        def __init__(self, start, end):
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(PAYLOAD)}"}
            self.data = PAYLOAD[start:end + 1]
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size):
            result, self.data = self.data[:size], self.data[size:]
            return result
    def fetch(request, timeout):
        assert request.full_url == URL
        assert request.get_header("Authorization") is None
        requests.append(request)
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        return Response(start, end)
    monkeypatch.setattr(ranged_download, "_cached_token", lambda: "SYNTHETIC TOKEN NEVER SENT")
    monkeypatch.setattr(ranged_download, "build_opener", lambda *args: SimpleNamespace(open=fetch))
    evidence = {"model_id": "synthetic-unit-test-only", "resolved_revision": "a" * 40, "destination": str(tmp_path / "weights"), "files": [item()]}
    ranged_download.download_plan(evidence, workers=1, chunk_bytes=8, authenticated=True)
    assert requests
    assert (tmp_path / "weights/fixture.pt").read_bytes() == PAYLOAD


def test_alternate_origin_is_bound_into_resume_journal(tmp_path, monkeypatch):
    evidence = {"model_id": "synthetic-unit-test-only", "resolved_revision": "a" * 40, "destination": str(tmp_path / "weights"), "files": [item()]}
    monkeypatch.setattr(ranged_download, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("synthetic connection failure")))
    monkeypatch.setattr(ranged_download.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        ranged_download.download_plan(evidence, workers=1, chunk_bytes=8)
    journal = next((tmp_path / "weights/.cache/stylesphere-ranges").glob("*.json"))
    retained = json.loads(journal.read_text())
    assert retained["source_url"] == URL
    retained["source_url"] = "SYNTHETIC DIFFERENT SOURCE"
    journal.write_text(json.dumps(retained))
    with pytest.raises(ValueError, match="journal differs"):
        ranged_download.download_plan(evidence, workers=1, chunk_bytes=8)
    retained["source_url"] = URL
    journal.write_text(json.dumps(retained))
    evidence["files"][0].pop("source_url")
    with pytest.raises(ValueError, match="journal differs"):
        ranged_download.download_plan(evidence, workers=1, chunk_bytes=8)
