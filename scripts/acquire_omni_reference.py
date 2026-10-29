"""Bounded acquisition of two official OmniObject3D reference categories.

No credentials, model inference, executable extraction, or automatic alignment.
Only the normal public large-file confirmation is accepted; abuse/malware and
TLS errors are fatal. A persistent shared byte ledger includes retries.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

REPO_COMMIT = "d1e05fa62759089298c25bcccf83c94d718375c4"
NETWORK_CAP = 3_000_000_000
EXPANSION_CAP = 20_000_000_000
ARCHIVES = [
    {"category": "shoe", "file_id": "1ymHpMULSCdgvwJU6--wLMVvsNczD31X9", "revision": "0BzPUStBk3iXMK3VhYTBWKzV5YjJIZlNCZU1mWHpVQ2JrSyt3PQ", "size_bytes": 1516939144, "modified_time": "2023-10-16T01:51:44.000Z"},
    {"category": "handbag", "file_id": "1jLEH2_VHNSf6x0QZxeEczWoFLfozwuEC", "revision": "0BzPUStBk3iXMNFUyUXBVaE1iU1JxVzEzQVUvRkNQVnVSbjljPQ", "size_bytes": 1121017275, "modified_time": "2023-10-15T20:21:49.000Z"},
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.1 * (attempt + 1))


class PublicForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields = None, {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self.action = a.get("action")
        if tag == "input" and a.get("name"):
            self.fields[a["name"]] = a.get("value", "")


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.value = json.loads(path.read_text()) if path.exists() else {"network_cap_bytes": NETWORK_CAP, "charged_bytes": 0}
        if self.value["network_cap_bytes"] != NETWORK_CAP:
            raise ValueError("Transfer cap ledger differs")

    def charge(self, count: int):
        # Charge BEFORE writing; interrupted writes conservatively remain charged.
        self.value["charged_bytes"] += count
        save(self.path, self.value)
        if self.value["charged_bytes"] > NETWORK_CAP:
            raise ValueError("Combined network/retry budget exhausted")


def public_response(session, item, ledger: Ledger, offset: int):
    url = "https://drive.usercontent.google.com/download"
    params = {"id": item["file_id"], "export": "download"}
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = session.get(url, params=params, headers=headers, stream=True, timeout=(30, 90))
    response.raise_for_status()
    notice = None
    if "text/html" in response.headers.get("Content-Type", ""):
        body = b""
        for block in response.iter_content(8192):
            ledger.charge(len(block))
            body += block
            if len(body) > 64 * 1024:
                raise ValueError("Oversized public download notice")
        response.close()
        text = body.decode("utf-8", errors="replace")
        lower = text.lower()
        if any(word in lower for word in ("infected", "malware", "malicious", "abuse", "violates")):
            raise ValueError("Provider security/abuse warning: download refused")
        if not ("too large" in lower and "can't scan" in lower and "scan" in lower):
            raise ValueError("Unexpected provider page; no authentication or warning bypass")
        form = PublicForm()
        form.feed(text)
        parsed = urlsplit(form.action or "")
        if (parsed.scheme, parsed.hostname, parsed.path) != ("https", "drive.usercontent.google.com", "/download") or parsed.query or parsed.fragment:
            raise ValueError("Unapproved public confirmation origin")
        if form.fields.get("id") != item["file_id"] or form.fields.get("export") != "download":
            raise ValueError("Public form changed archive identity")
        if set(form.fields) - {"id", "export", "confirm", "uuid", "authuser"}:
            raise ValueError("Unexpected public confirmation fields")
        notice = {"type": "large_file_not_scanned", "bytes": len(body), "html_sha256": hashlib.sha256(body).hexdigest(),
                  "authorization": "Normal public large-file confirmation explicitly authorized; no malware/abuse finding accepted."}
        response = session.get(form.action, params=form.fields, headers=headers, stream=True, timeout=(30, 90))
        response.raise_for_status()
    if "text/" in response.headers.get("Content-Type", ""):
        body = b""
        for block in response.iter_content(8192):
            ledger.charge(len(block))
            body += block
            if len(body) > 64 * 1024:
                break
        response.close()
        if b"quota exceeded" in body.lower():
            ledger.value["last_provider_notice"] = {"type": "quota_exceeded", "body_bytes": len(body),
                "body_sha256": hashlib.sha256(body).hexdigest(), "message": "Too many users have viewed or downloaded this file recently; may take up to24hours. No quota bypass attempted."}
            save(ledger.path, ledger.value)
            raise ValueError("Official provider quota exceeded; no bypass attempted")
        raise ValueError("Expected binary archive, received provider text; no bypass")
    if offset:
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server did not honor exact resume range; no redownload")
    elif response.status_code != 200:
        raise ValueError("Unexpected initial download status")
    expected_remaining = item["size_bytes"] - offset
    if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != expected_remaining:
        raise ValueError("Provider archive size changed")
    return response, notice


def download(root: Path, categories=None):
    import requests
    cache = root / ".cache/omni_reference"
    cache.mkdir(parents=True, exist_ok=True)
    folder = root / "datasets/reconstruction_reference"
    receipt_path = folder / "omni_acquisition_attempt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "source": "OmniObject3D", "official_repo_commit": REPO_COMMIT,
        "official_repo": "https://github.com/omniobject3d/OmniObject3D", "licence": "CC-BY-4.0",
        "network_cap_bytes": NETWORK_CAP, "expansion_cap_bytes": EXPANSION_CAP,
        "archives": [], "version_check_status": "before_metadata_checked_after_pending"}
    receipt.setdefault("planned_official_archives", ARCHIVES)
    ledger = Ledger(cache / "network_ledger.json")
    session = requests.Session()
    session.trust_env = False  # no netrc, inherited proxies, or authentication stores
    session.headers.update({"User-Agent": "StyleSphere-reference-acquisition/1.0"})
    free = shutil.disk_usage(root).free
    if free < EXPANSION_CAP + sum(x["size_bytes"] for x in ARCHIVES):
        raise ValueError("Insufficient disk for bounded archives plus extraction cap")
    receipt.setdefault("disk_preflight", {"free_bytes": free, "required_upper_bound_bytes": EXPANSION_CAP + sum(x["size_bytes"] for x in ARCHIVES)})
    save(receipt_path, receipt)
    for item in ARCHIVES:
        if categories and item["category"] not in categories:
            continue
        path = cache / f"{item['category']}.tar.gz"
        existing = next((x for x in receipt["archives"] if x["category"] == item["category"]), None)
        if path.is_file():
            if not existing or path.stat().st_size != item["size_bytes"] or sha(path) != existing.get("sha256"):
                raise ValueError("Existing complete archive lacks matching frozen receipt")
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        notice = None
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["size_bytes"]:
                raise ValueError("Oversized partial archive")
            if offset == item["size_bytes"]:
                break
            if ledger.value["charged_bytes"] + item["size_bytes"] - offset + 65536 > NETWORK_CAP:
                raise ValueError("Remaining pinned payload cannot fit shared retry/network budget")
            try:
                response, notice = public_response(session, item, ledger, offset)
                with response:
                    # Notice digest is preserved separately without public form tokens/URLs.
                    with partial.open("ab") as out:
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            ledger.charge(len(block))
                            if out.tell() + len(block) > item["size_bytes"]:
                                raise ValueError("Archive exceeds pinned size")
                            out.write(block)
                if partial.stat().st_size != item["size_bytes"]:
                    raise requests.ConnectionError("Truncated archive")
                break
            except ValueError as error:
                receipt.setdefault("attempts", []).append({"category": item["category"], "at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "blocked", "archive_binary_bytes": offset, "error": str(error), "provider_notice": ledger.value.get("last_provider_notice")})
                receipt["status"] = "blocked_no_archive_downloaded" if not receipt["archives"] else "partially_acquired"
                save(receipt_path, receipt)
                raise
            except requests.RequestException as error:
                # Never print request exception text: it may include public redirect tokens.
                save(cache / f"{item['category']}-last-transport.json", {"attempt": attempt + 1, "error_type": type(error).__name__, "partial_bytes": partial.stat().st_size if partial.exists() else 0})
                if attempt == 3:
                    raise RuntimeError("Transport failed; partial bytes retained, no URL logged") from None
                time.sleep(attempt + 1)
        with partial.open("rb") as f:
            if f.read(2) != b"\x1f\x8b":
                raise ValueError("Archive is not gzip")
        partial.replace(path)
        record = {**item, "path": path.relative_to(root).as_posix(), "sha256": sha(path),
                  "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                  "large_file_notice": "Normal public too-large-to-scan notice acknowledged; no abuse/malware bypass.",
                  "large_file_notice_evidence": notice,
                  "transfer_charged_bytes_total": ledger.value["charged_bytes"]}
        receipt["archives"].append(record)
        save(receipt_path, receipt)
        print(json.dumps({"category": item["category"], "verified_bytes": path.stat().st_size, "sha256": record["sha256"]}), flush=True)
    return receipt


def safe_extract(archive: Path, target: Path, remaining: int = EXPANSION_CAP) -> dict:
    """Extract only regular mesh/material/image data; reject traversal and links."""
    target = target.resolve()
    total, files, skipped = 0, 0, []
    allowed = {".obj", ".mtl", ".jpg", ".jpeg", ".png"}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in member.name or ":" in member.name:
                raise ValueError("Unsafe tar path")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError("Tar links/devices are forbidden")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Unsupported tar entry")
            total += member.size
            if total > remaining:
                raise ValueError("Archive expansion cap exceeded")
            if pure.suffix.lower() not in allowed:
                skipped.append(member.name)
                continue
            dest = target.joinpath(*pure.parts).resolve()
            if not dest.is_relative_to(target):
                raise ValueError("Resolved tar path escaped target")
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Missing tar member bytes")
            if dest.exists():
                old, incoming = sha(dest), hashlib.sha256()
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    incoming.update(block)
                if old != incoming.hexdigest() or dest.stat().st_size != member.size:
                    raise ValueError("Existing extracted asset differs; no overwrite")
            else:
                with dest.open("xb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
                if dest.stat().st_size != member.size:
                    raise ValueError("Extracted member size mismatch")
            files += 1
    return {"expanded_bytes": total, "data_files": files, "skipped_non_data_members": skipped}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--download", action="store_true")
    p.add_argument("--category", action="append", choices=["shoe", "handbag"], help="Optional category transport retry; same persistent combined cap")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--confirm-version-check", type=Path, help="Saved post-download official Drive revision metadata, independently queried through connector")
    a = p.parse_args()
    root = a.root.resolve()
    if a.download:
        download(root, a.category)
    folder = root / "datasets/reconstruction_reference"
    receipt_path = folder / "omni_acquisition_attempt.json"
    if a.confirm_version_check:
        receipt = json.loads(receipt_path.read_text())
        checked = json.loads(a.confirm_version_check.read_text())
        for item in ARCHIVES:
            x = checked[item["category"]]
            if x["currentRevisionId"] != item["revision"] or len(x["revisions"]) != 1:
                raise ValueError("Official archive revision changed; preparation refused")
            revision = x["revisions"][0]
            if int(revision["size"]) != item["size_bytes"] or revision["modifiedTime"] != item["modified_time"]:
                raise ValueError("Official archive metadata changed")
        receipt["version_check_status"] = "same_pinned_revision_before_and_after"
        receipt["post_download_revision_metadata"] = a.confirm_version_check.resolve().relative_to(root).as_posix()
        receipt["post_download_revision_metadata_sha256"] = sha(a.confirm_version_check)
        save(receipt_path, receipt)
    if a.extract:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("version_check_status") != "same_pinned_revision_before_and_after":
            raise ValueError("Complete post-download revision check before extraction")
        expanded, output = 0, {}
        for item in receipt["archives"]:
            path = root / item["path"]
            if sha(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                raise ValueError("Archive bytes differ from acquisition receipt")
            value = safe_extract(path, root / ".cache/omni_reference/extracted" / item["category"], EXPANSION_CAP - expanded)
            expanded += value["expanded_bytes"]
            output[item["category"]] = value
        save(folder / "extraction_receipt.json", output)
        print(json.dumps({"expanded_bytes": expanded, "categories": list(output)}))


if __name__ == "__main__":
    main()
