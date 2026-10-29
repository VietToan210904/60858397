"""Bounded, journalled recovery of explicitly planned official Hugging Face files.

No model is loaded or installed. Large files require their official LFS SHA256.
Partial source files are copied and never modified. Only verified full files are
published under their final names, and no expiring signed CDN URL is logged.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from .io import sha256_file, utc_now


def _origin(url: str) -> tuple:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


class CredentialBoundRedirectHandler(HTTPRedirectHandler):
    """Credentials stay on their original HTTPS origin, never signed storage."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme.lower() != "https":
            raise ValueError("Refusing a non-HTTPS checkpoint redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            for header in list(redirected.headers) + list(redirected.unredirected_hdrs):
                if header.lower() in {"authorization", "cookie", "proxy-authorization"}:
                    redirected.remove_header(header)
        return redirected


def _cached_token() -> str:
    # Read the official cached login file, never a token argument or CLI value.
    from huggingface_hub.constants import HF_TOKEN_PATH
    path = Path(HF_TOKEN_PATH)
    if not path.is_file():
        raise RuntimeError("No cached Hugging Face login is available; authenticate with the official hf CLI first")
    token = path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("The cached Hugging Face login credential has an invalid format")
    return token


def _valid_file(path: Path, item: dict) -> bool:
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        return False
    if item.get("expected_sha256"):
        return sha256_file(path) == item["expected_sha256"]
    if item.get("expected_git_blob_sha1"):
        data = path.read_bytes()
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == item["expected_git_blob_sha1"]
    return item["size_bytes"] <= 50_000_000


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Retry short Windows scanner locks locally, without re-fetching bytes."""
    for attempt in range(5):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * 2**attempt)


def _approved_source_url(item: dict) -> str | None:
    """Permit only the reviewed official OpenAI CLIP content-addressed origin."""
    url = item.get("source_url")
    if url is None:
        return None
    parsed = urlsplit(url)
    digest, filename = item.get("expected_sha256", ""), item["path"]
    if (parsed.scheme != "https" or parsed.netloc != "openaipublic.azureedge.net" or parsed.query or parsed.fragment
            or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            or Path(filename).name != filename or parsed.path != f"/clip/models/{digest}/{filename}"):
        raise ValueError("Alternate source must be the exact approved HTTPS OpenAI CLIP URL with its official SHA256 in the path")
    return url


def download_plan(evidence: dict, *, workers: int = 4, chunk_bytes: int = 8 * 1024**2,
                  log_path: Path | None = None, authenticated: bool = False) -> dict:
    """Download a source-audited plan; honour immutable revision and exact file list.

    Each file entry accepts optional ``source_prefix`` and requires ``path`` and
    ``size_bytes``. Files over 50 MB require official ``expected_sha256``. The
    caller must obtain download approval and perform size/disk planning first.
    """
    if type(workers) is not int or not 1 <= workers <= 16 or not 1 <= chunk_bytes <= 64 * 1024**2:
        raise ValueError("Use 1-16 workers and chunks no larger than 64 MiB")
    if type(authenticated) is not bool:
        raise ValueError("authenticated must be a Boolean; tokens are read only from the cached login")
    token = _cached_token() if authenticated else None
    open_request = build_opener(CredentialBoundRedirectHandler()).open if authenticated else urlopen
    revision = evidence["resolved_revision"]
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("An immutable 40-character official repository revision is required")
    destination = Path(evidence["destination"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / ".cache/stylesphere-ranges"
    cache.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    log_path = Path(log_path) if log_path else cache / "events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def event(**details):
        with lock:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp_utc": utc_now(), **details}) + "\n")
        if details.get("status") in {"file_verified", "download_complete"}:
            print(json.dumps(details), flush=True)

    states = []
    jobs = []
    for item in evidence["files"]:
        source_url = _approved_source_url(item)
        name = item["path"]
        target = (destination / name).resolve()
        if not target.is_relative_to(destination) or Path(name).is_absolute():
            raise ValueError("A planned file path escapes its repository destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        if item["size_bytes"] > 50_000_000 and not item.get("expected_sha256"):
            raise ValueError(f"Large planned file lacks official SHA256: {name}")
        if _valid_file(target, item):
            event(status="file_verified", path=name, bytes=item["size_bytes"], reused=True)
            continue
        identifier = hashlib.sha256(name.encode()).hexdigest()[:20]
        partial, journal = cache / (identifier + ".part"), cache / (identifier + ".json")
        descriptor = {"model_id": evidence["model_id"], "revision": revision, "file": name,
                      "size_bytes": item["size_bytes"], "sha256": item.get("expected_sha256"), "chunk_bytes": chunk_bytes}
        # Existing default-HF journals remain byte-for-byte schema compatible.
        # Explicit alternate origins must be retained to forbid source switching.
        if source_url is not None:
            descriptor["source_url"] = source_url
        if journal.exists():
            state = json.loads(journal.read_text(encoding="utf-8"))
            if any(state.get(key) != value for key, value in descriptor.items() if key != "chunk_bytes") or state.get("source_url") != source_url or not partial.exists():
                raise ValueError(f"Partial download journal differs from the planned checkpoint: {name}")
            previous_chunk = state["chunk_bytes"]
            if previous_chunk != chunk_bytes:
                if previous_chunk < chunk_bytes or previous_chunk % chunk_bytes:
                    raise ValueError("Chunk migration supports a smaller exact divisor of the previous chunk size")
                completed = []
                for old_start in state["completed_starts"]:
                    completed.extend(range(old_start, min(old_start + previous_chunk, item["size_bytes"]), chunk_bytes))
                state.update(chunk_bytes=chunk_bytes, completed_starts=sorted(set(completed)))
                journal.write_text(json.dumps(state), encoding="utf-8")
                event(status="chunk_size_migrated", path=name, previous_chunk_bytes=previous_chunk,
                      chunk_bytes=chunk_bytes, preserved_completed_ranges=len(completed))
        else:
            candidates = []
            if item.get("source_prefix"):
                candidates.append(Path(item["source_prefix"]))
            if item.get("expected_sha256"):
                candidates.extend((destination / ".cache/huggingface/download").glob(f"*.{item['expected_sha256']}.incomplete"))
                candidates.append(destination.parent.parent / "blobs" / (item["expected_sha256"] + ".incomplete"))
            candidates = [path for path in candidates if path.is_file() and 0 < path.stat().st_size <= item["size_bytes"]]
            prefix = max(candidates, key=lambda path: path.stat().st_size) if candidates else None
            if prefix:
                shutil.copyfile(prefix, partial)
            else:
                partial.touch()
            state = {**descriptor, "prefix_bytes": partial.stat().st_size, "completed_starts": []}
            journal.write_text(json.dumps(state), encoding="utf-8")
            event(status="prefix_preserved", path=name, prefix_bytes=state["prefix_bytes"])
        states.append((item, target, partial, journal, state))
        done = set(state["completed_starts"])
        for start in range(state["prefix_bytes"], item["size_bytes"], chunk_bytes):
            if start not in done:
                jobs.append((item, partial, journal, state, start, min(start + chunk_bytes, item["size_bytes"]) - 1))
    remaining = sum(end - start + 1 for _, _, _, _, start, end in jobs)
    if shutil.disk_usage(destination).free < remaining + 1_000_000_000:
        raise RuntimeError("Insufficient disk space for planned remaining checkpoint bytes")
    event(status="download_started", model_id=evidence["model_id"], revision=revision, remaining_bytes=remaining, ranges=len(jobs), workers=workers)

    def fetch(job):
        item, partial, journal, state, start, end = job
        url = _approved_source_url(item) or f"https://huggingface.co/{evidence['model_id']}/resolve/{revision}/{quote(item['path'], safe='/')}"
        last_error = None
        for attempt in range(2):
            try:
                request_headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                if token is not None and _origin(url) == ("https", "huggingface.co", 443):
                    request_headers["Authorization"] = "Bearer " + token
                request = Request(url, headers=request_headers)
                with open_request(request, timeout=30) as response:
                    status = response.status
                    content_range = response.headers.get("Content-Range", "")
                    expected_range = f"bytes {start}-{end}/{item['size_bytes']}"
                    small_whole = status == 200 and start == 0 and end + 1 == item["size_bytes"] and item["size_bytes"] <= chunk_bytes
                    if not small_whole and (status != 206 or content_range != expected_range):
                        raise ValueError(f"Invalid bounded response HTTP {status}, range {content_range!r}")
                    deadline = time.monotonic() + 60
                    pieces = []
                    remaining = end - start + 1
                    while remaining:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Bounded range exceeded the 60-second body deadline")
                        piece = response.read(min(1024 * 1024, remaining))
                        if not piece:
                            break
                        pieces.append(piece)
                        remaining -= len(piece)
                    data = b"".join(pieces)
                if len(data) != end - start + 1:
                    raise ValueError("Bounded response length differs from planned range")
                break
            except Exception as exc:
                last_error = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else f"{type(exc).__name__}: {exc}"
                event(status="range_retry", path=item["path"], start=start, end=end, attempt=attempt + 1, error=last_error)
                if attempt < 1:
                    time.sleep(1)
        else:
            raise RuntimeError(f"Bounded download failed for {item['path']} bytes {start}-{end}: {last_error}")
        # Local persistence is deliberately outside the network retry loop.
        # A scanner lock must not trigger another request for verified bytes.
        with partial.open("r+b") as stream:
            stream.seek(start)
            stream.write(data)
            stream.flush()
        with lock:
            state["completed_starts"] = sorted(set(state["completed_starts"]) | {start})
            temporary = journal.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            _replace_with_retry(temporary, journal)
        event(status="range_saved", path=item["path"], start=start, end=end, bytes=len(data))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            # Retain completed journals but do not keep processing thousands of
            # queued ranges after a genuine transport failure or interruption.
            for pending in futures:
                pending.cancel()
            raise
    for item, target, partial, journal, state in states:
        if not _valid_file(partial, item):
            raise ValueError(f"Completed file differs from official size/SHA256: {item['path']}; partial retained for diagnosis")
        _replace_with_retry(partial, target)
        journal.unlink()
        event(status="file_verified", path=item["path"], bytes=item["size_bytes"], sha256=item.get("expected_sha256"), reused=False)
    result = {"model_id": evidence["model_id"], "resolved_revision": revision, "status": "downloaded",
              "sha256": {item["path"]: sha256_file(destination / item["path"]) for item in evidence["files"]},
              "transport": "bounded_http_ranges", "completed_at_utc": utc_now()}
    event(status="download_complete", model_id=evidence["model_id"], files=len(evidence["files"]))
    return result
