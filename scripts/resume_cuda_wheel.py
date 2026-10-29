"""Recover the pinned Windows Python 3.11 CUDA torch wheel without installing it.

An optional source prefix is copied, never modified. Only a complete file that
matches the SHA256 published by the official PyTorch index becomes a .whl.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import time
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

FILENAME = "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl"
INDEX = "https://download.pytorch.org/whl/cu128/torch/"
URL = "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl"
SHA256 = "90ef0c2454e5296a9fb021ddd42252e4ce1abe2c0a4988a173ef90a6cded0bf5"
TOTAL_BYTES = 2_753_148_611


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag == "a":
            self.links.extend(value for key, value in attributes if key == "href" and value)


def official_link() -> str:
    with urlopen(INDEX, timeout=30) as response:
        body = response.read(2 * 1024 * 1024 + 1)
    if len(body) > 2 * 1024 * 1024:
        raise ValueError("Unexpectedly large official wheel index")
    parser = IndexLinks()
    parser.feed(body.decode("utf-8"))
    for link in parser.links:
        parsed = urlsplit(link)
        if unquote(parsed.path.rsplit("/", 1)[-1]) != FILENAME:
            continue
        if parsed.scheme != "https" or parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
            raise ValueError("Wheel index points outside the official PyTorch hosts")
        if parsed.fragment != "sha256=" + SHA256:
            raise ValueError("Official wheel checksum differs from the pinned checksum; investigate before downloading")
        return link
    raise ValueError("Exact pinned Windows cp311 CUDA wheel is absent from the official index")


def validate_range(status: int, content_range: str, length: int, start: int, end: int, total: int) -> None:
    if status != 206 or content_range != f"bytes {start}-{end}/{total}" or length != end - start + 1:
        raise ValueError(f"Server did not honour bounded range {start}-{end}: HTTP {status}, Content-Range={content_range!r}, bytes={length}")


def fetch_range(start: int, end: int, total: int) -> bytes:
    request = Request(URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=45) as response:
        status = response.status
        header = response.headers.get("Content-Range", "")
        # Reject a whole-file response before reading it into memory.
        if status != 206 or header != f"bytes {start}-{end}/{total}":
            raise ValueError(f"Invalid ranged response: HTTP {status}, Content-Range={header!r}")
        data = response.read(end - start + 2)
    validate_range(status, header, len(data), start, end, total)
    return data


def recover(args) -> Path:
    destination = args.destination.expanduser().resolve()
    if destination.is_dir():
        destination = destination / FILENAME
    if destination.name != FILENAME:
        raise ValueError(f"Destination must be a directory or a file named {FILENAME}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def event(**details):
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **details}
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    source_link = official_link()
    if destination.exists():
        if destination.stat().st_size != TOTAL_BYTES or sha256_file(destination) != SHA256:
            raise ValueError("Existing destination fails the official size/checksum; it has not been modified")
        event(status="already_verified", path=str(destination), bytes=TOTAL_BYTES, sha256=SHA256, official_index_link=source_link)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if not partial.exists():
        if args.source_prefix:
            source = args.source_prefix.expanduser().resolve()
            if not source.is_file() or source.stat().st_size > TOTAL_BYTES:
                raise ValueError("Source prefix must be an existing file no larger than the selected wheel")
            shutil.copyfile(source, partial)
            event(status="prefix_copied", copied_bytes=partial.stat().st_size)
        else:
            partial.touch()
    start = partial.stat().st_size
    if start > TOTAL_BYTES:
        raise ValueError("Partial file is larger than the pinned wheel")
    remaining = TOTAL_BYTES - start
    if shutil.disk_usage(destination.parent).free < remaining * 2 + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk space for remaining ranges and assembled wheel")
    # This small probe validates the published total and HTTP range behavior.
    fetch_range(0, 1023, TOTAL_BYTES)
    event(status="resume_started", offset=start, total_bytes=TOTAL_BYTES, remaining_bytes=remaining, official_index_link=source_link)
    chunk = args.chunk_mib * 1024 * 1024
    ranges = [(a, min(a + chunk, TOTAL_BYTES) - 1) for a in range(start, TOTAL_BYTES, chunk)]

    def download(bounds):
        a, b = bounds
        fragment = partial.with_name(partial.name + f".range-{a}-{b}")
        if fragment.exists() and fragment.stat().st_size == b - a + 1:
            return fragment
        for attempt in range(args.retries):
            try:
                data = fetch_range(a, b, TOTAL_BYTES)
                temporary = fragment.with_suffix(fragment.suffix + ".tmp")
                temporary.write_bytes(data)
                temporary.replace(fragment)
                return fragment
            except Exception:
                if attempt + 1 == args.retries:
                    raise
                time.sleep(min(attempt + 1, 3))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download, bounds): bounds for bounds in ranges}
        for future in as_completed(jobs):
            fragment = future.result()
            a, b = jobs[future]
            event(status="range_saved", start=a, end=b, bytes=fragment.stat().st_size)
    with partial.open("ab") as target:
        for a, b in ranges:
            fragment = partial.with_name(partial.name + f".range-{a}-{b}")
            with fragment.open("rb") as stream:
                shutil.copyfileobj(stream, target, 8 * 1024 * 1024)
    actual = sha256_file(partial)
    if partial.stat().st_size != TOTAL_BYTES or actual != SHA256:
        event(status="checksum_failed", expected_sha256=SHA256, actual_sha256=actual)
        raise ValueError("Completed partial fails the official checksum; retained for diagnosis, never installed")
    partial.replace(destination)
    for a, b in ranges:
        partial.with_name(partial.name + f".range-{a}-{b}").unlink()
    event(status="wheel_verified", path=str(destination), bytes=TOTAL_BYTES, sha256=actual, installation_performed=False)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recover the exact official torch2.11.0+cu128 Windows cp311 wheel through bounded HTTP ranges; verifies SHA256 and never installs packages.")
    parser.add_argument("--destination", type=Path, default=Path(".cache/wheels") / FILENAME, help="Complete .whl path, or existing directory (default: .cache/wheels/<pinned filename>)")
    parser.add_argument("--source-prefix", type=Path, help="Optional existing incomplete wheel; copied without changing its source")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ranged requests, 1-8 (default: 4)")
    parser.add_argument("--chunk-mib", type=int, default=8, help="Maximum response size in MiB, 1-32 (default: 8)")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per failed range, 1-5 (default: 3)")
    parser.add_argument("--log", type=Path, default=Path("environment/cuda-wheel-recovery.jsonl"), help="JSONL acquisition transcript")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or not 1 <= args.chunk_mib <= 32 or not 1 <= args.retries <= 5:
        parser.error("workers must be 1-8, chunk-mib 1-32, and retries 1-5")
    try:
        recover(args)
    except Exception as exc:
        parser.exit(1, f"Recovery failed: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
