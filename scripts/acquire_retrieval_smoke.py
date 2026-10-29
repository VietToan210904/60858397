"""Download at most 100 MB of genuine public Fashion200K smoke inputs.

This opt-in acquisition is separate from benchmark preparation. It never creates
relevance judgements and is not a representative or held-out evaluation sample.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

OFFSETS = [0, 25000, 50000, 75000, 100000, 125000, 150000, 175000]
LIMIT = 100_000_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_response(session: requests.Session, url: str, **kwargs):
    """Never include a signed image URL in request-failure diagnostics."""
    try:
        response = session.get(url, stream=True, timeout=45, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Pinned source request failed ({type(exc).__name__}); URL suppressed") from None
    if response.status_code != 200:
        response.close()
        raise RuntimeError(f"Pinned source unavailable (HTTP {response.status_code}); no replacement source used")
    return response


def _bounded_bytes(response, transferred: list[int], limit: int) -> bytes:
    declared_size = response.headers.get("Content-Length")
    if declared_size and int(declared_size) > limit - transferred[0]:
        response.close()
        raise RuntimeError("100 MB hard download limit would be exceeded; restoration stopped")
    chunks = []
    try:
        for chunk in response.iter_content(65536):
            if transferred[0] + len(chunk) > limit:
                raise RuntimeError("100 MB hard download limit exceeded; restoration stopped")
            transferred[0] += len(chunk)
            chunks.append(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(f"Pinned source stream failed ({type(exc).__name__}); URL suppressed") from None
    finally:
        response.close()
    return b"".join(chunks)


def restore_missing(root: Path, session=None, limit: int | None = None) -> dict:
    """Restore only missing images from a retained immutable manifest.

    Existing mismatched images are errors, not candidates for replacement. All
    downloaded images must pass the recorded SHA-256 before any image is written.
    The source manifest, catalogue, queries, card and licence remain byte-identical.
    """
    out = Path(root).resolve() / "datasets/retrieval_smoke"
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("--restore-missing requires an existing pinned smoke manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest["source_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Restoration requires the recorded full immutable source revision")
    if digest((out / "SOURCE_DATASET_CARD.md").read_bytes()) != manifest["source_card_sha256"]:
        raise RuntimeError("Existing source card hash mismatch")
    licence = manifest.get("licence_record")
    if not licence or Path(licence["filename"]).name != licence["filename"]:
        raise RuntimeError("Restoration requires the retained pinned licence record")
    if digest((out / licence["filename"]).read_bytes()) != licence["sha256"]:
        raise RuntimeError("Existing licence hash mismatch")
    missing = []
    for asset in manifest["assets"]:
        pid = asset["product_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            raise RuntimeError("Unsafe product identifier in smoke manifest")
        path = out / "images" / f"{pid}.jpg"
        if path.is_symlink() or not path.resolve().is_relative_to(out.resolve()):
            raise RuntimeError(f"Refusing symlink image path: {pid}")
        if path.exists():
            if digest(path.read_bytes()) != asset["image_sha256"]:
                raise RuntimeError(f"Existing smoke image hash mismatch: {pid}; no overwrite performed")
        else:
            missing.append((asset, path))
    session = session or requests.Session()
    limit = LIMIT if limit is None else min(limit, LIMIT)
    transferred, validated = [0], []
    for asset, path in missing:
        row_response = _safe_response(session, "https://datasets-server.huggingface.co/rows", params={
            "dataset": "Marqo/fashion200k", "config": "default", "split": "data",
            "offset": int(asset["row_index"]), "length": 1, "revision": revision})
        metadata = json.loads(_bounded_bytes(row_response, transferred, limit))
        record = metadata["rows"][0]
        if record["row_idx"] != asset["row_index"] or record["row"]["item_ID"] != asset["source_id"]:
            raise RuntimeError("Pinned source row identity differs; no replacement source used")
        url = record["row"]["image"]["src"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "datasets-server.huggingface.co" or f"/Marqo/fashion200k/--/{revision}/--/" not in parsed.path:
            raise RuntimeError("Pinned source revision unavailable from dataset server; no replacement source used")
        data = _bounded_bytes(_safe_response(session, url), transferred, limit)
        if digest(data) != asset["image_sha256"] or len(data) != asset["bytes"]:
            raise RuntimeError(f"Restored image hash/size mismatch: {asset['product_id']}; image not written")
        validated.append((path, data))
    for path, data in validated:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting an image created concurrently.
        with path.open("xb") as stream:
            stream.write(data)
    return {"status": "restored_missing" if validated else "verified_existing", "restored_images": len(validated),
            "downloaded_bytes_this_invocation": transferred[0], "catalogue_count": len(manifest["assets"]),
            "source_revision": revision, "source_manifest_unchanged": True}


def ensure_licence(session: requests.Session, out: Path, manifest: dict) -> None:
    """Retain the official upstream Apache licence, including its commit and hash."""
    if manifest.get("licence_record"):
        record = manifest["licence_record"]
        if digest((out / record["filename"]).read_bytes()) != record["sha256"]:
            raise RuntimeError("Existing licence hash mismatch")
        return
    info = session.get("https://api.github.com/repos/xthan/fashion-200k/commits/master", timeout=45)
    info.raise_for_status()
    revision = info.json()["sha"]
    url = f"https://raw.githubusercontent.com/xthan/fashion-200k/{revision}/LICENSE"
    response = session.get(url, timeout=45)
    response.raise_for_status()
    if b"Apache License" not in response.content:
        raise RuntimeError("Unexpected upstream licence text")
    filename = "SOURCE_LICENSE_APACHE_2_0.txt"
    (out / filename).write_bytes(response.content)
    manifest["licence_record"] = {"filename": filename, "source_url": url, "source_revision": revision, "sha256": digest(response.content)}
    manifest["downloaded_bytes"] += len(info.content) + len(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--restore-missing", action="store_true", help="Restore only missing images from the existing pinned manifest; enforce exact hashes and preserve all existing inputs.")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.restore_missing:
        print(json.dumps(restore_missing(root), indent=2))
        return
    out = root / "datasets/retrieval_smoke"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for asset in manifest["assets"]:
            path = out / "images" / f"{asset['product_id']}.jpg"
            if digest(path.read_bytes()) != asset["image_sha256"]:
                raise RuntimeError(f"Existing smoke image hash mismatch: {asset['product_id']}")
        if digest((out / "SOURCE_DATASET_CARD.md").read_bytes()) != manifest["source_card_sha256"]:
            raise RuntimeError("Existing source card hash mismatch")
        before_licence = bool(manifest.get("licence_record"))
        ensure_licence(session, out, manifest)
        if not before_licence:
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "verified_existing", "catalogue_count": len(manifest["assets"]),
                          "downloaded_bytes": manifest["downloaded_bytes"], "source_revision": manifest["source_revision"]}, indent=2))
        return
    info = session.get("https://huggingface.co/api/datasets/Marqo/fashion200k", timeout=45)
    info.raise_for_status()
    revision = info.json()["sha"]
    if len(revision) != 40:
        raise RuntimeError("A full dataset commit SHA is required")
    card_url = f"https://huggingface.co/datasets/Marqo/fashion200k/resolve/{revision}/README.md"
    card = session.get(card_url, timeout=45)
    card.raise_for_status()
    (out / "SOURCE_DATASET_CARD.md").write_bytes(card.content)
    downloaded_bytes = len(info.content) + len(card.content)
    catalogue, assets = [], []
    for offset in OFFSETS:
        response = session.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": "Marqo/fashion200k", "config": "default", "split": "data", "offset": offset, "length": 1}, timeout=45)
        response.raise_for_status()
        downloaded_bytes += len(response.content)
        record = response.json()["rows"][0]
        row = record["row"]
        url = row["image"]["src"]
        if f"/--/{revision}/--/" not in urlsplit(url).path:
            raise RuntimeError("Dataset server image revision does not match the pinned source revision")
        try:
            image = session.get(url, stream=True, timeout=45)
        except requests.RequestException as exc:
            raise RuntimeError(f"Image transfer failed ({type(exc).__name__}); signed URL suppressed") from None
        if image.status_code != 200:
            raise RuntimeError(f"Image transfer failed (HTTP {image.status_code}); signed URL suppressed")
        chunks = []
        try:
            for chunk in image.iter_content(65536):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > LIMIT:
                    raise RuntimeError("100 MB hard download limit exceeded; acquisition stopped")
                chunks.append(chunk)
        except requests.RequestException as exc:
            raise RuntimeError(f"Image stream failed ({type(exc).__name__}); signed URL suppressed") from None
        data = b"".join(chunks)
        source_id = row["item_ID"]
        product_id = "p-" + digest(f"Marqo/fashion200k:{source_id}".encode())[:16]
        image_path = out / "images" / f"{product_id}.jpg"
        image_path.parent.mkdir(exist_ok=True)
        image_path.write_bytes(data)
        from PIL import Image
        with Image.open(image_path) as decoded:
            decoded.verify()
        canonical_url = urlsplit(url)._replace(query="", fragment="").geturl()
        catalogue.append({"product_id": product_id, "image_path": image_path.relative_to(root).as_posix(),
                          "attributes": row["text"], "category": row["category1"], "source": "Marqo/fashion200k",
                          "source_id": source_id, "source_revision": revision, "licence": "Apache-2.0 (repository declaration; underlying-image caveat)",
                          "image_sha256": digest(data), "redistributable": "false"})
        assets.append({"product_id": product_id, "source_id": source_id, "row_index": record["row_idx"],
                       "url_without_signed_query": canonical_url, "image_sha256": digest(data), "bytes": len(data)})
    def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
        with (out / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_csv("catalogue.csv", list(catalogue[0]), catalogue)
    queries = [{"query_id": f"smoke-q{i}", "query_text": query, "query_category": "predefined smoke query", "qrels_complete": "false"}
               for i, query in enumerate(["a green dress", "a dark jacket", "a pair of trousers", "a patterned blouse"], 1)]
    write_csv("queries.csv", list(queries[0]), queries)
    write_csv("qrels.csv", ["query_id", "product_id", "relevance", "annotator_id", "annotation_date"], [])
    manifest = {"schema_version": 1, "purpose": "real-model smoke inference only; not final benchmark",
                "source": "https://huggingface.co/datasets/Marqo/fashion200k", "source_revision": revision,
                "accessed_at_utc": datetime.now(timezone.utc).isoformat(), "planned_max_download_bytes": LIMIT,
                "downloaded_bytes": downloaded_bytes, "selection": {"method": "predeclared fixed row offsets", "offsets": OFFSETS},
                "source_card_sha256": digest(card.content), "source_card_url": card_url, "assets": assets,
                "catalogue_count": len(catalogue), "query_count": len(queries), "judgement_count": 0,
                "metrics_eligible": False, "redistributable_assets": False, "synthetic": False,
                "limitation": "Known evaluation exposure to FashionSigLIP; eight convenience cases cannot establish retrieval quality."}
    ensure_licence(session, out, manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalogue_count": len(catalogue), "query_count": len(queries), "downloaded_bytes": downloaded_bytes,
                      "source_revision": revision, "manifest": str(out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
