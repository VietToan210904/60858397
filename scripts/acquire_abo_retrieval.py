"""Acquire a bounded, reproducible ABO fashion subset from official S3 objects.

Only small images of selected products are fetched. No judgement is generated.
The listings shard is an explicit sampling-frame limitation, not a random draw
from all ABO products. Existing assets are retained and checksummed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/"
IMAGE_BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
CATEGORIES = ("SHOES", "BOOT", "SANDAL", "HANDBAG", "BACKPACK", "WALLET", "HAT", "EARRING", "FINEEARRING", "NECKLACE", "FINENECKLACEBRACELETANKLET", "RING", "FINERING")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_metadata(key, path, cap=10_000_000):
    if path.is_file():
        return path.read_bytes()
    response = requests.head(BASE + key, timeout=40)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])
    if size > cap:
        raise ValueError("Metadata exceeds the predeclared per-file budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        for start in range(0, size, 2 * 1024 * 1024):
            end = min(size - 1, start + 2 * 1024 * 1024 - 1)
            r = requests.get(BASE + key, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            r.raise_for_status()
            if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}" or len(r.content) != end - start + 1:
                raise ValueError("Invalid official S3 range response")
            stream.write(r.content)
    temporary.replace(path)
    return path.read_bytes()


def english(items):
    if not isinstance(items, list):
        return str(items or "")
    values = [x.get("value", "") for x in items if x.get("language_tag", "en_US").startswith("en")]
    return "; ".join(dict.fromkeys(str(x) for x in values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=8410)
    parser.add_argument("--max-download-mb", type=int, default=200)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not 1 <= args.size <= 1000 or not 1 <= args.max_download_mb <= 1000:
        raise ValueError("Require size1..1000 and maximum download budget1..1000 MB")
    if shutil.disk_usage(root).free < args.max_download_mb * 1_000_000 + 1_000_000_000:
        raise RuntimeError("Insufficient free disk space")
    cache = root / ".cache" / "abo"
    listing_key = "listings/metadata/listings_0.json.gz"
    listing_data = fetch_metadata(listing_key, cache / "listings_0.json.gz")
    image_key = "images/metadata/images.csv.gz"
    image_data = fetch_metadata(image_key, cache / "images.csv.gz")
    metadata = list(csv.DictReader(io.StringIO(gzip.decompress(image_data).decode("utf-8"))))
    by_image = {r["image_id"]: r for r in metadata}
    groups = {key: [] for key in CATEGORIES}
    all_rows = [json.loads(line) for line in gzip.decompress(listing_data).decode("utf-8").splitlines() if line]
    for row in all_rows:
        category = row.get("product_type", [{}])[0].get("value")
        if category in groups and row.get("main_image_id") in by_image:
            groups[category].append(row)
    rng = random.Random(args.seed)
    for rows in groups.values():
        rows.sort(key=lambda row: row["item_id"])
        rng.shuffle(rows)
    selected, position = [], 0
    while len(selected) < args.size:
        added = False
        for rows in groups.values():
            if position < len(rows) and len(selected) < args.size:
                selected.append(rows[position])
                added = True
        if not added:
            raise ValueError(f"Only{len(selected)} eligible fashion products in the declared sampling frame")
        position += 1
    target = root / "datasets" / "retrieval"
    assets = target / "assets" / "abo"
    assets.mkdir(parents=True, exist_ok=True)
    lock, transferred = threading.Lock(), [0]
    connections = threading.local()
    def session():
        if not hasattr(connections, "session"):
            connections.session = requests.Session()
            retries = Retry(total=4, connect=4, read=4, status=4, status_forcelist=(429, 500, 502, 503, 504), backoff_factor=1)
            connections.session.mount("https://", HTTPAdapter(max_retries=retries))
        return connections.session
    def acquire(row):
        item = by_image[row["main_image_id"]]
        image_id = row["main_image_id"]
        identifier = "p-" + sha(f"amazon-berkeley-objects:{row['item_id']}".encode())[:16]
        path = assets / f"{identifier}.jpg"
        key = "images/small/" + item["path"]
        if not path.exists():
            response = session().get(IMAGE_BASE + key, stream=True, timeout=(15, 60))
            response.raise_for_status()
            chunks = []
            for chunk in response.iter_content(65536):
                with lock:
                    transferred[0] += len(chunk)
                    if transferred[0] > args.max_download_mb * 1_000_000:
                        raise RuntimeError("Predeclared image download budget exceeded")
                chunks.append(chunk)
                if sum(map(len, chunks)) > 2_000_000:
                    raise ValueError("Unexpectedly large small-image asset")
            data = b"".join(chunks)
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with path.open("xb") as stream:
                stream.write(data)
        data = path.read_bytes()
        category = row["product_type"][0]["value"]
        attributes = " | ".join(f"{key}: {english(row.get(key, []))}" for key in ("item_name", "color", "material", "style", "pattern"))
        return {"source_id": row["item_id"], "image_path": path.relative_to(root).as_posix(), "category": category, "attributes": attributes, "source": "amazon-berkeley-objects", "source_revision": "sha256:" + sha(listing_data), "licence": "CC-BY-4.0", "image_sha256": sha(data), "source_image_id": image_id, "source_url": BASE + key, "redistributable": True}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(acquire, selected))
    destination = target / "abo_metadata.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise ValueError("Existing selected metadata differs; archive it before changing the dataset")
    destination.write_text(content, encoding="utf-8")
    receipt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "selected_products": len(records), "seed": args.seed, "sampling": "seeded round-robin category-balanced selection within explicitly declared listings_0 shard; not all ABO", "categories": {category: sum(row["category"] == category for row in records) for category in CATEGORIES}, "metadata": {listing_key: {"bytes": len(listing_data), "sha256": sha(listing_data)}, image_key: {"bytes": len(image_data), "sha256": sha(image_data)}}, "new_image_download_bytes": transferred[0], "download_budget_bytes": args.max_download_mb * 1_000_000, "free_disk_bytes": shutil.disk_usage(root).free, "metadata_path": destination.relative_to(root).as_posix(), "metadata_sha256": sha(destination.read_bytes()), "licence": "CC-BY-4.0", "attribution": "Amazon.com; ABO dataset by Collins et al., CVPR2022; https://registry.opendata.aws/amazon-berkeley-objects/", "relevance_judgements_created": False}
    (root / "environment" / "abo-retrieval-acquisition.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        diagnostic = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "failed_acquisition", "type": type(error).__name__, "message": str(error), "model_inference_attempted": False}
        directory = Path(__file__).resolve().parents[1] / "environment"
        directory.mkdir(exist_ok=True)
        with (directory / "abo-acquisition-failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(diagnostic) + "\n")
        raise
