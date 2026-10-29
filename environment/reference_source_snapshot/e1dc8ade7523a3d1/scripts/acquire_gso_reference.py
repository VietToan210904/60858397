"""Acquire an official Google Scanned Objects fashion reference subset.

Uses public, versioned GoogleResearch Gazebo Fuel ZIPs, never credentials.
The complete source catalogue determines lexical selection before model output.
Twenty test objects comprise17shoes+3backpacks, with2development per category.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
from urllib.parse import quote, urlsplit

BASE = "https://fuel.gazebosim.org/1.0"
NETWORK_CAP = 3_000_000_000
EXPANSION_CAP = 20_000_000_000
OFFICIAL_SOURCE = "https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() == content:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(content)
    temp.replace(path)


def safe_zip(archive: Path, target: Path, remaining: int = EXPANSION_CAP):
    target = target.resolve()
    total, count, skipped = 0, 0, []
    allowed = {".obj", ".mtl", ".jpg", ".jpeg", ".png", ".config", ".sdf", ".pbtxt", ".txt"}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe ZIP path")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("ZIP links/devices forbidden")
            if info.flag_bits & 1:
                raise ValueError("Encrypted ZIP forbidden")
            total += info.file_size
            if total > remaining:
                raise ValueError("ZIP expansion cap exceeded")
            if info.is_dir():
                continue
            if p.suffix.lower() not in allowed:
                skipped.append(info.filename)
                continue
            path = target.joinpath(*p.parts).resolve()
            if not path.is_relative_to(target):
                raise ValueError("ZIP path escaped target")
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source:
                if path.exists():
                    h = hashlib.sha256()
                    for b in iter(lambda: source.read(1024 * 1024), b""):
                        h.update(b)
                    if path.stat().st_size != info.file_size or sha(path) != h.hexdigest():
                        raise ValueError("Existing extraction differs; no overwrite")
                else:
                    with path.open("xb") as stream:
                        shutil.copyfileobj(source, stream, 1024 * 1024)
            count += 1
    return {"expanded_bytes": total, "data_files": count, "skipped_non_data_members": skipped}


def category(x):
    if x.get("owner") != "GoogleResearch" or x.get("license_id") != 2 or x.get("private"):
        return None
    if "Shoe" in x.get("categories", []):
        return "shoe"
    # Do not relabel the source's lunch boxes/pencil cases as fashion bags.
    if "Bag" in x.get("categories", []) and "backpack" in (x["name"] + " " + x.get("description", "")).lower():
        return "backpack"
    return None


class Budget:
    def __init__(self, path):
        self.path = path
        self.value = json.loads(path.read_text()) if path.exists() else {
            "network_cap_bytes": NETWORK_CAP, "charged_bytes": 2_000_000,
            "conservative_prior_metadata_and_omni_probes_reserve_bytes": 2_000_000}

    def add(self, amount):
        self.value["charged_bytes"] += amount
        save(self.path, self.value)
        if self.value["charged_bytes"] > NETWORK_CAP:
            raise ValueError("Shared transfer/retry cap exhausted")


def get_json(session, url, budget):
    r = session.get(url, timeout=(20, 45), allow_redirects=False)
    budget.add(len(r.content))
    r.raise_for_status()
    if r.is_redirect or "application/json" not in r.headers.get("Content-Type", ""):
        raise ValueError("Unexpected official metadata response")
    return r.json()


def acquire(root):
    import requests
    folder, cache = root / "datasets/reconstruction_reference", root / ".cache/gso_reference"
    cache.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    budget = Budget(cache / "network_ledger.json")
    session = requests.Session()
    session.trust_env = False
    catalogue_path = cache / "official_models_metadata.json"
    if not catalogue_path.exists():
        retained = folder / "official_source_catalogue.json"
        if not retained.is_file():
            raise ValueError("First save the complete official paginated public owner catalogue")
        shutil.copyfile(retained, catalogue_path)
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    candidates = sorted([x for x in catalogue if category(x)], key=lambda x: x["name"])
    selection_path = folder / "source_selection.json"
    plan = {"source": "Google Scanned Objects", "owner": "GoogleResearch", "official_source": OFFICIAL_SOURCE,
            "catalogue_sha256": sha(catalogue_path), "catalogue_model_count": len(catalogue),
            "selection_policy": "Case-sensitive lexical source model name; first2 valid/category development; then17shoe+3backpack test; source geometry/texture QA only, before any model outputs.",
            "candidate_counts": {c: sum(category(x) == c for x in candidates) for c in ("shoe", "backpack")},
            "objects": []}
    # Freeze candidate order. Geometry QA may exclude invalid sources, documented later.
    for c, count in (("shoe", 19), ("backpack", 5)):
        for index, x in enumerate([x for x in candidates if category(x) == c][:count]):
            plan["objects"].append({"source_name": x["name"], "category": c,
                                    "intended_split": "development" if index < 2 else "test", "source_metadata": x})
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        if old != plan:
            raise ValueError("Frozen source selection differs; no replacement")
    else:
        save(selection_path, plan)
    if shutil.disk_usage(root).free < 5_000_000_000:
        raise ValueError("Need5GB disk for bounded subset and preparation")
    receipt_path = folder / "gso_acquisition_manifest.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        "source": "Google Scanned Objects", "licence": "CC-BY-4.0", "source_selection_sha256": sha(selection_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "network_cap_bytes": NETWORK_CAP,
        "expansion_cap_bytes": EXPANSION_CAP, "objects": [], "model_inference_performed": False}
    expanded = 0
    for item in plan["objects"]:
        name = item["source_name"]
        urlbase = f"{BASE}/GoogleResearch/models/{quote(name, safe='')}"
        before = get_json(session, urlbase, budget)
        if before.get("owner") != "GoogleResearch" or before.get("license_id") != 2 or before.get("private"):
            raise ValueError("Official ownership/CCBY4 visibility changed")
        version, size = int(before["version"]), int(before["filesize"])
        if size != item["source_metadata"]["filesize"] or size > 100_000_000:
            raise ValueError("Pinned metadata size changed/oversized source")
        path = cache / f"{name}-v{version}.zip"
        prior = next((x for x in receipt["objects"] if x["source_name"] == name), None)
        if path.exists():
            if not prior or prior["version"] != version or sha(path) != prior["archive_sha256"] or path.stat().st_size != size:
                raise ValueError("Existing source ZIP differs from pinned receipt")
        else:
            if budget.value["charged_bytes"] + size > NETWORK_CAP:
                raise ValueError("Pinned ZIP cannot fit remaining network budget")
            download_url = f"{urlbase}/{version}/{quote(name, safe='')}.zip"
            with session.get(download_url, stream=True, timeout=(20, 90), allow_redirects=False) as r:
                r.raise_for_status()
                if r.is_redirect or r.headers.get("Content-Type", "").split(";")[0] != "application/zip":
                    raise ValueError("Unexpected source ZIP response; no redirect or provider warning bypass")
                if int(r.headers.get("Content-Length", -1)) != size or int(r.headers.get("X-Ign-Resource-Version", -1)) != version:
                    raise ValueError("Versioned ZIP response differs from pinned official metadata")
                partial = path.with_suffix(".partial")
                if partial.exists():
                    raise ValueError("Incomplete source ZIP retained; inspect explicitly before retry")
                with partial.open("xb") as stream:
                    for block in r.iter_content(1024 * 1024):
                        budget.add(len(block))
                        if stream.tell() + len(block) > size:
                            raise ValueError("Source ZIP exceeds pinned size")
                        stream.write(block)
                if partial.stat().st_size != size:
                    raise ValueError("Truncated source ZIP")
                partial.replace(path)
        after = get_json(session, urlbase, budget)
        stable_fields = ("name", "owner", "version", "filesize", "modify_date", "license_id", "license_url")
        if any(before.get(k) != after.get(k) for k in stable_fields):
            raise ValueError("Source metadata/version changed during transfer")
        extracted = cache / "extracted" / name
        extraction = safe_zip(path, extracted, EXPANSION_CAP - expanded)
        expanded += extraction["expanded_bytes"]
        record = {"source_name": name, "category": item["category"], "intended_split": item["intended_split"],
                  "version": version, "archive_bytes": size, "archive_sha256": sha(path),
                  "archive_url": f"{urlbase}/{version}/{quote(name, safe='')}.zip",
                  "archive_path": path.relative_to(root).as_posix(), "extracted_path": extracted.relative_to(root).as_posix(),
                  "metadata_before": {k: before.get(k) for k in stable_fields},
                  "metadata_after": {k: after.get(k) for k in stable_fields}, "extraction": extraction,
                  "licence": "CC-BY-4.0", "attribution": "Google Research", "redistributable": True}
        if prior:
            if record != prior:
                raise ValueError("Acquisition record differs on verification")
        else:
            receipt["objects"].append(record)
            save(receipt_path, receipt)
        print(json.dumps({"source_name": name, "version": version, "verified_bytes": size}), flush=True)
    print(json.dumps({"verified_objects": len(receipt["objects"]), "expanded_bytes": expanded, "charged_network_bytes": budget.value["charged_bytes"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    acquire(args.root.resolve())
