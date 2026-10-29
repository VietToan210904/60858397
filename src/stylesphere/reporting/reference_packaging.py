"""Explicit redistribution whitelist for the separately prepared GSO subset."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote, urlsplit

from stylesphere.common.io import sha256_file

CC_BY_4_SHA256 = "419896aea50c15d6e40c5b4baf4bd346f78223b9a354c7357ad00262afdc08ec"
SAFE_DATA_SUFFIXES = {".obj", ".mtl", ".png", ".jpg", ".jpeg", ".glb", ".json", ".config", ".sdf", ".pbtxt"}


def approved_reference_assets(root: Path) -> set[Path]:
    root = Path(root).resolve()
    folder = root / "datasets/reconstruction_reference"
    manifest_path = folder / "preparation_manifest.json"
    if not manifest_path.exists():
        return set()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    rights, manifest = read(folder / "RIGHTS_AND_ATTRIBUTION.json"), read(manifest_path)
    if rights.get("licence") != "CC-BY-4.0" or rights.get("redistributable") is not True or rights.get("asset_attribution") != "Google Research; Copyright2020Google LLC":
        raise ValueError("GSO redistribution lacks explicit source-specific rights")
    licence = folder / "SOURCE_LICENSE_CC_BY_4_0.txt"
    if rights.get("licence_sha256") != CC_BY_4_SHA256 or sha256_file(licence) != CC_BY_4_SHA256:
        raise ValueError("GSO CC BY4 licence differs from verified official legal text")
    acquisition = read(folder / "gso_acquisition_manifest.json")
    acquisitions = {x["source_name"]: x for x in acquisition["objects"]}
    if len(acquisitions) != len(acquisition["objects"]):
        raise ValueError("Duplicate GSO source acquisition identity")
    rows = []
    for p in (folder / "objects.csv", folder / "development/objects.csv"):
        with p.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    ids = [x["object_id"] for x in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(rights.get("objects", [])) or set(ids) != {x["object_id"] for x in manifest["objects"]}:
        raise ValueError("GSO rows/rights/preparation object scope differs")
    approved = set()
    hashes = manifest["asset_sha256"]

    def check(value, expected=None):
        path = Path(value)
        if path.is_absolute():
            raise ValueError("GSO packaging paths must be root-relative")
        candidate = root / path
        if any(p.is_symlink() for p in (candidate, *candidate.parents) if p.is_relative_to(folder)):
            raise ValueError("GSO packaging symlinks are forbidden")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(folder.resolve()):
            raise ValueError("GSO packaging path escapes dataset")
        relative = resolved.relative_to(root).as_posix()
        if not resolved.is_file() or relative not in hashes or sha256_file(resolved) != hashes[relative] or expected and hashes[relative] != expected:
            raise ValueError("GSO asset missing or hash differs")
        return resolved

    for path in hashes:
        check(path)

    for row in rows:
        if row.get("source_dataset") != "Google Scanned Objects" or row.get("licence") != "CC-BY-4.0" or row.get("redistributable") != "true":
            raise ValueError("GSO row permission/source differs")
        prep = read(check(row["preparation_metadata"]))
        name, version = prep["source_name"], prep["source_version"]
        if row["object_id"] != "gso-" + name or name not in acquisitions:
            raise ValueError("GSO preparation/source identity differs")
        original = acquisitions[name]
        expected_url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{quote(name, safe='')}/{version}/{quote(name, safe='')}.zip"
        if prep.get("source_url") != expected_url or original.get("archive_url") != expected_url or prep.get("archive_sha256") != original.get("archive_sha256") or original.get("version") != version:
            raise ValueError("GSO asset lacks exact official versioned source")
        parsed = urlsplit(expected_url)
        if parsed.query or parsed.fragment or parsed.netloc != "fuel.gazebosim.org":
            raise ValueError("GSO source host differs")
        for when in ("metadata_before", "metadata_after"):
            meta = original[when]
            if meta.get("owner") != "GoogleResearch" or meta.get("name") != name or meta.get("version") != version or meta.get("license_id") != 2 or meta.get("filesize") != original["archive_bytes"]:
                raise ValueError("GSO source ownership/licence/version receipt differs")
        if original["metadata_before"] != original["metadata_after"]:
            raise ValueError("GSO source changed during acquisition")
        for field, hash_field in (("source_image", "image_sha256"), ("reference_mesh", "mesh_sha256")):
            check(row[field], row[hash_field])
        if prep.get("prepared_image_sha256") != row["image_sha256"] or prep.get("reference_mesh_sha256") != row["mesh_sha256"]:
            raise ValueError("GSO prepared asset differs from row")
        asset_root = folder / "assets" / row["object_id"]
        for path, expected in prep["original_file_sha256"].items():
            original_path = check(path, expected)
            if not original_path.is_relative_to(asset_root / "original"):
                raise ValueError("GSO original asset leaves authorised object directory")
        object_assets = [check(path) for path in hashes if (root / path).is_relative_to(asset_root)]
        if not object_assets:
            raise ValueError("GSO object has no frozen assets")
        for path in object_assets:
            if path.suffix.lower() not in SAFE_DATA_SUFFIXES:
                raise ValueError("Non-data GSO asset cannot be redistributed")
            approved.add(path)
    actual = {p.resolve() for p in (folder / "assets").rglob("*") if p.is_file()}
    if actual != approved:
        raise ValueError("Unlisted GSO asset found; no blanket redistribution")
    return approved
