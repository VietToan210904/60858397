"""Synthetic temporary source/rights graphs; no changes to genuine datasets."""
import csv
import json
from pathlib import Path

import pytest

from stylesphere.common.io import sha256_file
from stylesphere.reporting.reference_packaging import approved_reference_assets, CC_BY_4_SHA256


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    root = tmp_path
    folder = root / "datasets/reconstruction_reference"
    assets = folder / "assets/gso-test"
    for name in ("input.png", "reference.glb", "original/meshes/model.obj"):
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic test fixture")
    source_licence = Path(__file__).resolve().parents[1] / "datasets/reconstruction/SOURCE_LICENSE_CC_BY_4_0.txt"
    licence = folder / "SOURCE_LICENSE_CC_BY_4_0.txt"
    licence.write_bytes(source_licence.read_bytes())
    assert sha256_file(licence) == CC_BY_4_SHA256
    relative = lambda p: p.relative_to(root).as_posix()
    source_url = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/test/1/test.zip"
    metadata = {"name": "test", "owner": "GoogleResearch", "version": 1, "filesize": 20, "license_id": 2}
    acquisition = {"objects": [{"source_name": "test", "version": 1, "archive_bytes": 20, "archive_sha256": "a"*64,
                    "archive_url": source_url, "metadata_before": metadata, "metadata_after": metadata}]}
    put(folder / "gso_acquisition_manifest.json", acquisition)
    rights = {"licence": "CC-BY-4.0", "redistributable": True, "asset_attribution": "Google Research; Copyright2020Google LLC",
              "objects": ["gso-test"], "licence_sha256": CC_BY_4_SHA256}
    put(folder / "RIGHTS_AND_ATTRIBUTION.json", rights)
    preparation = {"source_name": "test", "source_version": 1, "source_url": source_url, "archive_sha256": "a"*64,
                   "prepared_image_sha256": sha256_file(assets / "input.png"), "reference_mesh_sha256": sha256_file(assets / "reference.glb"),
                   "original_file_sha256": {relative(assets / "original/meshes/model.obj"): sha256_file(assets / "original/meshes/model.obj")}}
    put(assets / "preparation.json", preparation)
    row = {"object_id": "gso-test", "source_dataset": "Google Scanned Objects", "licence": "CC-BY-4.0", "redistributable": "true",
           "source_image": relative(assets / "input.png"), "image_sha256": sha256_file(assets / "input.png"),
           "reference_mesh": relative(assets / "reference.glb"), "mesh_sha256": sha256_file(assets / "reference.glb"),
           "preparation_metadata": relative(assets / "preparation.json")}
    for path, rows in ((folder / "objects.csv", [row]), (folder / "development/objects.csv", [])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    hashes = {relative(p): sha256_file(p) for p in folder.rglob("*") if p.is_file()}
    put(folder / "preparation_manifest.json", {"objects": [{"object_id": "gso-test"}], "asset_sha256": hashes})
    return root, folder, assets


def test_only_explicit_frozen_data_assets_are_approved(tmp_path):
    root, _, assets = fixture(tmp_path)
    approved = approved_reference_assets(root)
    assert approved == {p.resolve() for p in assets.rglob("*") if p.is_file()}


def test_unlisted_asset_is_not_blanket_approved(tmp_path):
    root, _, assets = fixture(tmp_path)
    (assets / "extra.png").write_bytes(b"synthetic unapproved")
    with pytest.raises(ValueError, match="Unlisted"):
        approved_reference_assets(root)


def test_changed_source_bytes_are_rejected(tmp_path):
    root, _, assets = fixture(tmp_path)
    (assets / "original/meshes/model.obj").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        approved_reference_assets(root)


def test_source_host_and_scope_tampering_are_rejected(tmp_path):
    root, folder, assets = fixture(tmp_path)
    manifest_path = folder / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prep_path = assets / "preparation.json"
    prep = json.loads(prep_path.read_text());prep["source_url"] = "https://example.org/test.zip";put(prep_path, prep)
    manifest["asset_sha256"][prep_path.relative_to(root).as_posix()] = sha256_file(prep_path)
    put(manifest_path, manifest)
    with pytest.raises(ValueError, match="official versioned source"):
        approved_reference_assets(root)


def test_manifest_path_escape_is_rejected(tmp_path):
    root, folder, _ = fixture(tmp_path)
    p = root / "outside.png";p.write_bytes(b"synthetic")
    m = json.loads((folder / "preparation_manifest.json").read_text())
    m["asset_sha256"]["outside.png"] = sha256_file(p)
    put(folder / "preparation_manifest.json", m)
    with pytest.raises(ValueError, match="escapes dataset"):
        approved_reference_assets(root)
