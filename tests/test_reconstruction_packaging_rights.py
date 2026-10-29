"""Synthetic permission/provenance fixtures only; no benchmark records."""
import csv
import hashlib
import json

import pytest

from stylesphere.reporting.packaging import approved_dataset_assets


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(tmp_path):
    folder = tmp_path / "datasets/reconstruction"
    assets = folder / "assets"
    assets.mkdir(parents=True)
    original = assets / "original.jpg"
    prepared = assets / "prepared.png"
    rejected = assets / "rejected.jpg"
    original.write_bytes(b"SYNTHETIC ORIGINAL PHOTO FIXTURE")
    prepared.write_bytes(b"SYNTHETIC PREPARED PHOTO FIXTURE")
    rejected.write_bytes(b"SYNTHETIC QA-EXCLUDED PHOTO FIXTURE")
    licence = folder / "SOURCE_LICENSE_CC_BY_4_0.txt"
    licence.write_bytes(b"SYNTHETIC LICENCE FIXTURE")
    rights = {"redistributable": True, "licence": "CC-BY-4.0", "asset_attribution": "synthetic-test", "dataset_attribution": "synthetic-test",
              "licence_sha256": sha(licence.read_bytes()), "objects": ["abo-test"]}
    (folder / "RIGHTS_AND_ATTRIBUTION.json").write_text(json.dumps(rights))
    metadata = {"source_id": "test", "source_url": "https://amazon-berkeley-objects.s3.amazonaws.com/images/original/unit-test.jpg",
                "original_image": original.relative_to(tmp_path).as_posix(), "original_sha256": sha(original.read_bytes()), "prepared_image_sha256": sha(prepared.read_bytes())}
    prep = folder / "preparation.json"
    prep.write_text(json.dumps(metadata))
    row = {"object_id": "abo-test", "source_image": prepared.relative_to(tmp_path).as_posix(), "image_sha256": sha(prepared.read_bytes()),
           "preparation_metadata": prep.relative_to(tmp_path).as_posix(), "redistributable": "true"}
    with (folder / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return original, prepared, rejected, prep


def test_rights_authorises_raw_and_prepared_but_not_excluded_photo(tmp_path):
    original, prepared, rejected, _ = fixture(tmp_path)
    approved = approved_dataset_assets(tmp_path)
    assert original in approved and prepared in approved
    assert rejected not in approved


def test_rights_original_hash_mismatch_stops_packaging(tmp_path):
    original, _, _, _ = fixture(tmp_path)
    original.write_bytes(b"CHANGED SYNTHETIC PHOTO")
    with pytest.raises(ValueError, match="hash differs"):
        approved_dataset_assets(tmp_path)


def test_rights_rejects_unrelated_asset_origin(tmp_path):
    _, _, _, prep = fixture(tmp_path)
    metadata = json.loads(prep.read_text())
    metadata["source_url"] = "https://unapproved.example/image.jpg"
    prep.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official ABO S3"):
        approved_dataset_assets(tmp_path)
