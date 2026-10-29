"""Synthetic local geometry/images verify portable retained paths without OpenGL."""
import json
import csv
import hashlib
from pathlib import Path

import pytest

from stylesphere.reconstruction import runner
from stylesphere.reconstruction.metrics import verify_record
from stylesphere.reconstruction.worker import artifact_reference


def test_artifact_reference_is_root_relative_and_rejects_outside(tmp_path):
    output = tmp_path / "results/mesh.glb"
    assert artifact_reference(tmp_path, output) == "results/mesh.glb"
    with pytest.raises(ValueError, match="within the project root"):
        artifact_reference(tmp_path, tmp_path.parent / "outside.glb")


def test_nested_render_paths_survive_changed_working_directory(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image
    mesh_path = tmp_path / "results/synthetic-unit-box.glb"
    mesh_path.parent.mkdir()
    trimesh.creation.box().export(mesh_path)
    reference = tmp_path / "datasets/synthetic-reference.png"
    reference.parent.mkdir()
    Image.new("RGB", (8, 8), "white").save(reference)
    def fake_render(mesh, cameras, directory):
        directory.mkdir(parents=True)
        image, silhouette = directory / "image.png", directory / "mask.png"
        Image.new("RGB", (8, 8), "white").save(image)
        Image.new("L", (8, 8), 255).save(silhouette)
        return [{"view_id": "synthetic-unit-view", "image": str(image), "silhouette": str(silhouette),
                 "camera": {"reference_rgb": str(reference)}}]
    monkeypatch.setattr(runner, "render_views", fake_render)
    metrics, artifacts, skipped = runner.evaluate_mesh(tmp_path, {"geometry_eligible": "false"}, mesh_path,
                                                       mesh_path.parent / "evaluation", {}, 8410)
    assert all(not Path(value).is_absolute() and "\\" not in value for value in artifacts.values())
    manifest = json.loads((tmp_path / artifacts["renders_manifest"]).read_text())
    rendered = manifest["renders"][0]
    assert rendered["camera"]["reference_rgb"] == "datasets/synthetic-reference.png"
    assert str(tmp_path) not in json.dumps(manifest)
    elsewhere = tmp_path / "another-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    record = {"status": "success", "metrics": metrics, "artifacts": {"mesh": "results/synthetic-unit-box.glb", **artifacts}}
    assert verify_record(record, root=tmp_path) == []
    record["artifacts"]["mesh"] = record["artifacts"]["mesh"].replace("/", "\\")
    assert verify_record(record, root=tmp_path) == []


def test_invalid_mesh_export_remains_hashed_portable_evidence(tmp_path, monkeypatch):
    folder = tmp_path / "datasets/reconstruction"
    folder.mkdir(parents=True)
    with (folder / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object_id", "source_image"])
        writer.writeheader()
        writer.writerow({"object_id": "synthetic-unit-case", "source_image": "datasets/synthetic-input.png"})
    def fake_worker(worker, request, directory, python, timeout, working_dir):
        assert working_dir == tmp_path
        mesh = Path(request["cases"][0]["mesh"])
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"SYNTHETIC INVALID MESH UNIT FIXTURE")
        return {"results": [{"case_id": "synthetic-unit-case", "status": "success", "metrics": {},
                             "artifacts": {"mesh": mesh.relative_to(tmp_path).as_posix()}}]}
    monkeypatch.setattr(runner, "invoke_worker", fake_worker)
    outcome = runner.run(tmp_path, {"models": {"sf3d": {"revision": "a" * 40}}}, tmp_path / "results", 8410, model_keys=["sf3d"])
    assert outcome["failed"] == 1 and outcome["success"] == 0
    record = json.loads((tmp_path / "results/reconstruction/records.jsonl").read_text())
    retained = tmp_path / record["artifacts"]["mesh"]
    assert not Path(record["artifacts"]["mesh"]).is_absolute()
    assert record["artifact_sha256"]["mesh"] == hashlib.sha256(retained.read_bytes()).hexdigest()
    from stylesphere.generation.runner import digest
    descriptor = json.loads((tmp_path / record["artifacts"]["input_descriptor"]).read_text())
    assert digest(descriptor) == record["input_sha256"]
    assert descriptor["manifest"]["object_id"] == record["case_id"]
    assert descriptor["file_sha256"]["source_image"] == "missing"
