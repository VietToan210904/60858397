"""CPU-only synthetic freeze fixtures; never real inference or benchmark records."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("freeze_3d_deployment", Path(__file__).resolve().parents[1] / "scripts/freeze_3d_deployment.py")
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def save_records(root, records):
    path = root / "results/reconstruction/records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    for name in freeze.SOURCES:
        write(tmp_path / name, ("# SYNTHETIC UNIT SOURCE ONLY " + name).encode())
    for folder, case in (("reconstruction", "final-unit"), ("reconstruction_smoke", "smoke-unit")):
        base = tmp_path / "datasets" / folder
        image = base / "unit-input.png"
        digest = write(image, b"SYNTHETIC SOURCE BYTE FIXTURE; NO INFERENCE")
        row = {"object_id": case, "source_image": image.relative_to(tmp_path).as_posix(), "image_sha256": digest,
               "category": "synthetic-unit-test", "geometry_eligible": "false", "silhouette_eligible": "false", "render_similarity_eligible": "false"}
        with (base / "objects.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    dino = tmp_path / "models/dino"
    dino_sha = write(dino / "model.safetensors", b"SYNTHETIC DINO BYTES; NOT A MODEL")
    dino_config_sha = write(dino / "config.json", b'{"synthetic_unit_fixture":true}')
    models, executions = {}, {}
    libraries = {name: "SYNTHETIC-UNIT-VERSION" for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "transformers", "omegaconf", "trimesh", "safetensors", "huggingface-hub")}
    for key in freeze.MODEL_KEYS:
        source_revision = ("a" if key == "sf3d" else "b") * 40
        revision = ("c" if key == "sf3d" else "e") * 40
        weights = tmp_path / "models" / key
        config_data = b"synthetic_unit_fixture: true\n"
        config_sha = write(weights / "config.yaml", config_data)
        weights_data = b"SYNTHETIC FINAL WEIGHT BYTE FIXTURE; NOT A MODEL"
        weight_sha = write(weights / "model.safetensors", weights_data)
        blob = hashlib.sha1(b"blob " + str(len(config_data)).encode() + b"\0" + config_data).hexdigest()
        patch_path = tmp_path / "environment" / (key + ".patch")
        patch_sha = write(patch_path, b"SYNTHETIC REVIEWED PATCH FIXTURE")
        receipt = {"source_revision": source_revision, "extensions": [{"returncode": 0, "compatibility_patch": {
            "patch_path": patch_path.relative_to(tmp_path).as_posix(), "patch_sha256": patch_sha}}]}
        receipt_path = tmp_path / "environment" / (key + "-native.json")
        receipt_sha = write(receipt_path, json.dumps(receipt).encode())
        native_artifacts = {}
        for name in ("texture_baker._C", "uv_unwrapper._C"):
            binary = tmp_path / (".venv-" + key) / (name + ".pyd")
            binary_sha = write(binary, ("SYNTHETIC BINARY NOT EXECUTABLE " + name).encode())
            native_artifacts[name] = {"path": str(binary), "sha256": binary_sha}
        clip_path = tmp_path / "models" / (key + "-clip.bin")
        clip_sha = write(clip_path, b"SYNTHETIC CLIP BYTES; NOT A MODEL")
        clip_key = "open_clip" if key == "sf3d" else "alpha_clip"
        interpreter = tmp_path / (".venv-" + key) / "Scripts/python.exe"
        write(interpreter, b"SYNTHETIC INTERPRETER MARKER; NEVER EXECUTE")
        model = {"pair": "reconstruction", "model_id": "synthetic-unit-model-" + key, "revision": revision,
                 "checkout": "vendor/" + key, "source_revision": source_revision,
                 "python": interpreter.relative_to(tmp_path).as_posix(), "weights_path": weights.relative_to(tmp_path).as_posix(),
                 "checkpoint_files": {"config.yaml": {"size_bytes": len(config_data), "git_blob_sha1": blob},
                                      "model.safetensors": {"size_bytes": len(weights_data), "sha256": weight_sha}},
                 "extension_build": {"receipt": "../../environment/" + receipt_path.name, "receipt_sha256": receipt_sha,
                                     "patch_sha256": patch_sha, "binary_sha256": {name: item["sha256"] for name, item in native_artifacts.items()}},
                 "auxiliary": {"dinov2": {"path": "../dino", "revision": "d" * 40, "model_sha256": dino_sha, "config_sha256": dino_config_sha},
                               clip_key: {"path": "../" + clip_path.name, "sha256": clip_sha}}}
        models[key] = model
        acquisition = {"status": "downloaded", "model_id": model["model_id"], "resolved_revision": revision,
            "files": [{"path": "config.yaml", "size_bytes": len(config_data), "expected_git_blob_sha1": blob},
                      {"path": "model.safetensors", "size_bytes": len(weights_data), "expected_sha256": weight_sha}],
            "sha256": {"config.yaml": config_sha, "model.safetensors": weight_sha}, "fixture_note": "SYNTHETIC UNIT TEST ONLY"}
        write(tmp_path / "environment" / ("acquisition_" + key + ".json"), json.dumps(acquisition).encode())
        paths = ["image_tokenizer.pretrained_model_name_or_path"]
        if key == "spar3d": paths.append("pdiff_image_tokenizer.pretrained_model_name_or_path")
        executions[key] = {"python": str(interpreter), "libraries": libraries, "torch_version": libraries["torch"],
            "device": "SYNTHETIC UNIT DEVICE", "cuda_version": "SYNTHETIC UNIT VALUE", "seed_policy": "UNIT RESET POLICY",
            "source_revision": source_revision, "source_dirty": False, "resolved_checkpoint_revision": revision,
            "adapter_source_sha256": freeze.sha256_file(tmp_path / freeze.SOURCES[0]),
            "runner_source_sha256": freeze.sha256_file(tmp_path / freeze.SOURCES[1]),
            "checkpoint_provenance": {"revision": revision, "files": {"config.yaml": {"size_bytes": len(config_data), "sha256": config_sha},
                "model.safetensors": {"size_bytes": len(weights_data), "sha256": weight_sha}}},
            "native_extension_provenance": {"receipt_sha256": receipt_sha, "patch_sha256": patch_sha, "artifacts": native_artifacts},
            "auxiliary_provenance": {"dinov2": {"revision": "d" * 40, "weights_sha256": dino_sha, "config_sha256": dino_config_sha, "configuration_paths": paths},
                                     clip_key: {"sha256": clip_sha}}}
    full = {"models": models, "seed": 8410, "datasets": {"reconstruction": {"object_count": 1}},
            "evaluation": {"warmup_runs": 1, "reconstruction": {"dataset_dir": "datasets/reconstruction", "render_views": False}}}
    smoke = copy.deepcopy(full); smoke["smoke_only"] = True; smoke["evaluation"]["reconstruction"]["dataset_dir"] = "datasets/reconstruction_smoke"
    protocol = freeze.pair_protocol_sha256(smoke, "reconstruction", freeze.dataset_fingerprints(tmp_path, smoke))
    case = freeze.dataset(tmp_path, smoke)["cases"][0]
    records = []
    for key in freeze.MODEL_KEYS:
        base = tmp_path / "results/reconstruction/unit-run" / key
        base.mkdir(parents=True)
        mesh = base / "mesh.glb"; trimesh.creation.box().export(mesh)
        descriptor_path = base / "input_descriptor.json"
        descriptor_path.write_text(json.dumps(case["descriptor"]))
        from stylesphere.reconstruction.metrics import validate_glb
        artifacts = {"mesh": mesh.relative_to(tmp_path).as_posix(), "input_descriptor": descriptor_path.relative_to(tmp_path).as_posix()}
        records.append({"pair": "reconstruction", "model": key, "case_id": case["case_id"], "seed": 8410,
            "mode": "smoke", "status": "success", "provenance": "model_inference", "fixture_note": "SYNTHETIC UNIT TEST ONLY",
            "run_id": "synthetic-unit-run", "protocol_sha256": protocol, "config_sha256": freeze.canonical_hash(freeze.measurement_config(tmp_path, smoke, key)),
            "input_sha256": case["input_sha256"], "checkpoint_revision": models[key]["revision"], "execution": executions[key],
            "metrics": validate_glb(mesh), "artifacts": artifacts,
            "artifact_sha256": {name: freeze.sha256_file(tmp_path / value) for name, value in artifacts.items()}})
    save_records(tmp_path, records)
    monkeypatch.setattr(freeze, "git_state", lambda checkout: (models[checkout.name]["source_revision"], False))
    monkeypatch.setattr(freeze, "environment_metadata", lambda python, names: {"python": str(python), "python_version": "SYNTHETIC UNIT PYTHON", "libraries": libraries})
    return tmp_path, full, smoke, records


def test_freeze_is_complete_idempotent_and_preserves_original_records(fixture):
    root, full, smoke, records = fixture
    raw = (root / "results/reconstruction/records.jsonl").read_bytes()
    receipt = freeze.freeze_deployment(root, full, smoke)
    assert receipt["expected_final_executions"] == 2 and set(receipt["smoke_bindings"]) == set(freeze.MODEL_KEYS)
    assert receipt["model_inference_performed_by_freeze"] is False
    snapshot = root / receipt["source_snapshot"]
    for name, expected in receipt["source_sha256"].items():
        assert freeze.sha256_file(snapshot / name) == expected
    target = root / "environment/reconstruction-deployment-freeze.json"
    original_receipt = target.read_bytes()
    assert freeze.freeze_deployment(root, full, smoke)["freeze_id"] == receipt["freeze_id"]
    assert target.read_bytes() == original_receipt and (root / "results/reconstruction/records.jsonl").read_bytes() == raw


def test_freeze_refuses_missing_second_success_before_writing(fixture):
    root, full, smoke, records = fixture
    records[1]["status"] = "failed"; save_records(root, records)
    with pytest.raises(ValueError, match="Missing matching successful smoke"):
        freeze.freeze_deployment(root, full, smoke)
    assert not (root / "environment/reconstruction-deployment-freeze.json").exists()


def test_freeze_refuses_wrapper_drift_since_smoke(fixture):
    root, full, smoke, _ = fixture
    write(root / freeze.SOURCES[0], b"CHANGED SYNTHETIC WORKER")
    with pytest.raises(ValueError, match="Wrapper source changed"):
        freeze.freeze_deployment(root, full, smoke)


@pytest.mark.parametrize("kind", ["checkpoint", "binary", "artifact"])
def test_freeze_rechecks_checkpoint_native_and_output_bytes(fixture, kind):
    root, full, smoke, records = fixture
    paths = {"checkpoint": root / "models/sf3d/model.safetensors",
             "binary": Path(records[0]["execution"]["native_extension_provenance"]["artifacts"]["texture_baker._C"]["path"]),
             "artifact": root / records[0]["artifacts"]["mesh"]}
    path = paths[kind]; path.write_bytes(b"X" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze.freeze_deployment(root, full, smoke)


def test_saved_descriptor_must_match_semantic_input_anchor(fixture):
    root, full, smoke, records = fixture
    path = root / records[0]["artifacts"]["input_descriptor"]
    changed = json.loads(path.read_text()); changed["manifest"]["object_id"] = "CHANGED-SYNTHETIC-INPUT"
    path.write_text(json.dumps(changed)); records[0]["artifact_sha256"]["input_descriptor"] = freeze.sha256_file(path)
    save_records(root, records)
    with pytest.raises(ValueError, match="Saved smoke input descriptor differs"):
        freeze.freeze_deployment(root, full, smoke)


def test_freeze_refuses_environment_drift(fixture, monkeypatch):
    root, full, smoke, records = fixture
    libraries = dict(records[0]["execution"]["libraries"]); libraries["numpy"] = "CHANGED UNIT VERSION"
    monkeypatch.setattr(freeze, "environment_metadata", lambda python, names: {"python": str(python), "libraries": libraries})
    with pytest.raises(ValueError, match="environment versions changed"):
        freeze.freeze_deployment(root, full, smoke)


def test_freeze_binds_official_acquisition_receipt_revision(fixture):
    root, full, smoke, _ = fixture
    path = root / "environment/acquisition_sf3d.json"
    acquisition = json.loads(path.read_text()); acquisition["resolved_revision"] = "CHANGED SYNTHETIC REVISION"
    path.write_text(json.dumps(acquisition))
    with pytest.raises(ValueError, match="acquisition receipt differs"):
        freeze.freeze_deployment(root, full, smoke)


def test_initial_freeze_cannot_be_retroactive_but_existing_freeze_stays_immutable(fixture):
    root, full, smoke, records = fixture
    receipt = freeze.freeze_deployment(root, full, smoke)
    final = dict(records[0], mode="evaluation", protocol_sha256=receipt["protocol_sha256"])
    save_records(root, records + [final])
    assert freeze.freeze_deployment(root, full, smoke)["freeze_id"] == receipt["freeze_id"]
    with pytest.raises(ValueError, match="after final evaluation has begun"):
        freeze.freeze_deployment(root, full, smoke, output="environment/new-retrospective-freeze.json")


def test_existing_source_snapshot_is_never_silently_replaced(fixture):
    root, full, smoke, _ = fixture
    receipt = freeze.freeze_deployment(root, full, smoke)
    path = root / receipt["source_snapshot"] / freeze.SOURCES[0]
    write(path, b"CHANGED SYNTHETIC SNAPSHOT")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze.freeze_deployment(root, full, smoke)
    assert path.read_bytes() == b"CHANGED SYNTHETIC SNAPSHOT"
