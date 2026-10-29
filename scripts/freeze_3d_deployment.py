"""Freeze a reviewed 3D deployment after both genuine development smokes succeed.

Only local files, Git metadata and distribution metadata are inspected. No model,
CUDA operation, network request or authentication store is accessed. The first
freeze must precede final evaluation. Existing receipts/snapshots are immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylesphere.common.config import dataset_fingerprints, load_config, pair_protocol_sha256
from stylesphere.common.io import canonical_hash, portable_path, read_jsonl, sha256_file, utc_now
from stylesphere.reconstruction.metrics import verify_record

MODEL_KEYS = ("sf3d", "spar3d")
SOURCES = (
    "src/stylesphere/reconstruction/worker.py", "src/stylesphere/reconstruction/runner.py",
    "src/stylesphere/reconstruction/metrics.py", "src/stylesphere/reconstruction/rendering.py",
    "src/stylesphere/generation/runner.py", "src/stylesphere/common/config.py",
    "src/stylesphere/common/io.py", "src/stylesphere/common/runtime.py", "src/stylesphere/cli.py",
)


def local(root: Path, value) -> Path:
    path = portable_path(root, value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Freeze references must remain inside the project root")
    return path


def reference(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def checked_file(path: Path, *, sha256=None, git_blob_sha1=None, size_bytes=None) -> dict:
    if not path.is_file() or size_bytes is not None and path.stat().st_size != size_bytes:
        raise ValueError(f"Missing or changed file size: {path.name}")
    digest = sha256_file(path)
    if sha256 and digest != sha256:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    if git_blob_sha1:
        data = path.read_bytes()
        if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != git_blob_sha1:
            raise ValueError(f"Git blob mismatch: {path.name}")
    return {"sha256": digest, "bytes": path.stat().st_size}


def git_state(checkout: Path) -> tuple[str, bool]:
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    return revision, dirty


def environment_metadata(python: Path, names: list[str]) -> dict:
    # This child imports stdlib metadata only, never Torch or either model.
    code = """import json,sys
from importlib import metadata
versions={}
for name in json.loads(sys.argv[1]):
 try: versions[name]=metadata.version(name)
 except metadata.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':sys.executable,'python_version':sys.version,'libraries':versions}))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([str(python), "-c", code, json.dumps(names)], capture_output=True,
                            text=True, timeout=30, check=True, env=env)
    return json.loads(result.stdout)


def configuration(config: dict) -> dict:
    return {"models": {key: config["models"][key] for key in MODEL_KEYS}, "seed": config["seed"],
            "evaluation": config.get("evaluation", {}).get("reconstruction", {}),
            "warmup_runs": config.get("evaluation", {}).get("warmup_runs", 1)}


def measurement_config(root: Path, config: dict, key: str) -> dict:
    model = dict(config["models"][key])
    for field in ("checkout", "weights_path"):
        if model.get(field):
            model[field] = str(local(root, model[field]))
    if model.get("python") and any(separator in model["python"] for separator in ("/", "\\")):
        model["python"] = str(local(root, model["python"]))
    return {**model, "evaluation": config.get("evaluation", {}).get("reconstruction", {}),
            "warmup_runs": config.get("evaluation", {}).get("warmup_runs", 1)}


def descriptor(root: Path, row: dict) -> dict:
    result = {"manifest": row, "file_sha256": {}}
    for field in ("source_image", "reference_mesh", "camera_info", "preparation_metadata"):
        value = row.get(field)
        if value and not value.lstrip().startswith("{"):
            result["file_sha256"][field] = checked_file(local(root, value))["sha256"]
    for field, expected in (("source_image", "image_sha256"), ("reference_mesh", "mesh_sha256")):
        if row.get(field) and row.get(expected) and result["file_sha256"].get(field) != row[expected]:
            raise ValueError(f"Frozen input hash differs: {row['object_id']}/{field}")
    value = row.get("camera_info", "")
    cameras = json.loads(value) if value.lstrip().startswith("{") else json.loads(local(root, value).read_text(encoding="utf-8")) if value else {}
    for index, camera in enumerate(cameras.get("views", [])):
        for field in ("reference_rgb", "reference_silhouette"):
            if camera.get(field):
                result["file_sha256"][f"view_{index}_{field}"] = checked_file(local(root, camera[field]))["sha256"]
    return result


def dataset(root: Path, config: dict) -> dict:
    folder = local(root, config.get("evaluation", {}).get("reconstruction", {}).get("dataset_dir", "datasets/reconstruction"))
    with (folder / "objects.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["object_id"] for row in rows}) != len(rows):
        raise ValueError("Freeze requires nonempty, unique reconstruction object IDs")
    documents = {reference(root, path): checked_file(path) for path in sorted(folder.iterdir())
                 if path.is_file() and path.suffix in {".csv", ".json", ".jsonl", ".txt", ".md"}}
    manifest = folder / "manifest.json"
    if manifest.exists():
        for name, item in json.loads(manifest.read_text(encoding="utf-8"))["inputs"].items():
            checked_file(local(root, str(folder / name)), sha256=item["sha256"], size_bytes=item["bytes"])
    originals = {}
    cases = []
    for row in rows:
        item = descriptor(root, row)
        cases.append({"case_id": row["object_id"], "descriptor": item, "input_sha256": canonical_hash(item)})
        if row.get("preparation_metadata"):
            value = row["preparation_metadata"]
            prep = json.loads(value) if value.lstrip().startswith("{") else json.loads(local(root, value).read_text(encoding="utf-8"))
            original = prep.get("original_image", prep.get("original_source_image"))
            if original:
                originals[reference(root, local(root, original))] = checked_file(local(root, original), sha256=prep["original_sha256"])
    return {"directory": reference(root, folder), "documents": documents, "original_assets": originals, "cases": cases}


def model_assets(root: Path, key: str, model: dict, execution: dict) -> tuple[dict, set[Path]]:
    weights = local(root, model["weights_path"])
    expected = model["checkpoint_files"]
    if set(expected) != {"config.yaml", "model.safetensors"}:
        raise ValueError("Both pinned official main checkpoint identities are required")
    checkpoint = {}
    for name, item in expected.items():
        if not item.get("sha256") and not item.get("git_blob_sha1"):
            raise ValueError("Checkpoint files need an official SHA256 or Git blob identity")
        checkpoint[name] = checked_file(weights / name, **item)
    acquisition_path = root / "environment" / f"acquisition_{key}.json"
    acquisition_file = checked_file(acquisition_path)
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if (acquisition.get("status") != "downloaded" or acquisition.get("model_id") != model["model_id"]
            or acquisition.get("resolved_revision") != model["revision"]):
        raise ValueError("Main checkpoint acquisition receipt differs from the pinned deployment")
    official_files = {item["path"]: item for item in acquisition.get("files", [])}
    for name, item in expected.items():
        official = official_files.get(name, {})
        if (official.get("size_bytes") != item["size_bytes"]
                or item.get("sha256") and official.get("expected_sha256") != item["sha256"]
                or item.get("git_blob_sha1") and official.get("expected_git_blob_sha1") != item["git_blob_sha1"]
                or acquisition.get("sha256", {}).get(name) != checkpoint[name]["sha256"]):
            raise ValueError("Main acquisition receipt does not bind the exact official checkpoint bytes")
    actual_checkpoint = execution.get("checkpoint_provenance", {})
    if actual_checkpoint.get("revision") != model["revision"]:
        raise ValueError("Smoke checkpoint provenance does not match the pinned revision")
    for name, item in checkpoint.items():
        if actual_checkpoint.get("files", {}).get(name) != {"size_bytes": item["bytes"], "sha256": item["sha256"]}:
            raise ValueError("Smoke checkpoint provenance differs from verified local bytes")
    revision, dirty = git_state(local(root, model["checkout"]))
    if dirty or revision != model["source_revision"] or execution.get("source_revision") != revision or execution.get("source_dirty") is not False:
        raise ValueError("Official model source must be clean and match the actual smoke commit")
    build = model["extension_build"]
    receipt_path = local(root, weights / build["receipt"])
    checked_file(receipt_path, sha256=build["receipt_sha256"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("source_revision") != revision or not receipt.get("extensions") or any(item.get("returncode") != 0 for item in receipt["extensions"]):
        raise ValueError("Native build receipt must record successful builds against the official commit")
    copied = {receipt_path, acquisition_path}
    patches = [item["compatibility_patch"] for item in receipt["extensions"] if item.get("compatibility_patch")]
    if not patches:
        raise ValueError("The reviewed native compatibility patch must be recorded")
    for patch in patches:
        if patch["patch_sha256"] != build["patch_sha256"]:
            raise ValueError("Native compatibility patch identity differs")
        path = local(root, patch["patch_path"])
        checked_file(path, sha256=build["patch_sha256"])
        copied.add(path)
    native = execution.get("native_extension_provenance", {})
    if native.get("receipt_sha256") != build["receipt_sha256"] or native.get("patch_sha256") != build["patch_sha256"]:
        raise ValueError("Smoke native-build provenance differs from the selected deployment")
    for name, expected_sha in build["binary_sha256"].items():
        artifact = native.get("artifacts", {}).get(name, {})
        checked_file(local(root, artifact["path"]), sha256=expected_sha)
        if artifact.get("sha256") != expected_sha:
            raise ValueError("Smoke native binary identity differs")
    auxiliary = model["auxiliary"]
    dino = auxiliary["dinov2"]
    auxiliary_files = {}
    for name, path, sha in (("dinov2_model", weights / dino["path"] / "model.safetensors", dino["model_sha256"]),
                            ("dinov2_config", weights / dino["path"] / "config.json", dino["config_sha256"])):
        auxiliary_files[name] = checked_file(local(root, path), sha256=sha)
    clip = auxiliary.get("open_clip", auxiliary.get("alpha_clip"))
    auxiliary_files["clip_initializer"] = checked_file(local(root, weights / clip["path"]), sha256=clip["sha256"])
    evidence = execution.get("auxiliary_provenance", {})
    if not evidence:
        raise ValueError("Smoke must retain actual offline-initializer provenance")
    actual_dino = evidence.get("dinov2", {})
    clip_key = "open_clip" if "open_clip" in auxiliary else "alpha_clip"
    if (actual_dino.get("weights_sha256") != dino["model_sha256"]
            or actual_dino.get("config_sha256") != dino["config_sha256"]
            or actual_dino.get("revision") != dino["revision"]
            or evidence.get(clip_key, {}).get("sha256") != clip["sha256"]):
        raise ValueError("Smoke initializer provenance differs from the verified local bytes")
    required_paths = {"image_tokenizer.pretrained_model_name_or_path"}
    if clip_key == "alpha_clip":
        required_paths.add("pdiff_image_tokenizer.pretrained_model_name_or_path")
    if set(actual_dino.get("configuration_paths", [])) != required_paths:
        raise ValueError("Smoke must record every redirected DINO initializer")
    return {"checkpoint": checkpoint, "acquisition_receipt": {"path": reference(root, acquisition_path), **acquisition_file},
            "source_revision": revision, "source_dirty": False,
            "native_build": native, "auxiliary_files": auxiliary_files,
            "smoke_auxiliary_provenance": execution["auxiliary_provenance"]}, copied


def freeze_deployment(root: Path, config: dict, smoke_config: dict, *, output="environment/reconstruction-deployment-freeze.json",
                      records_path="results/reconstruction/records.jsonl") -> dict:
    root = Path(root).resolve()
    target = local(root, output)
    previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    raw = read_jsonl(local(root, records_path))
    fingerprints = dataset_fingerprints(root, config)
    full_protocol = pair_protocol_sha256(config, "reconstruction", fingerprints)
    smoke_protocol = pair_protocol_sha256(smoke_config, "reconstruction", dataset_fingerprints(root, smoke_config))
    if full_protocol == smoke_protocol or not smoke_config.get("smoke_only"):
        raise ValueError("Freeze requires a separate smoke-only development protocol")
    if not target.exists() and any(row.get("pair") == "reconstruction" and row.get("mode") == "evaluation"
                                   and row.get("protocol_sha256") == full_protocol and row.get("status") in {"success", "failed"} for row in raw):
        raise ValueError("Cannot create a first final freeze after final evaluation has begun")
    sources = {name: checked_file(local(root, name))["sha256"] for name in SOURCES}
    full_data, smoke_data = dataset(root, config), dataset(root, smoke_config)
    expected_count = config.get("datasets", {}).get("reconstruction", {}).get("object_count", 20)
    if len(full_data["cases"]) != expected_count:
        raise ValueError("Full reconstruction case count differs from the planned object_count")
    first_smoke = smoke_data["cases"][0]
    if first_smoke["case_id"] in {item["case_id"] for item in full_data["cases"]}:
        raise ValueError("The effective development smoke case overlaps the final input set")
    bindings, assets, environments, extra_sources = {}, {}, {}, set()
    for key in MODEL_KEYS:
        expected_config_sha = canonical_hash(measurement_config(root, smoke_config, key))
        candidates = [row for row in raw if row.get("pair") == "reconstruction" and row.get("model") == key
                      and row.get("status") == "success" and row.get("mode") == "smoke"
                      and row.get("provenance") == "model_inference" and not row.get("synthetic", False)
                      and row.get("protocol_sha256") == smoke_protocol and row.get("config_sha256") == expected_config_sha
                      and row.get("case_id") == first_smoke["case_id"] and row.get("seed") == smoke_config["seed"]
                      and row.get("input_sha256") == first_smoke["input_sha256"]]
        if not candidates:
            raise ValueError(f"Missing matching successful smoke evidence: {key}")
        previous_hash = (previous or {}).get("smoke_bindings", {}).get(key, {}).get("record_sha256")
        if previous_hash:
            candidates = [row for row in candidates if canonical_hash(row) == previous_hash]
            if not candidates:
                raise ValueError("The original frozen smoke record is no longer present")
        row = candidates[-1]
        execution = row.get("execution", {})
        for field, source in (("adapter_source_sha256", SOURCES[0]), ("runner_source_sha256", SOURCES[1])):
            if execution.get(field) != sources[source]:
                raise ValueError("Wrapper source changed since the successful smoke")
        if not row.get("artifacts", {}).get("mesh") or set(row["artifacts"]) != set(row.get("artifact_sha256", {})):
            raise ValueError("Successful smoke needs a mesh and complete artifact hashes")
        for name, value in row["artifacts"].items():
            checked_file(local(root, value), sha256=row["artifact_sha256"][name])
        if row.get("artifacts", {}).get("input_descriptor"):
            saved = json.loads(local(root, row["artifacts"]["input_descriptor"]).read_text(encoding="utf-8"))
            if saved != first_smoke["descriptor"]:
                raise ValueError("Saved smoke input descriptor differs from frozen source inputs")
        errors = verify_record(row, root)
        if errors or row.get("metrics", {}).get("glb_export_success") != 1:
            raise ValueError("Smoke mesh/metrics cannot be verified: " + "; ".join(errors))
        model = config["models"][key]
        if model != smoke_config["models"][key]:
            raise ValueError("Full and smoke model deployments must be identical")
        if row.get("checkpoint_revision") != model["revision"] or execution.get("resolved_checkpoint_revision") != model["revision"]:
            raise ValueError("Smoke checkpoint revision differs from the pinned deployment")
        assets[key], copied = model_assets(root, key, model, execution)
        extra_sources.update(copied)
        libraries = execution.get("libraries", {})
        required = {"torch", "torchvision", "numpy", "Pillow", "transformers", "omegaconf", "trimesh", "safetensors", "huggingface-hub"}
        if not required.issubset(libraries) or any(libraries[name] is None for name in required) or not execution.get("seed_policy") or not execution.get("device") or not execution.get("cuda_version"):
            raise ValueError("Smoke must retain actual library/CUDA/device/seed-policy evidence")
        python = local(root, model["python"])
        if local(root, execution.get("python", "missing")) != python:
            raise ValueError("Smoke used a different model interpreter")
        environment = environment_metadata(python, sorted(libraries))
        if environment["libraries"] != libraries or execution.get("torch_version") != libraries["torch"]:
            raise ValueError("Model environment versions changed since the smoke")
        environment["python"] = reference(root, python)
        environments[key] = environment
        bindings[key] = {"record_sha256": canonical_hash(row), "record": row,
                         "full_measurement_config_sha256": canonical_hash(measurement_config(root, config, key))}
    copied_paths = {local(root, name) for name in SOURCES} | extra_sources
    copied_paths |= {local(root, name) for name in full_data["documents"]} | {local(root, name) for name in smoke_data["documents"]}
    copy_hashes = {reference(root, path): checked_file(path)["sha256"] for path in sorted(copied_paths)}
    if any(copy_hashes[name] != digest for name, digest in sources.items()):
        raise ValueError("Wrapper source changed while constructing the freeze")
    if (full_protocol != pair_protocol_sha256(config, "reconstruction", dataset_fingerprints(root, config))
            or smoke_protocol != pair_protocol_sha256(smoke_config, "reconstruction", dataset_fingerprints(root, smoke_config))):
        raise ValueError("Dataset documents changed while constructing the freeze")
    core = {"schema_version": 1, "kind": "pre_final_reconstruction_deployment_freeze", "source_sha256": sources,
            "protocol_sha256": full_protocol, "smoke_protocol_sha256": smoke_protocol,
            "deployment_protocol": configuration(config), "smoke_protocol": configuration(smoke_config),
            "full_inputs": full_data, "smoke_inputs": smoke_data, "smoke_bindings": bindings,
            "model_assets": assets, "environments": environments, "snapshot_file_sha256": copy_hashes,
            "expected_final_executions": expected_count * len(MODEL_KEYS),
            "model_inference_performed_by_freeze": False, "network_access_performed": False,
            "note": "Successful development smoke evidence is bound to exact source/config/input/runtime bytes. Human quality and missing reference-geometry accuracy are not supplied by this freeze."}
    freeze_id = canonical_hash(core)
    if target.exists():
        if previous.get("freeze_id") != freeze_id or any(previous.get(name) != value for name, value in core.items()):
            raise ValueError("Existing reconstruction freeze differs; refuse overwrite. Preserve it and use a new version/output path")
        snapshot = local(root, previous["source_snapshot"])
        for name, expected in copy_hashes.items():
            checked_file(snapshot / name, sha256=expected)
        return previous
    snapshot = root / "environment" / "reconstruction_source_snapshot" / freeze_id[:16]
    for name, expected in copy_hashes.items():
        destination = snapshot / name
        if destination.exists():
            checked_file(destination, sha256=expected)
        else:
            data = local(root, name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Source/document bytes changed while copying the snapshot")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
    receipt = {**core, "freeze_id": freeze_id, "frozen_at_utc": utc_now(),
               "source_snapshot": reference(root, snapshot), "raw_records_path": reference(root, local(root, records_path)),
               "raw_records_sha256_at_freeze": sha256_file(local(root, records_path))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="config/reconstruction-local.yaml")
    parser.add_argument("--smoke-config", default="config/reconstruction_smoke.yaml")
    parser.add_argument("--output", default="environment/reconstruction-deployment-freeze.json")
    parser.add_argument("--records", default="results/reconstruction/records.jsonl")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = freeze_deployment(root, load_config(root, args.config), load_config(root, args.smoke_config),
                                output=args.output, records_path=args.records)
    print(json.dumps({"freeze_id": receipt["freeze_id"], "protocol_sha256": receipt["protocol_sha256"],
                      "expected_final_executions": receipt["expected_final_executions"],
                      "source_snapshot": receipt["source_snapshot"], "model_inference_performed": False}, indent=2))


if __name__ == "__main__":
    main()
