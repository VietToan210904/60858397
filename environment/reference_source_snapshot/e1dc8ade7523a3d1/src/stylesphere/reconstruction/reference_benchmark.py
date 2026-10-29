"""Separate, predeclared normalized-shape benchmark on real scanned references.

Inference uses the unchanged official adapters. Reference-assisted rigid ICP is
explicitly supplementary; it is never described as absolute reconstruction.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
from scipy.spatial import cKDTree

from stylesphere.common.config import dataset_fingerprints, load_config, pair_protocol_sha256
from stylesphere.common.io import canonical_hash, sha256_file
from stylesphere.generation.runner import digest
from .metrics import geometry_metrics, load_mesh, sample_surface, silhouette_iou, validate_glb

PROTOCOL = "config/reconstruction_reference_protocol.json"
FREEZE = "environment/reference_benchmark_freeze.json"
AUTHORIZATION = "environment/reference_benchmark_test_authorization.json"
SPLITS = {"development": "datasets/reconstruction_reference/development", "test": "datasets/reconstruction_reference"}
MODELS = ("sf3d", "spar3d")
NEW_SOURCES = (
    "src/stylesphere/reconstruction/reference_benchmark.py", "scripts/acquire_gso_reference.py",
    "scripts/acquire_omni_reference.py", "scripts/prepare_gso_reference.py",
    "config/reconstruction_reference_protocol.json", "config/reconstruction_reference_development.yaml",
    "config/reconstruction_reference_test.yaml", "docs/REFERENCE_RECONSTRUCTION.md",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(content)


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def config(root, split):
    return load_config(root, Path(f"config/reconstruction_reference_{split}.yaml"))


def result_dir(root, split):
    return root / f"results/reference_{split}"


def normalized_mesh(mesh):
    bounds = np.asarray(mesh.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Invalid normalization bounds")
    transform = np.eye(4)
    transform[:3, :3] /= diagonal
    transform[:3, 3] = -bounds.mean(axis=0) / diagonal
    result = mesh.copy()
    result.apply_transform(transform)
    return result, transform


def proper_rotations():
    """Exactly24 orientation-preserving cube rotations, deterministic order."""
    values = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                values.append(matrix)
    return sorted(values, key=lambda x: tuple(x.ravel()))


def apply(points, transform):
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def rigid_kabsch(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, b - rotation @ a
    return result


def chamfer_loss(a, b):
    x, y = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    return float(0.5 * (np.mean(x*x) + np.mean(y*y)))


def rigid_icp(predicted, reference, max_iterations=50, tolerance=1e-7):
    """Fixed multistart registration; no scale fit, trimming or reflections."""
    tree, trials = cKDTree(reference), []
    best_loss, best_transform, chosen = float("inf"), None, None
    for index, rotation in enumerate(proper_rotations()):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        current = apply(predicted, transform)
        previous = float("inf")
        history = []
        for iteration in range(max_iterations):
            distance, nearest = tree.query(current)
            update = rigid_kabsch(current, reference[nearest])
            current = apply(current, update)
            transform = update @ transform
            loss = float(np.mean(tree.query(current)[0]**2))
            history.append(loss)
            if abs(previous - loss) <= tolerance:
                break
            previous = loss
        loss = chamfer_loss(current, reference)
        trials.append({"trial": index, "initial_rotation": rotation.tolist(), "iterations": len(history),
                       "predicted_to_reference_mse_history": history, "symmetric_registration_chamfer_squared": loss,
                       "transform": transform.tolist()})
        if loss < best_loss:
            best_loss, best_transform, chosen = loss, transform, index
    return best_transform, {"chosen_trial": chosen, "chosen_loss": best_loss, "trials": trials}


def freeze(root):
    if (root / FREEZE).exists():
        check_freeze(root)
        return {"status": "verified_existing_freeze", "freeze_id": read(root / FREEZE)["freeze_id"]}
    if any((result_dir(root, s) / "reconstruction/records.jsonl").exists() for s in SPLITS):
        raise ValueError("First design freeze must precede all reference model outputs")
    original_path = root / "environment/reconstruction-deployment-freeze.json"
    original = read(original_path)
    sources = dict(original["source_sha256"])
    for path, expected in sources.items():
        if sha256_file(root / path) != expected:
            raise ValueError(f"Original frozen inference source changed: {path}")
    sources.update({p: sha256_file(root / p) for p in NEW_SOURCES})
    dataset = root / "datasets/reconstruction_reference"
    prepared = read(dataset / "preparation_manifest.json")
    assets = dict(prepared["asset_sha256"])
    # Include subsequent metadata-only source/access records before any inference.
    assets.update({relative(root, p): sha256_file(p) for p in dataset.iterdir() if p.is_file()})
    for path, expected in assets.items():
        if sha256_file(root / path) != expected:
            raise ValueError(f"Prepared dataset hash differs: {path}")
    configs = {}
    for split in SPLITS:
        c = config(root, split)
        configs[split] = {"pair_protocol_sha256": pair_protocol_sha256(c, "reconstruction", dataset_fingerprints(root, c)),
                          "models": {m: c["models"][m] for m in MODELS},
                          "evaluation": c["evaluation"]["reconstruction"], "seed": c["seed"], "warmup_runs": c["evaluation"]["warmup_runs"]}
    value = {"kind": "controlled_reference_pre_development_design_freeze", "created_at_utc": now(),
             "source_sha256": sources, "asset_sha256": assets, "configs": configs,
             "metric_protocol": read(root / PROTOCOL), "metric_protocol_sha256": sha256_file(root / PROTOCOL),
             "original_3d_deployment_freeze_sha256": sha256_file(original_path),
             "original_3d_deployment_freeze_id": original["freeze_id"],
             "checkpoint_and_build_identities": original["model_assets"], "expected_environments": original["environments"],
             "expected_execution_bindings": {m: {k: original["smoke_bindings"][m]["record"]["execution"][k]
                for k in ("device", "source_revision", "source_dirty", "resolved_checkpoint_revision", "checkpoint_provenance", "native_extension_provenance", "auxiliary_provenance")}
                for m in MODELS},
             "main_checkpoint_bytes_rehashed_here": False,
             "note": "Existing checkpoint bytes were verified by prior deployment acquisition and are rechecked by unchanged inference worker; this receipt binds prior exact identities."}
    value["freeze_id"] = canonical_hash(value)
    snapshot = root / "environment/reference_source_snapshot" / value["freeze_id"][:16]
    for path in sources:
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    value["source_snapshot"] = relative(root, snapshot)
    write(root / FREEZE, value, exclusive=True)
    return {"freeze_id": value["freeze_id"], "created_at_utc": value["created_at_utc"], "bound_assets": len(assets)}


def check_freeze(root):
    value = read(root / FREEZE)
    binding = {k: v for k, v in value.items() if k not in ("freeze_id", "source_snapshot")}
    if canonical_hash(binding) != value["freeze_id"]:
        raise ValueError("Design freeze content differs from its identity")
    for group in ("source_sha256", "asset_sha256"):
        for path, expected in value[group].items():
            if sha256_file(root / path) != expected:
                raise ValueError(f"Reference freeze integrity mismatch: {path}")
    for path, expected in value["source_sha256"].items():
        if sha256_file(root / value["source_snapshot"] / path) != expected:
            raise ValueError("Frozen source copy changed")
    for split in SPLITS:
        c = config(root, split)
        observed = pair_protocol_sha256(c, "reconstruction", dataset_fingerprints(root, c))
        if observed != value["configs"][split]["pair_protocol_sha256"]:
            raise ValueError(f"Resolved inference protocol changed: {split}")
    return value


def run(root, split, model):
    design = check_freeze(root)
    if split == "test":
        authorization = read(root / AUTHORIZATION)
        if authorization["design_freeze_sha256"] != sha256_file(root / FREEZE):
            raise ValueError("Test authorization does not bind current design")
        for path, expected in authorization["development_evidence_sha256"].items():
            if sha256_file(root / path) != expected:
                raise ValueError("Authorized development proof changed")
    output = result_dir(root, split)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "stylesphere.cli", "evaluate", "--pair", "reconstruction", "--model", model,
               "--config", f"config/reconstruction_reference_{split}.yaml", "--output-dir", relative(root, output), "--seed", "8410"]
    started = now()
    log = output / f"inference-{model}-{uuid.uuid4().hex[:8]}.log"
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, text=True)
    receipt = {"split": split, "model": model, "command": [relative(root, Path(command[0])) if Path(command[0]).is_relative_to(root) else command[0], *command[1:]],
               "started_at_utc": started, "completed_at_utc": now(), "return_code": completed.returncode,
               "log": relative(root, log), "log_sha256": sha256_file(log), "freeze_id": design["freeze_id"]}
    append(root / "environment/reference_execution_commands.jsonl", receipt)
    if completed.returncode:
        raise RuntimeError(f"Inference retained failures; inspect {relative(root, log)}")
    return receipt


def validate_raw(root, split, row, design):
    if row.get("model") not in MODELS or row.get("protocol_sha256") != design["configs"][split]["pair_protocol_sha256"]:
        raise ValueError("Reference raw model/protocol does not match freeze")
    execution = row.get("execution", {})
    for name, expected in design["expected_execution_bindings"][row["model"]].items():
        if execution.get(name) != expected:
            raise ValueError(f"Raw deployment binding differs: {name}")
    if execution.get("libraries") != design["expected_environments"][row["model"]]["libraries"]:
        raise ValueError("Actual model library versions differ from frozen deployment")
    for name, path in (("adapter_source_sha256", "src/stylesphere/reconstruction/worker.py"), ("runner_source_sha256", "src/stylesphere/reconstruction/runner.py")):
        if execution.get(name) != design["source_sha256"][path]:
            raise ValueError("Raw source binding differs")
    if row.get("seed") != 8410:
        raise ValueError("Raw seed differs")
    if row.get("timestamp_utc", "") < design["created_at_utc"]:
        raise ValueError("Reference raw record predates design freeze")
    for name, path in row.get("artifacts", {}).items():
        if sha256_file(root / path) != row["artifact_sha256"][name]:
            raise ValueError("Raw artifact hash differs")
    descriptor = read(root / row["artifacts"]["input_descriptor"])
    if digest(descriptor) != row["input_sha256"]:
        raise ValueError("Saved raw input descriptor differs from input anchor")
    return descriptor["manifest"]


def score_one(root, split, raw, manifest, protocol, directory):
    from PIL import Image
    from .rendering import render_views
    started = time.perf_counter()
    predicted_path, reference_path = root / raw["artifacts"]["mesh"], root / manifest["reference_mesh"]
    validate_glb(predicted_path)
    predicted, reference = load_mesh(predicted_path), load_mesh(reference_path)
    pn, ptransform = normalized_mesh(predicted)
    rn, rtransform = normalized_mesh(reference)
    p = sample_surface(pn.vertices, pn.faces, protocol["measurement_surface_points"], protocol["measurement_predicted_seed"])
    r = sample_surface(rn.vertices, rn.faces, protocol["measurement_surface_points"], protocol["measurement_reference_seed"])
    rp = sample_surface(pn.vertices, pn.faces, protocol["registration_surface_points"], protocol["registration_predicted_seed"])
    rr = sample_surface(rn.vertices, rn.faces, protocol["registration_surface_points"], protocol["registration_reference_seed"])
    transform, registration = rigid_icp(rp, rr, protocol["registration_max_iterations"], protocol["registration_absolute_tolerance"])
    aligned = apply(p, transform)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "sampled_surfaces.npz", predicted_normalized=p, predicted_rigid_aligned=aligned,
                        reference_normalized=r, registration_predicted=rp, registration_reference=rr)
    registration.update({"predicted_normalization_transform": ptransform.tolist(), "reference_normalization_transform": rtransform.tolist(),
                         "selected_rigid_transform": transform.tolist(), "determinant": float(np.linalg.det(transform[:3, :3])),
                         "scale_fitted_after_normalization": False, "reflection_allowed": False})
    write(directory / "registration.json", registration)
    thresholds = protocol["fscore_thresholds"]
    measured = {"unaligned_" + k: v for k, v in geometry_metrics(p, r, thresholds).items()}
    measured.update({"rigid_aligned_" + k: v for k, v in geometry_metrics(aligned, r, thresholds).items()})
    views = read(root / manifest["camera_info"])["views"]
    view_metrics = []
    for condition, extra in (("unaligned", np.eye(4)), ("rigid_aligned", transform)):
        display = predicted.copy()
        # Explicitly reverse REFERENCE normalization to use saved reference cameras.
        display.apply_transform(np.linalg.inv(rtransform) @ extra @ ptransform)
        rendered = render_views(display, views, directory / condition)
        per_view = []
        for view in rendered:
            with Image.open(view["silhouette"]) as a, Image.open(root / view["camera"]["reference_silhouette"]) as b:
                value = silhouette_iou(np.asarray(a) > 0, np.asarray(b.convert("L")) > 0)
            per_view.append(value)
            view_metrics.append({"condition": condition, "view_id": view["view_id"], "silhouette_iou": value,
                                 "predicted_silhouette": relative(root, view["silhouette"]),
                                 "reference_silhouette": view["camera"]["reference_silhouette"]})
        measured[condition + "_supplementary_silhouette_iou"] = float(np.mean(per_view))
    write(directory / "views.json", {"interpretation": "Supplementary independently normalized shape views, with explicitly reference-assisted rigid alignment where labelled", "per_view": view_metrics})
    return measured, time.perf_counter() - started


def score(root, split):
    design = check_freeze(root)
    protocol = design["metric_protocol"]
    output = result_dir(root, split)
    raw_path = output / "reconstruction/records.jsonl"
    records, metric_path = lines(raw_path), output / "reference_metrics.jsonl"
    prior = {x["raw_record_sha256"]: x for x in lines(metric_path) if x["status"] in ("success", "not_evaluated_raw_failure")}
    expected_ids = {x["object_id"] for x in csv.DictReader((root / SPLITS[split] / "objects.csv").open(encoding="utf-8-sig", newline=""))}
    for raw in records:
        if raw["case_id"] not in expected_ids:
            raise ValueError("Raw case is outside frozen split")
        manifest = validate_raw(root, split, raw, design)
        key = canonical_hash(raw)
        if key in prior:
            for path, expected in prior[key].get("artifact_sha256", {}).items():
                if sha256_file(root / path) != expected:
                    raise ValueError("Resumed supplemental metric artifact changed")
            continue
        result = {"model": raw["model"], "case_id": raw["case_id"], "category": manifest["category"], "split": split,
                  "seed": raw["seed"], "raw_record_sha256": key, "raw_records": relative(root, raw_path),
                  "input_sha256": raw["input_sha256"], "protocol_sha256": design["metric_protocol_sha256"],
                  "freeze_id": design["freeze_id"], "status": "not_evaluated_raw_failure", "metrics": {}, "artifact_sha256": {}}
        if raw["status"] == "success":
            directory = output / "shape_metrics" / raw["model"] / raw["case_id"] / uuid.uuid4().hex[:8]
            try:
                result["metrics"], result["scoring_wall_s"] = score_one(root, split, raw, manifest, protocol, directory)
                result["status"] = "success"
                result["artifact_sha256"] = {relative(root, p): sha256_file(p) for p in directory.rglob("*") if p.is_file()}
            except Exception as error:
                result["status"] = "failed_metric_computation"
                result["error"] = f"{type(error).__name__}: {error}"
                append(metric_path, result)
                raise
        append(metric_path, result)
        print(json.dumps({"split": split, "model": raw["model"], "case_id": raw["case_id"], "status": result["status"]}), flush=True)
    return summarize(root, split)


def summarize(root, split):
    output = result_dir(root, split)
    raw = lines(output / "reconstruction/records.jsonl")
    metrics = lines(output / "reference_metrics.jsonl")
    retained = {}
    for row in metrics:
        if row["status"] == "success":
            retained[(row["model"], row["case_id"])] = row
    paired = sorted({c for m, c in retained if m == MODELS[0]} & {c for m, c in retained if m == MODELS[1]})
    summary = {"split": split, "interpretation": "Normalized shape; rigid-aligned scores explicitly reference-assisted, no absolute size/pose accuracy claim",
               "raw_attempts": len(raw), "models": {}, "paired_objects": len(paired), "paired_differences_spar3d_minus_sf3d": {}}
    for model in MODELS:
        attempts = [r for r in raw if r["model"] == model]
        good = [r for (m, _), r in retained.items() if m == model]
        names = sorted(set.intersection(*(set(r["metrics"]) for r in good))) if good else []
        summary["models"][model] = {"attempts": len(attempts), "success": sum(r["status"] == "success" for r in attempts),
            "failed": sum(r["status"] == "failed" for r in attempts), "blocked": sum(r["status"] == "blocked" for r in attempts),
            "scored_objects": len(good), "category_counts": {c: sum(r["category"] == c for r in good) for c in ("shoe", "backpack")},
            "mean_metrics": {n: float(np.mean([r["metrics"][n] for r in good])) for n in names}}
    if paired:
        groups = [[i for i, c in enumerate(paired) if retained[(MODELS[0], c)]["category"] == category] for category in ("shoe", "backpack")]
        rng = np.random.default_rng(8414)
        indices = np.stack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if g]) for _ in range(2000)])
        names = sorted(set.intersection(*(set(retained[(m, c)]["metrics"]) for c in paired for m in MODELS)))
        for name in names:
            differences = np.array([retained[(MODELS[1], c)]["metrics"][name] - retained[(MODELS[0], c)]["metrics"][name] for c in paired])
            low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
            summary["paired_differences_spar3d_minus_sf3d"][name] = {"mean": float(differences.mean()), "ci95_low": float(low), "ci95_high": float(high), "paired_objects": len(paired)}
    write(output / "reference_summary.json", summary)
    fields = ["model", "case_id", "category", "split", "status"] + sorted({k for r in metrics for k in r["metrics"]})
    with (output / "reference_per_case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in metrics:
            writer.writerow({**{k: r[k] for k in fields[:5]}, **r["metrics"]})
    return summary


def authorize_test(root):
    design = check_freeze(root)
    output = result_dir(root, "development")
    raw, scores = lines(output / "reconstruction/records.jsonl"), lines(output / "reference_metrics.jsonl")
    ids = {r["object_id"] for r in csv.DictReader((root / SPLITS["development"] / "objects.csv").open(encoding="utf-8-sig"))}
    required = {(m, c) for m in MODELS for c in ids}
    if not required.issubset({(r["model"], r["case_id"]) for r in raw if r["status"] == "success"}) or not required.issubset({(r["model"], r["case_id"]) for r in scores if r["status"] == "success"}):
        raise ValueError("Both models must export and score all4development objects before test")
    if (result_dir(root, "test") / "reconstruction/records.jsonl").exists():
        raise ValueError("First test authorization must precede test outputs")
    value = {"authorized_at_utc": now(), "design_freeze_id": design["freeze_id"], "design_freeze_sha256": sha256_file(root / FREEZE),
             "development_objects": 4, "successful_development_executions": len(required), "protocol_changed_after_development": False,
             "development_evidence_sha256": {relative(root, output / p): sha256_file(output / p) for p in ("reconstruction/records.jsonl", "reference_metrics.jsonl", "reference_summary.json")}}
    write(root / AUTHORIZATION, value, exclusive=True)
    return value


def verify(root):
    design, checked, errors = check_freeze(root), 0, []
    for split in SPLITS:
        output = result_dir(root, split)
        raw = {canonical_hash(x): x for x in lines(output / "reconstruction/records.jsonl")}
        for score_row in lines(output / "reference_metrics.jsonl"):
            row = raw.get(score_row["raw_record_sha256"])
            if row is None:
                errors.append("Supplemental score lost raw-record anchor")
                continue
            validate_raw(root, split, row, design)
            for path, expected in score_row["artifact_sha256"].items():
                if sha256_file(root / path) != expected:
                    errors.append(f"Metric artifact changed: {path}")
            if score_row["status"] != "success":
                continue
            sample = next(p for p in score_row["artifact_sha256"] if p.endswith("sampled_surfaces.npz"))
            with np.load(root / sample, allow_pickle=False) as arrays:
                for prefix, field in (("unaligned_", "predicted_normalized"), ("rigid_aligned_", "predicted_rigid_aligned")):
                    recalculated = geometry_metrics(arrays[field], arrays["reference_normalized"], design["metric_protocol"]["fscore_thresholds"])
                    for name, value in recalculated.items():
                        if not np.isclose(value, score_row["metrics"][prefix + name], rtol=1e-10, atol=1e-12):
                            errors.append(f"Recomputed geometry differs: {score_row['case_id']}:{prefix+name}")
            checked += 1
    value = {"checked_at_utc": now(), "integrity_passed": not errors, "freeze_id": design["freeze_id"], "scored_executions_checked": checked, "errors": errors}
    write(root / "environment/reference_benchmark_integrity.json", value)
    if errors:
        raise ValueError("Reference integrity checks failed")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "score", "authorize-test", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--model", choices=MODELS, default="sf3d")
    args = parser.parse_args()
    root = args.root.resolve()
    actions = {"freeze": lambda: freeze(root), "run": lambda: run(root, args.split, args.model),
               "score": lambda: score(root, args.split), "authorize-test": lambda: authorize_test(root), "verify": lambda: verify(root)}
    print(json.dumps(actions[args.command](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
