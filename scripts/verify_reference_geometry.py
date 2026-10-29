"""Independently resample actual frozen meshes and audit supplementary scores.

CPU-only: no inference, rendering, network, or checkpoint re-download/re-hash.
This verifies that saved clouds and view metrics remain anchored to real files.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from stylesphere.common.io import canonical_hash, sha256_file
from stylesphere.reconstruction.metrics import geometry_metrics, load_mesh, sample_surface, silhouette_iou
from stylesphere.reconstruction.reference_benchmark import apply, check_freeze, lines, normalized_mesh, result_dir, validate_raw


def audit(root):
    root = root.resolve()
    design = check_freeze(root)
    protocol, errors, split_results = design["metric_protocol"], [], {}
    for split, expected_count in (("development", 8), ("test", 40)):
        directory = result_dir(root, split)
        raw = lines(directory / "reconstruction/records.jsonl")
        by_hash = {canonical_hash(r): r for r in raw}
        dataset = root / "datasets/reconstruction_reference" / ("development" if split == "development" else "")
        with (dataset / "objects.csv").open(encoding="utf-8-sig", newline="") as stream:
            expected_manifests = {r["object_id"]: r for r in csv.DictReader(stream)}
        successful = {}
        for r in lines(directory / "reference_metrics.jsonl"):
            if r["status"] == "success":
                successful[(r["model"], r["case_id"])] = r
        models = {}
        for model in ("sf3d", "spar3d"):
            attempts = [r for r in raw if r["model"] == model]
            completed = [r for r in attempts if r["status"] == "success"]
            models[model] = {"attempts": len(attempts), "successful_exports": len(completed),
                "failed": sum(r["status"] == "failed" for r in attempts), "blocked": sum(r["status"] == "blocked" for r in attempts),
                "mean_inference_s": float(np.mean([r["timings"]["inference_s"] for r in completed])) if completed else None,
                "recorded_model_load_s_values": sorted({r["timings"]["model_load_s"] for r in completed}),
                "mean_preprocessing_s": float(np.mean([r["timings"]["preprocessing_s"] for r in completed])) if completed else None,
                "mean_export_s": float(np.mean([r["timings"]["export_s"] for r in completed])) if completed else None,
                "mean_end_to_end_s_excluding_load_warmup": float(np.mean([r["timings"]["end_to_end_s"] for r in completed])) if completed else None,
                "peak_allocated_mib": max((r["metrics"]["peak_allocated_mb"] for r in completed), default=None),
                "peak_reserved_mib": max((r["metrics"]["peak_reserved_mb"] for r in completed), default=None),
                "checkpoint_revision": sorted({r["checkpoint_revision"] for r in completed}),
                "source_revision": sorted({r["execution"]["source_revision"] for r in completed}),
                "runner_source_sha256": sorted({r["execution"]["runner_source_sha256"] for r in completed}),
                "adapter_source_sha256": sorted({r["execution"]["adapter_source_sha256"] for r in completed}),
                "run_ids": sorted({r["run_id"] for r in attempts})}
        checked = 0
        for key, score in successful.items():
            record = by_hash[score["raw_record_sha256"]]
            manifest = validate_raw(root, split, record, design)
            if manifest != expected_manifests[record["case_id"]]:
                errors.append(f"Raw descriptor manifest differs from frozen split:{split}:{key}")
            descriptor = json.loads((root / record["artifacts"]["input_descriptor"]).read_text())
            expected_hashes = {field: sha256_file(root / manifest[field]) for field in ("source_image", "reference_mesh", "camera_info", "preparation_metadata")}
            cameras = json.loads((root / manifest["camera_info"]).read_text())["views"]
            for index, camera in enumerate(cameras):
                for field in ("reference_rgb", "reference_silhouette"):
                    expected_hashes[f"view_{index}_{field}"] = sha256_file(root / camera[field])
            if descriptor["file_sha256"] != expected_hashes:
                errors.append(f"Raw descriptor file bindings differ from actual frozen inputs:{split}:{key}")
            meshes = {"predicted": load_mesh(root / record["artifacts"]["mesh"]), "reference": load_mesh(root / manifest["reference_mesh"])}
            normalization = {k: normalized_mesh(m) for k, m in meshes.items()}
            normalized = {k: value[0] for k, value in normalization.items()}
            sample_path = next(p for p in score["artifact_sha256"] if p.endswith("sampled_surfaces.npz"))
            registration_path = next(p for p in score["artifact_sha256"] if p.endswith("registration.json"))
            views_path = next(p for p in score["artifact_sha256"] if p.endswith("views.json"))
            for path, expected in score["artifact_sha256"].items():
                if sha256_file(root / path) != expected:
                    errors.append(f"Artifact hash mismatch:{split}:{key}:{path}")
            reg = json.loads((root / registration_path).read_text())
            for kind in ("predicted", "reference"):
                if not np.allclose(normalization[kind][1], reg[kind + "_normalization_transform"], atol=1e-12):
                    errors.append(f"Declared normalization differs from actual mesh bounds:{split}:{key}:{kind}")
            transform = np.asarray(reg["selected_rigid_transform"])
            if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-9):
                errors.append(f"Registration includes scale/reflection:{split}:{key}")
            chosen = reg["trials"][reg["chosen_trial"]]
            if len(reg["trials"]) != 24 or chosen["symmetric_registration_chamfer_squared"] != min(t["symmetric_registration_chamfer_squared"] for t in reg["trials"]):
                errors.append(f"Fixed trial selection differs:{split}:{key}")
            if not np.allclose(chosen["transform"], transform, atol=1e-12):
                errors.append(f"Selected transform differs from trial:{split}:{key}")
            with np.load(root / sample_path, allow_pickle=False) as arrays:
                for kind, field, count_name, seed_name in (
                    ("predicted", "predicted_normalized", "measurement_surface_points", "measurement_predicted_seed"),
                    ("reference", "reference_normalized", "measurement_surface_points", "measurement_reference_seed"),
                    ("predicted", "registration_predicted", "registration_surface_points", "registration_predicted_seed"),
                    ("reference", "registration_reference", "registration_surface_points", "registration_reference_seed")):
                    mesh = normalized[kind]
                    actual = sample_surface(mesh.vertices, mesh.faces, protocol[count_name], protocol[seed_name])
                    if not np.allclose(actual, arrays[field], rtol=0, atol=1e-12):
                        errors.append(f"Saved cloud does not resample actual mesh:{split}:{key}:{field}")
                if not np.allclose(apply(arrays["predicted_normalized"], transform), arrays["predicted_rigid_aligned"], atol=1e-12):
                    errors.append(f"Aligned points differ from declared rigid transform:{split}:{key}")
                for prefix, field in (("unaligned_", "predicted_normalized"), ("rigid_aligned_", "predicted_rigid_aligned")):
                    calculated = geometry_metrics(arrays[field], arrays["reference_normalized"], protocol["fscore_thresholds"])
                    for name, value in calculated.items():
                        if not np.isclose(value, score["metrics"][prefix + name], rtol=1e-10, atol=1e-12):
                            errors.append(f"Geometry metric differs:{split}:{key}:{prefix+name}")
            views = json.loads((root / views_path).read_text())["per_view"]
            for condition in ("unaligned", "rigid_aligned"):
                stored_cameras_path = (root / views_path).parent / condition / "cameras.json"
                if json.loads(stored_cameras_path.read_text()) != cameras:
                    errors.append(f"Scored view cameras differ from frozen reference cameras:{split}:{key}:{condition}")
                values = []
                for view in [v for v in views if v["condition"] == condition]:
                    with Image.open(root / view["predicted_silhouette"]) as a, Image.open(root / view["reference_silhouette"]) as b:
                        value = silhouette_iou(np.asarray(a) > 0, np.asarray(b.convert("L")) > 0)
                    if not np.isclose(value, view["silhouette_iou"], atol=1e-12):
                        errors.append(f"View metric differs:{split}:{key}:{view['view_id']}")
                    values.append(value)
                if len(values) != 4 or not np.isclose(np.mean(values), score["metrics"][condition + "_supplementary_silhouette_iou"], atol=1e-12):
                    errors.append(f"View aggregate differs:{split}:{key}:{condition}")
            checked += 1
        if checked != expected_count:
            errors.append(f"Incomplete unique scored coverage:{split}:{checked}/{expected_count}")
        split_results[split] = {"raw_attempts": len(raw), "scored_objects_checked": checked, "expected_executions": expected_count,
                                "coverage_complete": checked == expected_count, "models": models}
    value = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "integrity_passed": not errors,
             "freeze_id": design["freeze_id"], "freeze_precedes_development": True,
             "interpretation": "Explicit independently normalized shape, additional reference-assisted rigid pose alignment, no absolute reconstruction accuracy",
             "checks": ["Actual mesh deterministic surface resampling", "Raw source/checkpoint/library/native/input/artifact bindings", "Independent measurement and registration points", "Proper rigid transforms and selected fixed trial", "Recomputed Chamfer/F-scores and4view silhouette IoU"],
             "splits": split_results, "errors": errors}
    authorization = json.loads((root / "environment/reference_benchmark_test_authorization.json").read_text())
    value["test_authorization_precedes_all_test_records"] = all(r["timestamp_utc"] > authorization["authorized_at_utc"] for r in lines(result_dir(root, "test") / "reconstruction/records.jsonl"))
    if not value["test_authorization_precedes_all_test_records"]:
        errors.append("Test record predates development proof authorization")
        value["integrity_passed"] = False
    bindings = ["environment/reference_benchmark_freeze.json", "environment/reference_benchmark_test_authorization.json",
                "config/reconstruction_reference_protocol.json", "datasets/reconstruction_reference/objects.csv",
                "datasets/reconstruction_reference/development/objects.csv", "datasets/reconstruction_reference/preparation_manifest.json",
                "scripts/verify_reference_geometry.py", "src/stylesphere/reconstruction/reference_benchmark.py"]
    for split in ("development", "test"):
        for suffix in ("reconstruction/records.jsonl", "reference_metrics.jsonl", "reference_summary.json", "reference_per_case.csv"):
            bindings.append(f"results/reference_{split}/{suffix}")
    value["source_sha256"] = {p: sha256_file(root / p) for p in bindings}
    path = root / "environment/reference_geometry_audit.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Supplementary geometry audit failed")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    audit(Path.cwd())
