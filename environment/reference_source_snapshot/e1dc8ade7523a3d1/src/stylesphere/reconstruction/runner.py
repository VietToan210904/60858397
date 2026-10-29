"""Reconstruct actual objects, keeping missing or ineligible metrics explicit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from stylesphere.generation.runner import append, base_record, file_digest, invoke_worker, resolve, safe_id, successful_prior
from .metrics import align_mesh, geometry_metrics, load_mesh, rendered_similarity, sample_surface, silhouette_iou, validate_glb
from .rendering import render_views, standard_cameras
from .worker import artifact_reference

MODEL_IDS = {"spar3d": "stabilityai/stable-point-aware-3d", "sf3d": "stabilityai/stable-fast-3d"}


def flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def metadata(root: Path, value: str) -> dict:
    if not value:
        return {}
    return json.loads(value) if value.lstrip().startswith("{") else json.loads(resolve(root, value).read_text(encoding="utf-8"))


def evaluate_mesh(root: Path, row: dict, mesh_path: Path, destination: Path, options: dict, seed: int) -> tuple[dict, dict, list[str]]:
    metrics = validate_glb(mesh_path)
    artifacts, skipped = {}, []
    mesh = load_mesh(mesh_path)
    preparation = metadata(root, row.get("preparation_metadata", ""))
    alignment = preparation.get("alignment_transform")
    if alignment is not None:
        mesh = align_mesh(mesh, alignment)
    if flag(row.get("geometry_eligible")):
        if alignment is None or not row.get("reference_mesh") or not preparation.get("coordinate_units"):
            skipped.append("Geometry requires reference mesh, predeclared alignment_transform and coordinate_units")
        else:
            reference = load_mesh(resolve(root, row["reference_mesh"]))
            count = int(options.get("sample_points", options.get("surface_samples", 10000)))
            predicted_geometry = mesh.copy()
            geometry_reference = reference.copy()
            units = preparation["coordinate_units"]
            normalization = options.get("normalization", "none")
            if normalization == "shared_reference_unit_diagonal":
                bounds = np.asarray(reference.bounds)
                diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
                if diagonal <= 0:
                    raise ValueError("Reference bounding-box diagonal is zero")
                for geometry in (predicted_geometry, geometry_reference):
                    geometry.apply_translation(-bounds.mean(axis=0))
                    geometry.apply_scale(1 / diagonal)
                units = "reference_bounding_box_diagonal"
            elif normalization != "none":
                raise ValueError(f"Unsupported geometry normalization: {normalization}")
            predicted_points = sample_surface(predicted_geometry.vertices, predicted_geometry.faces, count, seed)
            reference_points = sample_surface(geometry_reference.vertices, geometry_reference.faces, count, seed + 1)
            metrics.update(geometry_metrics(predicted_points, reference_points, options.get("fscore_thresholds", [0.01, 0.02, 0.05])))
            metrics["surface_samples"] = count
            destination.mkdir(parents=True, exist_ok=True)
            points_path = destination / "sampled_surfaces.npz"
            np.savez_compressed(points_path, predicted=predicted_points, reference=reference_points)
            artifacts["surface_samples"] = artifact_reference(root, points_path)
            metric_metadata = destination / "geometry_protocol.json"
            metric_metadata.write_text(json.dumps({"coordinate_units": units, "normalization": normalization,
                                                   "alignment_transform": alignment, "sample_seed": seed,
                                                   "reference_sample_seed": seed + 1, "fscore_thresholds": options.get("fscore_thresholds", [0.01, 0.02, 0.05])}, indent=2), encoding="utf-8")
            artifacts["geometry_protocol"] = artifact_reference(root, metric_metadata)
    else:
        skipped.append("No eligible aligned 3D reference; Chamfer and F-score unavailable")
    camera_info = metadata(root, row.get("camera_info", ""))
    cameras = camera_info.get("views", [])
    eligible_views = alignment is not None and bool(cameras)
    if options.get("render_views", True):
        # Standard previews use object-centered scale only for visualization, never
        # for reference comparisons. Matching cameras use the unchanged aligned mesh.
        preview_mesh = mesh.copy()
        if not eligible_views:
            bounds = np.asarray(preview_mesh.bounds)
            span = float(np.max(bounds[1] - bounds[0]))
            if span <= 0:
                raise ValueError("Degenerate preview bounds")
            preview_mesh.apply_translation(-bounds.mean(axis=0))
            preview_mesh.apply_scale(1 / span)
        try:
            renders = render_views(preview_mesh, cameras if eligible_views else standard_cameras(), destination / "renders")
            for index, rendering in enumerate(renders):
                artifacts[f"render_{index:02d}"] = artifact_reference(root, rendering["image"])
                artifacts[f"silhouette_{index:02d}"] = artifact_reference(root, rendering["silhouette"])
            from PIL import Image
            view_metrics = []
            for rendering in renders:
                spec, measured = rendering["camera"], {}
                if eligible_views and flag(row.get("silhouette_eligible")) and spec.get("reference_silhouette"):
                    with Image.open(rendering["silhouette"]) as a, Image.open(resolve(root, spec["reference_silhouette"])) as b:
                        measured["silhouette_iou"] = silhouette_iou(np.asarray(a) > 0, np.asarray(b.convert("L")) > 0)
                if eligible_views and flag(row.get("render_similarity_eligible")) and spec.get("reference_rgb"):
                    with Image.open(rendering["image"]) as a, Image.open(resolve(root, spec["reference_rgb"])) as b:
                        measured.update(rendered_similarity(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))))
                if measured:
                    view_metrics.append({"view_id": rendering["view_id"], **measured})
            rendering_path = destination / "renders" / "manifest.json"
            portable_renders = []
            for rendering in renders:
                spec = dict(rendering["camera"])
                for field in ("reference_rgb", "reference_silhouette"):
                    if spec.get(field):
                        spec[field] = artifact_reference(root, resolve(root, spec[field]))
                portable_renders.append({**rendering, "image": artifact_reference(root, rendering["image"]),
                                         "silhouette": artifact_reference(root, rendering["silhouette"]), "camera": spec})
            rendering_path.write_text(json.dumps({"renders": portable_renders, "per_view_metrics": view_metrics, "preview_normalized": not eligible_views}, indent=2), encoding="utf-8")
            (destination / "renders" / "cameras.json").write_text(json.dumps([row["camera"] for row in portable_renders], indent=2), encoding="utf-8")
            artifacts["renders_manifest"] = artifact_reference(root, rendering_path)
            artifacts["cameras"] = artifact_reference(root, destination / "renders" / "cameras.json")
            for name in ("silhouette_iou", "render_rgb_mae", "render_psnr_db"):
                values = [v[name] for v in view_metrics if v.get(name) is not None]
                if values:
                    metrics[name] = float(np.mean(values))
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            skipped.append(f"Rendering/reference-view metrics unavailable: {type(exc).__name__}: {exc}")
    if not eligible_views:
        skipped.append("No matching cameras and explicit alignment; silhouette/render similarity unavailable")
    return metrics, artifacts, skipped


def run(root: Path, config: dict, output_dir: Path, seed: int, smoke: bool = False, model_keys: list[str] | None = None) -> dict:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    runner_source_sha256 = file_digest(Path(__file__))
    target, run_id = output_dir / "reconstruction", uuid.uuid4().hex
    options = config.get("evaluation", {}).get("reconstruction", {})
    dataset_dir = root / options.get("dataset_dir", "datasets/reconstruction")
    with (dataset_dir / "objects.csv").open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if smoke:
        rows = rows[:1]
    if len({r.get("object_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate reconstruction object IDs")
    for row in rows:
        safe_id(row["object_id"])
    records, resumed = [], 0
    for key in model_keys or list(MODEL_IDS):
        if key not in MODEL_IDS:
            raise ValueError(f"Unknown reconstruction model {key}")
        model_config = dict(config.get("models", {}).get(key, {}))
        for name in ("checkout", "weights_path"):
            if model_config.get(name):
                model_config[name] = str(resolve(root, model_config[name]))
        if model_config.get("python") and any(s in model_config["python"] for s in ("/", "\\")):
            model_config["python"] = str(resolve(root, model_config["python"]))
        if not rows:
            record = base_record("reconstruction", key, MODEL_IDS[key], "__dataset_preflight__", model_config, seed, smoke, run_id, {})
            record["error"] = "No acquired reference objects in datasets/reconstruction/objects.csv; no inference attempted"
            record["inference_attempted"] = False
            record["protocol_sha256"] = config.get("protocol_sha256")
            append(target / "records.jsonl", record)
            records.append(record)
            continue
        cases, templates, descriptors = [], {}, {}
        pending_rows = []
        for row in rows:
            descriptor = {"manifest": row, "file_sha256": {}}
            for field in ("source_image", "reference_mesh", "camera_info", "preparation_metadata"):
                value = row.get(field)
                if value and not value.lstrip().startswith("{"):
                    candidate = resolve(root, value)
                    descriptor["file_sha256"][field] = file_digest(candidate) if candidate.is_file() else "missing"
            try:
                cameras = metadata(root, row.get("camera_info", "")).get("views", [])
                for index, camera in enumerate(cameras):
                    for field in ("reference_rgb", "reference_silhouette"):
                        if camera.get(field):
                            candidate = resolve(root, camera[field])
                            descriptor["file_sha256"][f"view_{index}_{field}"] = file_digest(candidate) if candidate.is_file() else "missing"
            except (OSError, ValueError) as exc:
                descriptor["camera_metadata_error"] = str(exc)
            measurement_config = {**model_config, "evaluation": options, "warmup_runs": config.get("evaluation", {}).get("warmup_runs", 1)}
            record = base_record("reconstruction", key, MODEL_IDS[key], row["object_id"], measurement_config, seed, smoke, run_id, descriptor)
            record["protocol_sha256"] = config.get("protocol_sha256")
            previous = successful_prior(target / "records.jsonl", model=key, case_id=row["object_id"], seed=seed, mode=record["mode"],
                                        config_sha=record["config_sha256"], input_sha=record["input_sha256"], protocol_sha=record["protocol_sha256"], root=root)
            if previous:
                records.append(previous)
                resumed += 1
            else:
                cases.append({"case_id": row["object_id"], "source_image": str(resolve(root, row["source_image"])), "seed": seed,
                              "mesh": str(target / run_id / key / row["object_id"] / "mesh.glb")})
                templates[row["object_id"]] = record
                descriptors[row["object_id"]] = descriptor
                pending_rows.append(row)
        if not cases:
            continue
        for case in cases:
            descriptor_path = Path(case["mesh"]).parent / "input_descriptor.json"
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor_path.write_text(json.dumps(descriptors[case["case_id"]], sort_keys=True, indent=2), encoding="utf-8")
        request = {"model": key, "model_id": MODEL_IDS[key], "model_config": model_config, "project_root": str(root),
                   "warmup_runs": int(config.get("evaluation", {}).get("warmup_runs", 1)),
                   "texture_resolution": int(options.get("texture_resolution", 1024)), "cases": cases}
        response = invoke_worker(Path(__file__).with_name("worker.py"), request, target / run_id / key / "diagnostics",
                                 model_config.get("python", sys.executable), float(options.get("timeout_s", 7200)), working_dir=root)
        lookup = {r["case_id"]: r for r in response.get("results", [])}
        for row in pending_rows:
            record = templates[row["object_id"]]
            result = lookup.get(row["object_id"])
            record.update(result or {"status": "blocked", "error": response.get("error", "No worker case result")})
            if result is None and response.get("diagnostics"):
                record["diagnostics"] = response["diagnostics"]
            record.setdefault("artifacts", {})["input_descriptor"] = artifact_reference(
                root, target / run_id / key / row["object_id"] / "input_descriptor.json")
            record["execution"] = {k: response.get(k) for k in ("python", "device", "torch_version", "cuda_version", "libraries", "seed_policy", "adapter_source_sha256", "source_revision", "source_dirty", "resolved_checkpoint_revision", "checkpoint_provenance", "native_extension_provenance", "auxiliary_provenance")}
            record["execution"]["runner_source_sha256"] = runner_source_sha256
            if response.get("resolved_checkpoint_revision"):
                record["checkpoint_revision"] = response["resolved_checkpoint_revision"]
            if record["status"] == "success":
                try:
                    mesh_path = resolve(root, record["artifacts"]["mesh"])
                    metrics, artifacts, skipped = evaluate_mesh(root, row, mesh_path, mesh_path.parent / "evaluation", options, seed)
                    record["metrics"].update(metrics)
                    record["artifacts"].update(artifacts)
                    record["metrics_unavailable"] = skipped
                except Exception as exc:
                    record.update(status="failed", error=f"Mesh validation/metrics failed: {type(exc).__name__}: {exc}")
            # Invalid exports remain useful evidence, with actual retained bytes
            # hashed even when mesh validation failed.
            record["artifacts"] = {name: artifact_reference(root, resolve(root, path))
                                   for name, path in record.get("artifacts", {}).items()
                                   if resolve(root, path).is_file()}
            record["artifact_sha256"] = {name: file_digest(resolve(root, path)) for name, path in record["artifacts"].items()}
            append(target / "records.jsonl", record)
            records.append(record)
    return {"pair": "reconstruction", "run_id": run_id, "records": len(records), "resumed": resumed, "success": sum(r["status"] == "success" for r in records),
            "blocked": sum(r["status"] == "blocked" for r in records), "failed": sum(r["status"] == "failed" for r in records)}
