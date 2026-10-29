"""Prepare calibrated views of real GSO scans, separately from real-photo ABO.

Run in the pinned preview environment after acquisition. The mesh/texture QA,
source-only normalization and view cameras are independent of model outputs.
No model inference or prediction-dependent alignment occurs here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import trimesh

from stylesphere.datasets import OBJECT_FIELDS
from stylesphere.reconstruction.rendering import render_views, standard_cameras


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing preparation differs; no overwrite: {path.name}")
    else:
        path.write_bytes(content)


def portable(root, value):
    return Path(value).resolve().relative_to(root).as_posix()


def normalized_reference(obj: Path, texture: Path):
    scene = trimesh.load(obj, force="scene", process=False)
    if len(scene.geometry) != 1:
        raise ValueError("Expected single original textured geometry")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Nonfinite/empty source vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid/empty source triangles")
    uv = np.asarray(getattr(mesh.visual, "uv", None))
    if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
        raise ValueError("Missing/nonfinite original UV coordinates")
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise ValueError("No positive finite source surface area")
    with Image.open(texture) as source:
        source.load()
        image = source.convert("RGB")
    if min(image.size) < 16:
        raise ValueError("Source texture too small")
    # GSO MTL says texture.png while archive keeps it in materials/textures.
    # Bind the original pixels explicitly; leave original files unchanged.
    mesh.visual.material.image = image
    original_bounds = np.asarray(mesh.bounds).copy()
    # Fixed Gazebo +Z-up -> comparison +Y-up, Rx(-90 degrees).
    rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    rotated = vertices @ rotation.T
    bounds = np.stack([rotated.min(axis=0), rotated.max(axis=0)])
    span = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Degenerate source bounds")
    transform = np.eye(4)
    transform[:3, :3] = rotation / span
    transform[:3, 3] = -bounds.mean(axis=0) / span
    mesh.apply_transform(transform)
    return mesh, transform, {"source_vertices": len(vertices), "source_faces": len(faces),
                            "source_degenerate_faces": int(np.sum(areas <= 1e-15)),
                            "source_texture_size": list(image.size), "source_bounds": original_bounds.tolist(),
                            "normalized_bounds": mesh.bounds.tolist(), "uv_valid": True, "texture_decoded": True}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if list(csv.DictReader(stream)) != [{k: str(v) for k, v in x.items()} for x in rows]:
                raise ValueError("Existing object manifest differs; no overwrite")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OBJECT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "datasets/reconstruction_reference"
    acquisition_path = folder / "gso_acquisition_manifest.json"
    acquisition = json.loads(acquisition_path.read_text())
    selection = json.loads((folder / "source_selection.json").read_text())
    if len(acquisition["objects"]) != 24 or sha(folder / "source_selection.json") != acquisition["source_selection_sha256"]:
        raise ValueError("Complete pinned24-object acquisition required")
    manifest_path = folder / "preparation_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for path, expected in prior["asset_sha256"].items():
            if sha(root / path) != expected:
                raise ValueError("Frozen prepared asset changed")
        print(json.dumps({"status": "verified_existing", "objects": len(prior["objects"]), "gpu_model_inference": False}))
        return
    cases, qa = [], []
    for item in acquisition["objects"]:
        source = root / item["extracted_path"]
        obj, mtl, texture = source / "meshes/model.obj", source / "meshes/model.mtl", source / "materials/textures/texture.png"
        try:
            mesh, transform, stats = normalized_reference(obj, texture)
            qa.append({"source_name": item["source_name"], "eligible": True, "exclusion_reason": None, **stats})
            cases.append((item, mesh, transform, stats, obj, mtl, texture))
        except (ValueError, OSError, TypeError) as error:
            qa.append({"source_name": item["source_name"], "eligible": False, "exclusion_reason": f"{type(error).__name__}: {error}"})
    save(folder / "source_qa.json", {"selection_before_model_outputs": True, "objects": qa})
    if len(cases) != 24:
        raise ValueError("Invalid source candidates recorded; stop for explicit honest subset/selection amendment before any inference")
    if args.qa_only:
        print(json.dumps({"eligible_source_objects": len(cases), "model_inference": False}))
        return
    rows = {"development": [], "test": []}
    prepared = []
    hash_paths = []
    for item, mesh, transform, stats, obj, mtl, texture in cases:
        name = item["source_name"]
        object_id = "gso-" + name
        directory = folder / "assets" / object_id
        directory.mkdir(parents=True, exist_ok=True)
        originals = {}
        for p in (obj, mtl, texture, obj.parents[1] / "metadata.pbtxt", obj.parents[1] / "model.sdf", obj.parents[1] / "model.config"):
            relative = p.relative_to(obj.parents[1])
            dest = directory / "original" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if sha(dest) != sha(p):
                    raise ValueError("Existing original differs")
            else:
                shutil.copyfile(p, dest)
            originals[portable(root, dest)] = sha(dest)
            hash_paths.append(dest)
        reference_path = directory / "reference.glb"
        data = mesh.export(file_type="glb")
        if reference_path.exists() and reference_path.read_bytes() != data:
            raise ValueError("Existing normalized reference differs")
        reference_path.write_bytes(data)
        input_spec = {"view_id": "conditioning", "camera_to_world": [[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,-2.5],[0.,0.,0.,1.]],
                      "width": 512, "height": 512, "yfov": float(np.deg2rad(40.)),
                      "coordinate_convention": "OpenGL camera +X right,+Y up,-Z forward"}
        input_render = render_views(mesh, [input_spec], directory / "input_render")[0]
        with Image.open(input_render["image"]) as image, Image.open(input_render["silhouette"]) as alpha:
            rgba = image.convert("RGBA")
            rgba.putalpha(alpha.convert("L"))
            input_path = directory / "input.png"
            rgba.save(input_path)
            foreground_fraction = float(np.mean(np.asarray(alpha) > 0))
        if not (0.005 < foreground_fraction < 0.95):
            raise ValueError("Calibrated conditioning image foreground/clipping QA failed")
        heldout = render_views(mesh, standard_cameras(2.5, 512), directory / "reference_views")
        views = []
        for row in heldout:
            spec = dict(row["camera"])
            spec["reference_rgb"] = portable(root, row["image"])
            spec["reference_silhouette"] = portable(root, row["silhouette"])
            spec["coordinate_convention"] = "OpenGL +X right,+Y up,-Z forward"
            views.append(spec)
        camera_path = directory / "camera_info.json"
        save(camera_path, {"input_view": input_spec, "views": views, "heldout_from_conditioning_view": True,
                           "near": 0.01, "far": 100., "lighting": {"ambient": [0.35,0.35,0.35], "camera_aligned_directional_intensity": 2.0},
                           "reference_background": "white", "conditioning_alpha": "depth>0, exact renderer foreground"})
        metadata = {"object_id": object_id, "source_name": name, "source_dataset": "Google Scanned Objects", "source_version": item["version"],
                    "split": item["intended_split"], "input_kind": "rendered_view_of_real_scanned_reference", "ai_generated": False,
                    "real_photograph_input": False, "real_scanned_reference": True, "source_url": item["archive_url"],
                    "licence": "CC-BY-4.0", "attribution": "Google Research; Copyright2020Google LLC",
                    "original_file_sha256": originals, "archive_sha256": item["archive_sha256"],
                    "source_to_reference_transform": transform.tolist(), "source_coordinate_convention": "Gazebo +Z up, fixed Rx(-90deg) maps to comparison +Y up",
                    "source_axes_reference": "https://gazebosim.org/api/sim/10/frame_reference.html",
                    "normalization": "REFERENCE-ONLY axis-aligned bounding-box center and maximum extent=1, no prediction used",
                    "coordinate_units": "dimensionless_reference_maximum_extent", "normalization_physical_scale_claim": False,
                    "reference_mesh": portable(root, reference_path), "reference_mesh_sha256": sha(reference_path),
                    "prepared_image": portable(root, input_path), "prepared_image_sha256": sha(input_path),
                    "image_sha256": sha(input_path), "prepared_sha256": sha(input_path),
                    "camera_info": portable(root, camera_path), "camera_info_sha256": sha(camera_path),
                    "input_alpha_foreground_fraction": foreground_fraction, "source_qa": stats,
                    "alignment_status": "No prediction alignment claimed. Model-global development calibration or separately declared nuisance-aligned analysis required.",
                    "geometry_eligible": False, "silhouette_eligible": False, "render_similarity_eligible": False,
                    "redistributable": True, "changes": "Original files retained unchanged; original UV+texture explicitly bound; fixed coordinate rotation and reference-only normalization; GLB conversion and calibrated rendered views/alpha."}
        prep_path = directory / "preparation.json"
        save(prep_path, metadata)
        row = {k: "" for k in OBJECT_FIELDS}
        row.update({"object_id": object_id, "source_image": portable(root, input_path), "category": item["category"],
                    "source_dataset": "Google Scanned Objects", "reference_mesh": portable(root, reference_path), "camera_info": portable(root, camera_path),
                    "preparation_metadata": portable(root, prep_path), "geometry_eligible": "false", "silhouette_eligible": "false", "render_similarity_eligible": "false",
                    "licence": "CC-BY-4.0", "source_revision": f"GoogleResearch/{name}/v{item['version']};zip-sha256:{item['archive_sha256']}",
                    "image_sha256": sha(input_path), "mesh_sha256": sha(reference_path), "redistributable": "true"})
        rows[item["intended_split"]].append(row)
        prepared.append({"object_id": object_id, "split": item["intended_split"], "category": item["category"], "metadata": portable(root, prep_path)})
        hash_paths.extend([p for p in directory.rglob("*") if p.is_file()])
        print(json.dumps({"prepared": object_id, "split": item["intended_split"], "alpha_fraction": foreground_fraction}), flush=True)
    write_csv(folder / "objects.csv", rows["test"])
    write_csv(folder / "development/objects.csv", rows["development"])
    licence_source = root / "datasets/reconstruction/SOURCE_LICENSE_CC_BY_4_0.txt"
    licence_path = folder / "SOURCE_LICENSE_CC_BY_4_0.txt"
    if not licence_path.exists():
        shutil.copyfile(licence_source, licence_path)
    rights = {"licence": "CC-BY-4.0", "official_source": "https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/",
              "official_licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_sha256": sha(licence_path),
              "asset_attribution": "Google Research; Copyright2020Google LLC", "dataset_attribution": "Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B.McHugh, Vincent Vanhoucke",
              "publication_citation": "Google Scanned Objects: A High-Quality Dataset of3DScanned Household Items, ICRA2022",
              "redistributable": True, "objects": [x["object_id"] for x in prepared],
              "scope": "Only24selected original meshes/materials/textures/metadata and their normalized GLB and deterministic reference-render derivatives; source ZIP cache excluded.",
              "changes": "Fixed reference coordinate rotation, reference-only normalization, explicit original texture binding, GLB conversion, rendered RGB/alpha/reference silhouettes; no AI-generated reference content."}
    save(folder / "RIGHTS_AND_ATTRIBUTION.json", rights)
    hash_paths += [folder / "objects.csv", folder / "development/objects.csv", folder / "source_qa.json", folder / "source_selection.json", acquisition_path, licence_path, folder / "RIGHTS_AND_ATTRIBUTION.json"]
    versions = {x: importlib.metadata.version(x) for x in ("numpy", "scipy", "trimesh", "pyrender", "Pillow", "PyOpenGL")}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "objects": prepared,
                "test_category_counts": {"shoe": 17, "backpack": 3}, "development_category_counts": {"shoe": 2, "backpack": 2},
                "preparation_libraries": versions, "preparation_script_sha256": sha(Path(__file__)),
                "renderer_source_sha256": sha(root / "src/stylesphere/reconstruction/rendering.py"),
                "model_inference_performed": False, "asset_sha256": {portable(root, p): sha(p) for p in sorted(set(hash_paths))}}
    save(manifest_path, manifest)
    print(json.dumps({"test_objects": len(rows['test']), "development_objects": len(rows['development']), "asset_files": len(manifest['asset_sha256']), "model_inference_performed": False}), flush=True)


if __name__ == "__main__":
    main()
