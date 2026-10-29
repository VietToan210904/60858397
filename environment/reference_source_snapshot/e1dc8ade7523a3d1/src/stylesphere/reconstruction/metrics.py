"""Transparent geometry metrics in an explicitly shared coordinate frame."""
from __future__ import annotations

from pathlib import Path
import json
import struct

import numpy as np
from stylesphere.common.io import portable_path


def point_array(points) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite Nx3 point array")
    return result


def sample_surface(vertices, faces, count: int, seed: int = 8410) -> np.ndarray:
    """Area-weighted triangle selection and uniform barycentric sampling."""
    vertices = point_array(vertices)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Faces must be a nonempty integer Mx3 array")
    if faces.min() < 0 or faces.max() >= len(vertices) or count <= 0:
        raise ValueError("Invalid face indices or sample count")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Mesh has no finite positive surface area")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(faces), size=count, p=areas / areas.sum())]
    u, v = rng.random((2, count))
    root = np.sqrt(u)
    weights = np.stack((1 - root, root * (1 - v), root * v), axis=1)
    return np.einsum("ni,nij->nj", weights, chosen)


def geometry_metrics(predicted, reference, thresholds=(0.01, 0.02, 0.05)) -> dict:
    """Symmetric squared Chamfer (half-sum), and F-score at distance tau.

    Inputs must already share units, scale and alignment. No hidden ICP fitting,
    axis flipping or separate object normalization is performed.
    """
    from scipy.spatial import cKDTree

    predicted, reference = point_array(predicted), point_array(reference)
    a = cKDTree(reference).query(predicted, k=1)[0]
    b = cKDTree(predicted).query(reference, k=1)[0]
    result = {"chamfer_squared": float(0.5 * (np.mean(a**2) + np.mean(b**2))),
              "chamfer_l1": float(0.5 * (np.mean(a) + np.mean(b)))}
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("F-score thresholds must be positive finite coordinate-unit distances")
        precision, recall = float(np.mean(a <= threshold)), float(np.mean(b <= threshold))
        label = format(threshold, "g")
        result.update({f"precision_at_{label}": precision, f"recall_at_{label}": recall,
                       f"fscore_at_{label}": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result


def silhouette_iou(predicted, reference) -> float | None:
    predicted, reference = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("Silhouettes must be matching 2D masks")
    union = int(np.logical_or(predicted, reference).sum())
    return float(np.logical_and(predicted, reference).sum() / union) if union else None


def rendered_similarity(predicted, reference) -> dict:
    predicted, reference = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("Rendered views must be matching HxWx3 arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite rendered pixels")
    if predicted.min() < 0 or reference.min() < 0 or predicted.max() > 255 or reference.max() > 255:
        raise ValueError("Rendered RGB pixels must be in [0,255]")
    mse = float(np.mean(((predicted - reference) / 255)**2))
    return {"render_rgb_mae": float(np.mean(np.abs(predicted - reference)) / 255),
            "render_psnr_db": float(-10 * np.log10(mse)) if mse else None,
            "render_exact_match": mse == 0}


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="scene", process=False)
    if isinstance(mesh, trimesh.Scene):
        # dump applies scene-node transforms; geometry.values() alone would not.
        mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)
    return mesh


def validate_glb(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError("GLB output is missing")
    with path.open("rb") as fh:
        header = fh.read(12)
    if len(header) != 12:
        raise ValueError("Truncated GLB header")
    magic, version, length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2 or length != path.stat().st_size:
        raise ValueError("Invalid GLB v2 header or file length")
    mesh = load_mesh(path)
    vertices, faces = point_array(mesh.vertices), np.asarray(mesh.faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Output has no triangle geometry")
    sample_surface(vertices, faces, 1)
    degenerate = int(np.sum(np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                                   vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) <= 1e-15))
    return {"glb_export_success": 1, "mesh_valid": int(degenerate == 0), "mesh_vertices": len(vertices),
            "mesh_faces": len(faces), "mesh_degenerate_faces": degenerate, "mesh_watertight": bool(mesh.is_watertight)}


def align_mesh(mesh, transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("alignment_transform must be a finite affine 4x4 matrix")
    if abs(np.linalg.det(transform[:3, :3])) < 1e-15:
        raise ValueError("Alignment transform is singular")
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def verify_record(record: dict, root: Path | None = None) -> list[str]:
    """Recompute stored mesh/surface/view metrics without running any AI model."""
    if record.get("status") != "success":
        return []
    errors, measured = [], record.get("metrics", {})
    artifacts = record.get("artifacts", {})
    root = Path(root or Path.cwd())
    def location(value):
        return portable_path(root, value)
    def compare(name, value):
        observed = measured.get(name)
        if value is None:
            if observed is not None:
                errors.append(f"{name}: expected missing value")
        elif observed is None or not np.isclose(float(observed), float(value), rtol=1e-6, atol=1e-9):
            errors.append(f"{name}: differs from recomputed output")
    try:
        validity = validate_glb(location(artifacts["mesh"]))
        for name, value in validity.items():
            compare(name, value)
        if artifacts.get("surface_samples"):
            protocol = json.loads(location(artifacts["geometry_protocol"]).read_text(encoding="utf-8"))
            with np.load(location(artifacts["surface_samples"]), allow_pickle=False) as surfaces:
                calculated = geometry_metrics(surfaces["predicted"], surfaces["reference"], protocol["fscore_thresholds"])
                compare("surface_samples", len(surfaces["predicted"]))
            for name, value in calculated.items():
                compare(name, value)
        if artifacts.get("renders_manifest"):
            from PIL import Image
            manifest = json.loads(location(artifacts["renders_manifest"]).read_text(encoding="utf-8"))
            views = {v["view_id"]: v for v in manifest["renders"]}
            aggregate = {}
            for stored in manifest["per_view_metrics"]:
                view = views[stored["view_id"]]
                spec, values = view["camera"], {}
                if "silhouette_iou" in stored:
                    with Image.open(location(view["silhouette"])) as a, Image.open(location(spec["reference_silhouette"])) as b:
                        values["silhouette_iou"] = silhouette_iou(np.asarray(a) > 0, np.asarray(b.convert("L")) > 0)
                if "render_rgb_mae" in stored:
                    with Image.open(location(view["image"])) as a, Image.open(location(spec["reference_rgb"])) as b:
                        values.update(rendered_similarity(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))))
                for name, value in values.items():
                    previous = stored.get(name)
                    if value is not None and (previous is None or not np.isclose(float(previous), float(value))):
                        errors.append(f"View {stored['view_id']} {name}: differs from recomputed image")
                    if name in ("silhouette_iou", "render_rgb_mae", "render_psnr_db") and value is not None:
                        aggregate.setdefault(name, []).append(value)
            for name, values in aggregate.items():
                compare(name, np.mean(values))
    except Exception as exc:
        errors.append(f"Reconstruction revalidation failed: {type(exc).__name__}: {exc}")
    return errors
