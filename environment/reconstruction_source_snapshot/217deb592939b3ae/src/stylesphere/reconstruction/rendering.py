"""Optional pyrender standard views; no renderer is bundled or auto-installed."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def standard_cameras(distance: float = 2.5, size: int = 512) -> list[dict]:
    cameras = []
    for azimuth in (0, 90, 180, 270):
        theta, elevation = np.deg2rad(azimuth), np.deg2rad(20)
        eye = distance * np.array([np.sin(theta) * np.cos(elevation), np.sin(elevation), np.cos(theta) * np.cos(elevation)])
        back = eye / np.linalg.norm(eye)
        right = np.cross([0, 1, 0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, up, back], axis=1)
        pose[:3, 3] = eye
        cameras.append({"view_id": f"azimuth_{azimuth:03d}", "camera_to_world": pose.tolist(), "width": size, "height": size, "yfov": float(np.deg2rad(45))})
    return cameras


def render_views(mesh, cameras: list[dict], output_dir: Path) -> list[dict]:
    """Render fixed white background and camera-aligned light using OpenGL poses.

    Caller supplies an already aligned mesh. Camera matrices use OpenGL camera
    axes (+X right, +Y up, -Z forward), not OpenCV extrinsics.
    """
    import pyrender
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, spec in enumerate(cameras):
        width, height = int(spec["width"]), int(spec["height"])
        pose = np.asarray(spec["camera_to_world"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or width <= 0 or height <= 0:
            raise ValueError("Invalid camera dimensions or pose")
        if "intrinsics" in spec:
            intrinsic = np.asarray(spec["intrinsics"], dtype=float)
            camera = pyrender.IntrinsicsCamera(fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2], znear=0.01, zfar=100)
        else:
            camera = pyrender.PerspectiveCamera(yfov=float(spec.get("yfov", np.deg2rad(45))), aspectRatio=width / height, znear=0.01, zfar=100)
        scene = pyrender.Scene(bg_color=[1., 1., 1., 1.], ambient_light=[0.35, 0.35, 0.35])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        scene.add(camera, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose)
        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, depth = renderer.render(scene)
        finally:
            renderer.delete()
        image_path, mask_path = output_dir / f"view_{index:02d}.png", output_dir / f"view_{index:02d}_mask.png"
        Image.fromarray(color[:, :, :3]).save(image_path)
        Image.fromarray((depth > 0).astype(np.uint8) * 255).save(mask_path)
        outputs.append({"view_id": spec.get("view_id", str(index)), "image": str(image_path), "silhouette": str(mask_path), "camera": spec})
    (output_dir / "cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    return outputs
