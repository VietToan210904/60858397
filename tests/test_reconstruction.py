"""Synthetic numerical fixtures only; not benchmark geometry or observations."""
import struct

import numpy as np
import pytest

from stylesphere.reconstruction.metrics import geometry_metrics, rendered_similarity, sample_surface, silhouette_iou, validate_glb, verify_record
from stylesphere.reconstruction.rendering import standard_cameras


def test_failed_cuda_window_keeps_observed_peaks_and_warmup_is_not_inference(monkeypatch):
    from types import SimpleNamespace
    from stylesphere.reconstruction import worker
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append("sync"), reset_peak_memory_stats=lambda: calls.append("reset"),
                           max_memory_allocated=lambda: 42 * 2**20, max_memory_reserved=lambda: 80 * 2**20)
    moments = iter([10.0, 10.25])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(moments))
    failure = RuntimeError("synthetic CUDA OOM fixture; no actual GPU use")
    def fail():
        raise failure
    with pytest.raises(RuntimeError) as caught:
        worker.timed(SimpleNamespace(cuda=cuda), fail)
    assert caught.value is failure
    assert calls == ["sync", "reset", "sync"]
    case = {"case_id": "synthetic_unit_case"}
    observed = worker.failed_case(case, failure, "inference", None, is_oom=True)
    assert observed["timings"] == {"inference_s": .25}
    assert observed["metrics"] == {"peak_allocated_mb": 42., "peak_reserved_mb": 80.}
    assert observed["diagnostics"] == {"stage": "inference", "cuda_out_of_memory": True}
    warmup = worker.failed_case(case, failure, "warmup", None, is_oom=True)
    assert warmup["timings"] == {} and warmup["metrics"] == {}
    assert warmup["diagnostics"]["failed_window"] == {"elapsed_s": .25, "peak_allocated_mb": 42., "peak_reserved_mb": 80.}
    partial = worker.failed_case(case, failure, "inference", None, is_oom=True,
        wall_timings=worker.case_wall_timings(5., 5.5, 5., 10.75))
    assert partial["timings"] == {"inference_s": .25, "preprocessing_s": .5,
                                   "end_to_end_s": .75, "warmup_excluded_s": 5.}
    # A failed warmup excludes its actually observed partial interval, while a
    # preprocessing failure retains its partial CPU window with no invented infer time.
    failed_warmup = worker.failed_case(case, failure, "warmup", None, is_oom=True,
        wall_timings=worker.case_wall_timings(5., 5.5, 3., 8.5))
    assert failed_warmup["timings"] == {"preprocessing_s": .5, "end_to_end_s": .5, "warmup_excluded_s": 3.}
    preprocess = worker.failed_case(case, ValueError("synthetic invalid input"), "preprocessing", None, is_oom=False,
        wall_timings=worker.case_wall_timings(5., None, None, 5.25))
    assert preprocess["timings"] == {"preprocessing_s": .25, "end_to_end_s": .25}


def test_measured_case_seed_is_independent_of_consumed_warmup_rng():
    import random
    from types import SimpleNamespace
    from stylesphere.reconstruction.worker import seed_case
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_rng, cuda_rng = random.Random(), random.Random()
    torch = SimpleNamespace(manual_seed=cpu_rng.seed, cuda=SimpleNamespace(manual_seed_all=cuda_rng.seed))
    def draw():
        return random.random(), np.random.random(), cpu_rng.random(), cuda_rng.random()
    try:
        seed_case(torch, np, 8410)
        expected = draw()
        for _ in range(19):  # discarded warmup/other-case draws
            draw()
        seed_case(torch, np, 8410)
        assert draw() == expected
        seed_case(torch, np, 8411)
        assert draw() != expected
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_chamfer_hand_calculated_and_fscore():
    # A->B distances [0,2], B->A distances [0].
    metrics = geometry_metrics([[0, 0, 0], [2, 0, 0]], [[0, 0, 0]], [0.1, 2])
    assert metrics["chamfer_squared"] == 1
    assert metrics["chamfer_l1"] == .5
    assert metrics["precision_at_0.1"] == .5
    assert metrics["recall_at_0.1"] == 1
    assert metrics["fscore_at_0.1"] == pytest.approx(2 / 3)
    assert metrics["fscore_at_2"] == 1


def test_geometry_identical_points_and_empty_rejection():
    points = np.array([[0, 0, 0], [1, 1, 1]])
    assert geometry_metrics(points, points)["chamfer_squared"] == 0
    with pytest.raises(ValueError):
        geometry_metrics([], points)
    with pytest.raises(ValueError):
        geometry_metrics(points, points, [0])


def test_surface_sampling_uniform_triangle_expected_mean_and_seed():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    points = sample_surface(vertices, [[0, 1, 2]], 30000, seed=9)
    assert np.all(points >= 0)
    assert np.all(points[:, 0] + points[:, 1] <= 1)
    assert np.mean(points, axis=0) == pytest.approx([1 / 3, 1 / 3, 0], abs=.005)
    assert np.array_equal(points, sample_surface(vertices, [[0, 1, 2]], 30000, seed=9))


def test_surface_sampling_selects_by_area():
    # Disjoint right triangles, area ratio 1:4.
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]]
    samples = sample_surface(vertices, [[0, 1, 2], [3, 4, 5]], 20000, 2)
    assert np.mean(samples[:, 0] > 5) == pytest.approx(.8, abs=.01)


@pytest.mark.parametrize("vertices,faces", [([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]),
                                           ([[0, 0, 0]], [[0, 1, 2]]),
                                           ([[float("nan"), 0, 0]], [[0, 0, 0]])])
def test_surface_sampling_rejects_invalid_geometry(vertices, faces):
    with pytest.raises(ValueError):
        sample_surface(vertices, faces, 3)


def test_silhouette_iou_missing_union_and_shape():
    assert silhouette_iou([[1, 0], [1, 0]], [[1, 1], [0, 0]]) == pytest.approx(1 / 3)
    assert silhouette_iou([[0]], [[0]]) is None
    with pytest.raises(ValueError):
        silhouette_iou([[1]], [[1, 1]])


def test_render_similarity_exact_and_known_pixel_difference():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    result = rendered_similarity(a, a)
    assert result["render_exact_match"] is True
    assert result["render_psnr_db"] is None  # infinity never serialized as a JSON number.
    assert rendered_similarity(a, np.full_like(a, 255))["render_rgb_mae"] == 1


def test_invalid_glb_header_fails_before_optional_library_import(tmp_path):
    path = tmp_path / "invalid.glb"
    path.write_bytes(struct.pack("<4sII", b"nope", 2, 12))
    with pytest.raises(ValueError, match="Invalid GLB"):
        validate_glb(path)


def test_camera_poses_are_orthonormal_and_look_at_origin():
    cameras = standard_cameras()
    assert len(cameras) == 4
    for camera in cameras:
        pose = np.asarray(camera["camera_to_world"])
        assert pose[:3, :3].T @ pose[:3, :3] == pytest.approx(np.eye(3))
        assert pose[:3, 2] == pytest.approx(pose[:3, 3] / np.linalg.norm(pose[:3, 3]))


def test_glb_roundtrip_and_regeneration_from_stored_surfaces(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    import json
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    path = tmp_path / "synthetic_unit_box.glb"
    mesh.export(path)
    metrics = validate_glb(path)
    assert metrics["mesh_valid"] == 1
    assert metrics["mesh_watertight"] is True
    assert metrics["mesh_faces"] == 12
    points = sample_surface(mesh.vertices, mesh.faces, 100, 1)
    reference = points + [1, 0, 0]
    metrics.update(geometry_metrics(points, reference, [.01, 1]))
    metrics["surface_samples"] = len(points)
    surfaces = tmp_path / "synthetic_unit_samples.npz"
    np.savez(surfaces, predicted=points, reference=reference)
    protocol = tmp_path / "synthetic_unit_protocol.json"
    protocol.write_text(json.dumps({"fscore_thresholds": [.01, 1]}))
    record = {"status": "success", "metrics": metrics, "artifacts": {"mesh": str(path), "surface_samples": str(surfaces), "geometry_protocol": str(protocol)}}
    assert verify_record(record) == []
    record["metrics"]["chamfer_squared"] += 1
    assert any("chamfer_squared" in error for error in verify_record(record))
