"""Small synthetic CPU fixtures for the supplementary geometry protocol."""
import numpy as np
import trimesh

from stylesphere.reconstruction.reference_benchmark import apply, normalized_mesh, proper_rotations, rigid_kabsch, rigid_icp


def test_cube_rotations_are_unique_and_proper():
    values = proper_rotations()
    assert len(values) == 24
    assert len({tuple(x.ravel()) for x in values}) == 24
    assert all(np.allclose(x.T @ x, np.eye(3)) and np.isclose(np.linalg.det(x), 1) for x in values)


def test_normalization_removes_uniform_size_and_translation_only():
    a = trimesh.creation.box(extents=[1, 2, 4])
    b = a.copy()
    b.apply_scale(3)
    b.apply_translation([7, -5, 2])
    an, _ = normalized_mesh(a)
    bn, _ = normalized_mesh(b)
    assert np.allclose(an.vertices, bn.vertices)
    assert np.isclose(np.linalg.norm(np.ptp(an.vertices, axis=0)), 1)
    # Aspect ratio is preserved, not independently scaled per axis.
    assert np.allclose(np.ptp(an.vertices, axis=0)[1:] / np.ptp(an.vertices, axis=0)[:-1], [2, 2])


def test_kabsch_recovers_rigid_motion_without_scale_or_reflection():
    source = np.random.default_rng(3).normal(size=(60, 3))
    target = source @ proper_rotations()[7].T + [0.3, -0.1, 0.2]
    transform = rigid_kabsch(source, target)
    assert np.allclose(apply(source, transform), target)
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)
    mirrored = source * [-1, 1, 1]
    fitted = rigid_kabsch(source, mirrored)
    assert np.isclose(np.linalg.det(fitted[:3, :3]), 1)
    assert not np.allclose(apply(source, fitted), mirrored)


def test_fixed_multistart_icp_recovers_asymmetric_fixture():
    source = np.random.default_rng(9).uniform([-0.4, -0.2, -0.1], [0.4, 0.2, 0.1], size=(160, 3))
    target = source @ proper_rotations()[16].T + [0.05, 0.02, -0.03]
    transform, evidence = rigid_icp(source, target, max_iterations=50, tolerance=1e-9)
    assert evidence["chosen_loss"] < 1e-18
    assert len(evidence["trials"]) == 24
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1)
    assert np.max(np.linalg.svd(transform[:3, :3])[1]) < 1.0000001
    assert all(x["iterations"] <= 50 for x in evidence["trials"])
