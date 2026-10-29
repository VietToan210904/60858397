"""Offline preflight tests use only explicitly synthetic files in tmp_path."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stylesphere.reconstruction import worker


def write(path, data=b"SYNTHETIC UNIT TEST ASSET"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_checked_asset_requires_existing_exact_hash(tmp_path):
    digest = write(tmp_path / "asset.bin")
    assert worker.checked_asset(tmp_path, "asset.bin", digest, "unit test") == tmp_path / "asset.bin"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        worker.checked_asset(tmp_path, "asset.bin", "0" * 64, "unit test")
    with pytest.raises(FileNotFoundError, match="automatic auxiliary downloads are disabled"):
        worker.checked_asset(tmp_path, "missing.bin", digest, "unit test")


def test_native_preflight_verifies_receipt_patch_and_binary(tmp_path, monkeypatch):
    weights = tmp_path / "models/sf3d"
    weights.mkdir(parents=True)
    patch_sha = write(tmp_path / "environment/test.patch")
    receipt = {"source_revision": "synthetic-commit", "extensions": [{"returncode": 0, "compatibility_patch": {
        "patch_sha256": patch_sha, "patch_path": "environment/test.patch"}}]}
    receipt_path = tmp_path / "environment/receipt.json"
    receipt_sha = write(receipt_path, json.dumps(receipt).encode())
    binary_sha = write(tmp_path / "binary.pyd")
    config = {"source_revision": "synthetic-commit", "extension_build": {"receipt": "../../environment/receipt.json", "receipt_sha256": receipt_sha,
              "patch_sha256": patch_sha, "binary_sha256": {"texture_baker._C": binary_sha, "uv_unwrapper._C": binary_sha}}}
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda module: SimpleNamespace(origin=str(tmp_path / "binary.pyd")))
    assert worker.verify_native_build(config, weights)["patch_sha256"] == patch_sha
    write(tmp_path / "binary.pyd", b"CHANGED SYNTHETIC BYTES")
    with pytest.raises(ValueError, match="differs from the recorded build"):
        worker.verify_native_build(config, weights)


def test_offline_paths_preserve_architecture_and_exact_initializers(tmp_path):
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    weights = tmp_path / "models/sf3d"
    weights.mkdir(parents=True)
    native = {"image_tokenizer": {"pretrained_model_name_or_path": "facebook/dinov2-large", "width": 512},
              "image_estimator": {"model": "ViT-B-32", "pretrain": "laion2b_s34b_b79k"}, "architecture_sentinel": "unchanged"}
    OmegaConf.save(OmegaConf.create(native), weights / "config.yaml")
    dino_model_sha = write(tmp_path / "models/dinov2/model.safetensors")
    dino_config_sha = write(tmp_path / "models/dinov2/config.json", b'{}')
    clip_sha = write(tmp_path / "models/clip/open_clip_pytorch_model.bin")
    config = {"auxiliary": {"dinov2": {"path": "../dinov2", "revision": "synthetic-revision", "model_sha256": dino_model_sha, "config_sha256": dino_config_sha},
                            "open_clip": {"path": "../clip/open_clip_pytorch_model.bin", "revision": "synthetic-revision", "sha256": clip_sha}}}
    path, evidence = worker.offline_initialisation_config(config, weights, "sf3d", tmp_path / "operational.yaml")
    derived = OmegaConf.load(path)
    assert derived.architecture_sentinel == "unchanged"
    assert derived.image_tokenizer.width == 512
    assert Path(derived.image_tokenizer.pretrained_model_name_or_path) == tmp_path / "models/dinov2"
    assert derived.image_estimator.model == "ViT-B-32"
    assert evidence["open_clip"]["sha256"] == clip_sha
    # SPAR constructs two independently configured DINO tokenizers. Both must
    # use the same verified local initializer without changing other settings.
    point_diffusion = OmegaConf.create({"image_tokenizer": native["image_tokenizer"],
        "pdiff_image_tokenizer_cls": "spar3d.models.tokenizers.image.DINOV2SingleImageTokenizer",
        "pdiff_image_tokenizer": {"pretrained_model_name_or_path": "facebook/dinov2-large", "width": 512},
        "inference_time_steps": 32})
    redirected = worker.redirect_dinov2_initializers(point_diffusion, "spar3d", tmp_path / "models/dinov2")
    assert len(redirected) == 2
    assert point_diffusion.image_tokenizer.pretrained_model_name_or_path == point_diffusion.pdiff_image_tokenizer.pretrained_model_name_or_path == str(tmp_path / "models/dinov2")
    assert point_diffusion.pdiff_image_tokenizer.width == 512 and point_diffusion.inference_time_steps == 32
    point_diffusion.image_tokenizer.pretrained_model_name_or_path = "facebook/dinov2-large"
    point_diffusion.pdiff_image_tokenizer.pretrained_model_name_or_path = "UNAPPROVED SUBSTITUTE"
    with pytest.raises(ValueError, match="without substituting a backbone"):
        worker.redirect_dinov2_initializers(point_diffusion, "spar3d", tmp_path / "models/dinov2")
    native["image_tokenizer"]["pretrained_model_name_or_path"] = "UNAPPROVED SUBSTITUTE"
    OmegaConf.save(OmegaConf.create(native), weights / "config.yaml")
    with pytest.raises(ValueError, match="without substituting a backbone"):
        worker.offline_initialisation_config(config, weights, "sf3d", path)


def test_network_guard_is_explicit_and_leaves_file_events_allowed():
    worker.deny_network("open", ("local-test",))
    with pytest.raises(RuntimeError, match="offline 3D inference"):
        worker.deny_network("socket.connect", ())


def test_checkpoint_preflight_requires_original_config_and_final_weights(tmp_path):
    native = b'SYNTHETIC ORIGINAL CONFIG FOR UNIT TEST ONLY'
    weights = b'SYNTHETIC FINAL WEIGHTS FOR UNIT TEST ONLY'
    write(tmp_path / "config.yaml", native)
    weights_sha = write(tmp_path / "model.safetensors", weights)
    blob = hashlib.sha1(b"blob " + str(len(native)).encode() + b"\0" + native).hexdigest()
    config = {"revision": "a" * 40, "checkpoint_files": {
        "config.yaml": {"size_bytes": len(native), "git_blob_sha1": blob},
        "model.safetensors": {"size_bytes": len(weights), "sha256": weights_sha}}}
    evidence = worker.verify_checkpoint(config, tmp_path)
    assert evidence["files"]["model.safetensors"]["sha256"] == weights_sha
    write(tmp_path / "config.yaml", b"X" * len(native))
    with pytest.raises(ValueError, match="Git blob differs"):
        worker.verify_checkpoint(config, tmp_path)
    write(tmp_path / "config.yaml", native)
    write(tmp_path / "model.safetensors", b"X" * len(weights))
    with pytest.raises(ValueError, match="SHA256 differs"):
        worker.verify_checkpoint(config, tmp_path)
