"""Verify local 3D source/native/initializer bytes without loading models or using CUDA.

Run inside the corresponding model environment. Missing gated main files are
reported as pending; this command never acquires files or accesses the network.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")

import torch  # Loads native runtime DLLs before extension inspection on Windows.
import yaml

from stylesphere.reconstruction.worker import checked_asset, verify_checkpoint, verify_native_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sf3d", "spar3d"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = yaml.safe_load((root / "config/reconstruction-local.yaml").read_text(encoding="utf-8"))["models"][args.model]
    model["revision"] = yaml.safe_load((root / "config/models.yaml").read_text(encoding="utf-8"))["models"][args.model]["revision"]
    weights, checkout = (root / model["weights_path"]).resolve(), (root / model["checkout"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    if revision != model["source_revision"] or dirty:
        raise RuntimeError("Official 3D checkout differs from the reviewed clean commit")
    native = verify_native_build(model, weights)
    auxiliary = model["auxiliary"]
    dino = auxiliary["dinov2"]
    checked_asset(weights, str(Path(dino["path"]) / "model.safetensors"), dino["model_sha256"], "DINOv2 weights")
    checked_asset(weights, str(Path(dino["path"]) / "config.json"), dino["config_sha256"], "DINOv2 config")
    clip = auxiliary["open_clip" if args.model == "sf3d" else "alpha_clip"]
    checked_asset(weights, clip["path"], clip["sha256"], "CLIP initializer")
    missing = [name for name in model["checkpoint_files"] if not (weights / name).is_file()]
    checkpoint = {"status": "pending_authorized_local_acquisition", "missing_files": missing} if missing else verify_checkpoint(model, weights)
    report = {"model": args.model, "source_revision": revision, "source_dirty": dirty,
              "torch": torch.__version__, "compiled_cuda": torch.version.cuda,
              "native_build": native, "initializer_hashes_verified": True, "checkpoint": checkpoint,
              "model_weights_loaded": False, "gpu_inference_performed": False, "network_access_performed": False}
    target = root / "environment" / f"audit-local-3d-prerequisites-{args.model}.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
