"""Record a newly built local 3D extension profile without loading model weights.

Run inside the corresponding isolated model environment after building. This
updates expected local binary/receipt hashes, never source or checkpoint identity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch  # Required before loading the compiled extensions on Windows.
import texture_baker
import uv_unwrapper
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["spar3d", "sf3d"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config/reconstruction-local.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["models"][args.model]
    receipt_path = root / "environment" / f"native-build-{args.model}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["source_revision"] != model["source_revision"] or any(row["returncode"] != 0 for row in receipt["extensions"]):
        raise RuntimeError("Native build must succeed against the exact reviewed official source commit")
    patches = [row["compatibility_patch"] for row in receipt["extensions"] if row.get("compatibility_patch")]
    expected_patch = "c59b6e46066c0f60cf9b566b14f963366b98cd39fd3a17af26f22eb29509667d"
    if not patches or any(row["patch_sha256"] != expected_patch for row in patches):
        raise RuntimeError("Expected reviewed include-only compatibility patch is not recorded")
    binaries = {name: digest(importlib.util.find_spec(name).origin) for name in ("texture_baker._C", "uv_unwrapper._C")}
    model["extension_build"] = {"receipt": f"../../environment/native-build-{args.model}.json", "receipt_sha256": digest(receipt_path),
                                "patch_sha256": expected_patch, "binary_sha256": binaries}
    config_path.write_text("# Local native build hashes; official source/checkpoint/auxiliary identities remain unchanged.\n" + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"model": args.model, "recorded_binary_sha256": binaries, "model_inference_performed": False}, indent=2))


if __name__ == "__main__":
    main()
