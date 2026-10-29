"""Acquire only source-reviewed public initialisers used by the selected 3D models.

User approval covers the remaining 50-60 GB project budget. This script records
its own exact 2.76 GB subset, checks disk, uses two 8 MiB range workers and never
accepts gated-model conditions or performs GPU inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi
import requests

ASSETS = {
    "dinov2_large": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", ["config.json", "model.safetensors", "README.md", "LICENSE"]),
    "sf3d_clip": ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "1a25a446712ba5ee05982a381eed697ef9b435cf", ["open_clip_pytorch_model.bin", "open_clip_config.json", "README.md", "LICENSE"]),
}
ALPHA_URL = "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
ALPHA_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "hf", "alpha"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from stylesphere.common.ranged_download import download_plan
    from stylesphere.common.io import sha256_file
    api, plans = HfApi(), {}
    for key, (repo, revision, names) in ASSETS.items():
        info = api.model_info(repo, revision=revision, files_metadata=True, token=False)
        files = []
        for item in info.siblings:
            if item.rfilename in names:
                files.append({"path": item.rfilename, "size_bytes": item.size,
                              "expected_sha256": item.lfs.sha256 if item.lfs else None,
                              "expected_git_blob_sha1": item.blob_id if not item.lfs else None})
        required = "model.safetensors" if key == "dinov2_large" else "open_clip_pytorch_model.bin"
        if not any(item["path"] == required for item in files) or any(item["size_bytes"] is None for item in files):
            raise RuntimeError("Official source file/size differs from reviewed auxiliary plan")
        plans[key] = {"model_id": repo, "resolved_revision": revision, "destination": str(root / "models" / key), "files": files}
    head = requests.head(ALPHA_URL, timeout=30, allow_redirects=True)
    head.raise_for_status()
    alpha_size = int(head.headers["Content-Length"])
    if alpha_size != 934088680 or head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise RuntimeError("Official OpenAI CLIP file size or range support differs from inspected source")
    plans["spar3d_clip"] = {"model_id": "OpenAI/CLIP-ViT-L-14-336px", "resolved_revision": "ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff",
                            "destination": str(root / "models/spar3d_clip"), "source_identity": "Official URL plus SHA256, source revision records reviewed AlphaCLIP loader",
                            "files": [{"path": "ViT-L-14-336px.pt", "source_url": ALPHA_URL, "size_bytes": alpha_size, "expected_sha256": ALPHA_SHA}]}
    selected = {key: value for key, value in plans.items() if args.only == "all" or (key != "spar3d_clip") == (args.only == "hf")}
    total = sum(item["size_bytes"] for plan in selected.values() for item in plan["files"])
    free = shutil.disk_usage(root).free
    evidence = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "approval": "User approved remaining official assets within 50-60 GB project budget",
                "selected_bytes": total, "free_disk_before": free, "workers": 2, "range_chunk_bytes": 8 * 1024**2, "gpu_inference_performed": False, "plans": selected}
    (root / "environment" / f"3d-auxiliary-plan-{args.only}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if free < total * 2 + 2_000_000_000:
        raise RuntimeError("Insufficient working disk for auxiliary acquisition")
    print(json.dumps({"selected_bytes": total, "free_disk_before": free, "selected": list(selected)}, indent=2), flush=True)
    for key, plan in selected.items():
        result = download_plan(plan, workers=2, chunk_bytes=8 * 1024**2,
                               log_path=root / "environment" / f"acquisition-ranges-{key}.jsonl")
        receipt = {**plan, **result, "sha256": {item["path"]: sha256_file(Path(plan["destination"]) / item["path"]) for item in plan["files"]},
                   "free_disk_after": shutil.disk_usage(root).free, "gpu_inference_performed": False}
        (root / "environment" / f"acquisition_{key}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Verified {key}", flush=True)


if __name__ == "__main__":
    main()
