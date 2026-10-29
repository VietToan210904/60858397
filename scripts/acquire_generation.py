"""Acquire only explicitly approved official generation assets, with receipts.

Requires the user's approval recorded by the calling workflow. This script never
downloads a second SDXL base UNet, BF16 FLUX, FP4 encoders or unrelated variants.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi

ASSETS = {
    "sdxl_lightning": ("ByteDance/SDXL-Lightning", "c9a24f48e1c025556787b0c58dd67a091ece2e44", ["sdxl_lightning_4step_unet.safetensors", "README.md", "LICENSE*"]),
    "sdxl_base_components": ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b", ["model_index.json", "scheduler/*.json", "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors", "tokenizer/*", "tokenizer_2/*", "unet/config.json", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "README.md", "LICENSE*"]),
    "flux2_klein_fp8": ("black-forest-labs/FLUX.2-klein-4b-fp8", "5b4408e59397a4a37ccb46afe426d8ed86379441", ["flux-2-klein-4b-fp8.safetensors", "README.md", "LICENSE*"]),
    "flux2_klein_components": ("Comfy-Org/flux2-klein-4B", "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965", ["split_files/text_encoders/qwen_3_4b.safetensors", "split_files/vae/flux2-vae.safetensors", "README.md", "LICENSE*"]),
}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=4)
    parser.add_argument("--chunk-mib", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=8)
    parser.add_argument("--assets", nargs="+", choices=tuple(ASSETS), default=list(ASSETS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from stylesphere.common.ranged_download import download_plan
    cache, evidence = root / "models" / "hub", root / "environment"
    cache.mkdir(parents=True, exist_ok=True)
    plans = {}
    api = HfApi()
    for key, (repo, revision, patterns) in ASSETS.items():
        if key not in args.assets:
            continue
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = []
        for entry in info.siblings:
            if any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in patterns):
                lfs = entry.lfs
                files.append({"path": entry.rfilename, "bytes": entry.size,
                              "official_lfs_sha256": getattr(lfs, "sha256", None) if lfs else None})
        if not files or any(f["bytes"] is None for f in files):
            raise ValueError(f"Unknown selected size for {key}")
        plans[key] = {"model_id": repo, "revision": info.sha, "files": files, "expected_bytes": sum(f["bytes"] for f in files)}
    total = sum(p["expected_bytes"] for p in plans.values())
    free = shutil.disk_usage(cache).free
    plan = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "approval": "User authorized remaining official checkpoint/dependency downloads up to 50-60 GB; no drivers/toolkits", "total_selected_bytes": total, "free_bytes_before": free, "assets": plans}
    plan_name = "generation-acquisition-plan" + ("" if set(args.assets) == set(ASSETS) else "-" + "_".join(sorted(args.assets))) + ".json"
    (evidence / plan_name).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if free < total * 2 + 10_000_000_000:
        raise RuntimeError("Insufficient free disk for generation acquisition and working storage")
    print(f"Selected exact generation assets: {total:,} bytes; free disk: {free:,} bytes", flush=True)
    for key, entry in plans.items():
        print(f"Acquiring/resuming {key}: {entry['expected_bytes']:,} bytes", flush=True)
        snapshot = cache / ("models--" + entry["model_id"].replace("/", "--")) / "snapshots" / entry["revision"]
        transport = download_plan({"model_id": entry["model_id"], "resolved_revision": entry["revision"], "destination": str(snapshot),
                                   "files": [{"path": f["path"], "size_bytes": f["bytes"], "expected_sha256": f["official_lfs_sha256"]} for f in entry["files"]]},
                                  workers=args.workers, chunk_bytes=args.chunk_mib * 1024**2, log_path=evidence / f"generation-ranges-{key}.jsonl")
        for record in entry["files"]:
            path = snapshot / record["path"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"File size mismatch: {key}/{record['path']}")
            record["sha256"] = sha256(path)
            if record["official_lfs_sha256"] and record["sha256"] != record["official_lfs_sha256"]:
                raise ValueError(f"Official SHA256 mismatch: {key}/{record['path']}")
        receipt = {**entry, "status": "downloaded_and_hash_verified", "snapshot": str(snapshot.relative_to(root)),
                   "transport": transport["transport"],
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        (evidence / f"acquisition_{key}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"Verified {key}: {snapshot}", flush=True)


if __name__ == "__main__":
    main()
