"""Explicit size-checked checkpoint acquisition; inference adapters stay offline."""
import fnmatch
import os
import shutil
from pathlib import Path
from .io import sha256_file, utc_now, write_json

PATTERNS = {
    "fashion_siglip": ["open_clip_config.json", "open_clip_model.safetensors", "*token*", "spiece.model", "README.md", "LICENSE*"],
    "siglip2": ["*.json", "*.model", "model.safetensors", "README.md", "LICENSE*"],
    "fashion_tokenizer": ["*token*", "README.md", "LICENSE*"],
    "phi4_mini": ["*.safetensors", "*.json", "tokenizer*", "vocab*", "merges.txt", "chat_template*", "README.md", "LICENSE*"],
    "qwen35_4b": ["*.safetensors", "*.json", "tokenizer*", "vocab*", "merges.txt", "chat_template*", "README.md", "LICENSE*"],
    "spar3d": ["model.safetensors", "config.yaml", "README.md", "LICENSE*"],
    "sf3d": ["model.safetensors", "config.yaml", "README.md", "LICENSE*"],
}


def plan(config, model, destination):
    from huggingface_hub import HfApi
    if model not in PATTERNS:
        raise ValueError("Automated acquisition supports documented retrieval, assistance and reconstruction checkpoints")
    entry = config["models"][model] if model != "fashion_tokenizer" else {"model_id": "timm/ViT-B-16-SigLIP", "revision": "41f575766f40e752fdd1383e9565b7f02388c1c4"}
    info = HfApi().model_info(entry["model_id"], revision=entry["revision"], files_metadata=True)
    files = []
    for item in info.siblings:
        if not any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in PATTERNS[model]):
            continue
        lfs = item.lfs
        expected_sha256 = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        files.append({"path": item.rfilename, "size_bytes": item.size, "expected_sha256": expected_sha256,
                      "expected_git_blob_sha1": getattr(item, "blob_id", None) if not lfs else None})
    if not files or any(x["size_bytes"] is None for x in files):
        raise ValueError("Unknown file sizes; inspect official repository before downloading")
    size = sum(x["size_bytes"] for x in files)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    existing = sum(item["size_bytes"] for item in files if (destination / item["path"]).is_file() and (destination / item["path"]).stat().st_size == item["size_bytes"])
    return {"timestamp_utc": utc_now(), "model": model, "model_id": entry["model_id"], "resolved_revision": info.sha, "download_bytes": size,
            "existing_bytes_by_size": existing, "remaining_bytes_by_size": size - existing, "free_disk_bytes": free, "destination": str(destination), "files": files}


def acquire(root, config, model, *, approve_large=False, dry_run=False, ranged=False, range_workers=4, range_chunk_mib=8):
    root = Path(root)
    cache_dir = root / "models" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep acquisition/cache writes within the project; local_dir stores one copy
    # of each selected weight file and Hub metadata resumes partial transfers.
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    from huggingface_hub import snapshot_download
    destination = root / "models" / model
    evidence = plan(config, model, destination)
    evidence["cache_dir"] = str(cache_dir)
    write_json(root / "environment" / f"acquisition_plan_{model}.json", evidence)
    if dry_run:
        return evidence
    if evidence["download_bytes"] > 5_000_000_000 and not approve_large:
        raise ValueError("Download exceeds 5 GB. Obtain user approval, then use --approve-large-download")
    if evidence["free_disk_bytes"] < evidence["download_bytes"] * 2 + 2_000_000_000:
        raise RuntimeError("Insufficient free disk space for download and working copy")
    try:
        if ranged:
            from .ranged_download import download_plan
            recovered = download_plan(evidence, workers=range_workers, chunk_bytes=range_chunk_mib * 1024**2,
                                      log_path=root / "environment" / f"acquisition_ranges_{model}.jsonl", authenticated=model in {"spar3d", "sf3d"})
            evidence["transport"] = recovered["transport"]
        else:
            snapshot_download(evidence["model_id"], revision=evidence["resolved_revision"], local_dir=destination, cache_dir=cache_dir, allow_patterns=[x["path"] for x in evidence["files"]])
        evidence["sha256"] = {x["path"]: sha256_file(destination / x["path"]) for x in evidence["files"]}
        for item in evidence["files"]:
            if (destination / item["path"]).stat().st_size != item["size_bytes"]:
                raise ValueError(f"Downloaded size differs from official metadata: {item['path']}")
            if item["expected_sha256"] and evidence["sha256"][item["path"]] != item["expected_sha256"]:
                raise ValueError(f"Downloaded SHA256 differs from official LFS metadata: {item['path']}")
        evidence["status"] = "downloaded"
        evidence["completed_at_utc"] = utc_now()
    except Exception as exc:
        evidence.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)}, failed_at_utc=utc_now())
        write_json(root / "environment" / f"acquisition_{model}.json", evidence)
        raise
    write_json(root / "environment" / f"acquisition_{model}.json", evidence)
    return evidence
