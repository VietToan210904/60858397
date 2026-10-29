"""Freeze 20 real ABO structured-product photos with CPU-only common matting.

Source cases are selected from the actual frozen retrieval subset, independently
of model outputs. No reference geometry/camera eligibility or human ratings are
invented. The retained selection plan and QA ledger document every exclusion.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy import ndimage


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "datasets/reconstruction"
    folder.mkdir(parents=True, exist_ok=True)
    source = root / "datasets/retrieval/abo_metadata.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    target_counts = {"BOOT": 5, "SHOES": 5, "HANDBAG": 10}
    plan_path = folder / "selection_plan.json"
    if (folder / "source_objects.jsonl").exists():
        raise SystemExit("Frozen source_objects.jsonl already exists; preserve it rather than reselection")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["source_metadata_sha256"] != sha(source):
            raise RuntimeError("Source catalogue changed after selection plan freeze")
    else:
        candidates = {}
        for category in target_counts:
            cases = sorted([row for row in records if row["category"] == category], key=lambda row: row["source_id"])
            random.Random(8410).shuffle(cases)
            candidates[category] = [row["source_id"] for row in cases]
        plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 8410, "source_metadata_sha256": sha(source),
                "source_scope": "600 retrieval products selected from official ABO listings_0 shard only",
                "selection": "Sort source_id, then seed8410 shuffle within each exact category; take first QA-eligible cases until quota",
                "targets": target_counts, "candidate_order": candidates, "model_outputs_consulted": False,
                "quality_rules": {"minimum_source_side_px": 256, "foreground_fraction_min": 0.05, "foreground_fraction_max": 0.90,
                                  "largest_component_fraction_min": 0.85, "largest_secondary_component_fraction_max": 0.15,
                                  "maximum_input_side_px": 1024, "alpha_threshold": 127},
                "human_quality_review": "pending; deterministic mask eligibility is not a human quality rating",
                "reference_geometry_or_cameras": "unavailable; all geometry and matching-view metric flags false"}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.environ["U2NET_HOME"] = str(root / "models/background_removal")
    import rembg
    model = root / "models/background_removal/u2netp.onnx"
    if hashlib.md5(model.read_bytes()).hexdigest() != "8e83ca70e441ab06c318d82300c84806":
        raise RuntimeError("Reviewed U2NetP model is unavailable or changed")
    session = rembg.new_session("u2netp", providers=["CPUExecutionProvider"])
    if session.inner_session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Only CPU background preparation is authorised by this command")
    by_id = {row["source_id"]: row for row in records}
    retained, ledger, total_transferred = [], [], 0
    for category, quota in plan["targets"].items():
        accepted = 0
        for source_id in plan["candidate_order"][category]:
            if accepted == quota:
                break
            row = by_id[source_id]
            entry = {"source_id": source_id, "category": category, "outcome": "excluded", "reasons": []}
            name = row.get("attributes", "")
            if re.search(r"\b(?:pack of|lot of|set of|[2-9][ -]pack)\b", name, flags=re.I):
                entry["reasons"].append("metadata indicates a multi-item package")
                ledger.append(entry)
                continue
            url = row["source_url"].replace("/images/small/", "/images/original/")
            if not url.startswith("https://amazon-berkeley-objects.s3.amazonaws.com/images/original/"):
                raise RuntimeError("Unexpected source image origin")
            original = folder / "assets/original" / f"{source_id}.jpg"
            original.parent.mkdir(parents=True, exist_ok=True)
            response_meta = {}
            if not original.exists():
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                chunks = []
                for chunk in response.iter_content(65536):
                    total_transferred += len(chunk)
                    if total_transferred > 100_000_000:
                        raise RuntimeError("100 MB image acquisition cap reached")
                    chunks.append(chunk)
                original.write_bytes(b"".join(chunks))
                response_meta = {"etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
            with Image.open(original) as decoded:
                rgb = decoded.convert("RGB")
            entry.update(original_sha256=sha(original), original_size=list(rgb.size), source_url=url, **response_meta)
            if min(rgb.size) < 256:
                entry["reasons"].append("source image shorter side below 256 pixels")
                ledger.append(entry)
                continue
            original_size = list(rgb.size)
            rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            rgba = rembg.remove(rgb, session=session, alpha_matting=False, post_process_mask=False).convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            foreground = alpha > 127
            fraction = float(np.mean(foreground))
            labels, count = ndimage.label(foreground)
            areas = np.bincount(labels.ravel())[1:]
            areas = np.sort(areas)[::-1]
            largest_fraction = float(areas[0] / areas.sum()) if len(areas) else 0.0
            secondary_fraction = float(areas[1] / areas.sum()) if len(areas) > 1 else 0.0
            entry["mask_qa"] = {"foreground_fraction": fraction, "largest_component_fraction": largest_fraction,
                                "largest_secondary_component_fraction": secondary_fraction, "component_count": int(count)}
            if not 0.05 <= fraction <= 0.90:
                entry["reasons"].append("foreground coverage outside predeclared [0.05,0.90]")
            if largest_fraction < 0.85 or secondary_fraction > 0.15:
                entry["reasons"].append("multiple major disconnected foreground regions; possible multi-object photo or mask fragmentation")
            if alpha.min() != 0 or alpha.max() < 250:
                entry["reasons"].append("insufficient transparent/opaque alpha range")
            if entry["reasons"]:
                ledger.append(entry)
                (folder / "quality_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
                continue
            object_id = "abo-" + source_id
            prepared = folder / "assets/prepared" / f"{object_id}.png"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            rgba.save(prepared)
            metadata_path = folder / f"{object_id}-preparation.json"
            metadata = {"original_image": original.relative_to(root).as_posix(), "original_sha256": sha(original), "original_size": original_size,
                        "source_url": url, "source_dataset": "Amazon Berkeley Objects", "source_id": source_id,
                        "original_source_metadata_sha256": plan["source_metadata_sha256"], "prepared_image_sha256": sha(prepared),
                        "prepared_size": list(rgba.size), "max_input_side": 1024, "resize_filter": "PIL LANCZOS",
                        "background_removal": {"library": "rembg", "version": "2.0.57", "model": "u2netp", "model_sha256": sha(model),
                                               "providers": ["CPUExecutionProvider"], "alpha_matting": False, "post_process_mask": False},
                        "mask_qa": entry["mask_qa"], "human_review_status": "pending", "input_kind": "real_catalogue_photo", "synthetic": False,
                        "reference_geometry_available": False, "camera_calibration_available": False, "alignment_known": False}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            retained.append({"object_id": object_id, "source_image": prepared.relative_to(root).as_posix(), "category": category,
                             "source_dataset": "Amazon Berkeley Objects", "reference_mesh": "", "camera_info": "",
                             "preparation_metadata": metadata_path.relative_to(root).as_posix(), "geometry_eligible": "false", "silhouette_eligible": "false",
                             "render_similarity_eligible": "false", "licence": "CC-BY-4.0", "source_revision": row["source_revision"], "redistributable": "true"})
            entry["outcome"] = "retained"
            entry["prepared_image_sha256"] = sha(prepared)
            ledger.append(entry)
            accepted += 1
            (folder / "quality_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        if accepted < quota:
            raise RuntimeError(f"Only {accepted} eligible {category} cases; quota {quota} not reached")
    licence_path = folder / "SOURCE_LICENSE_CC_BY_4_0.txt"
    if not licence_path.exists():
        licence_response = requests.get("https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt", timeout=45)
        licence_response.raise_for_status()
        if hashlib.sha256(licence_response.content).hexdigest() != "419896aea50c15d6e40c5b4baf4bd346f78223b9a354c7357ad00262afdc08ec":
            raise RuntimeError("ABO licence bytes changed from reviewed source; inspect before redistribution")
        licence_path.write_bytes(licence_response.content)
    if sha(licence_path) != "419896aea50c15d6e40c5b4baf4bd346f78223b9a354c7357ad00262afdc08ec":
        raise RuntimeError("Retained ABO licence bytes differ from the reviewed source")
    rights = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "licence": "CC-BY-4.0",
              "official_licence_url": "https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt", "licence_sha256": sha(licence_path),
              "asset_attribution": "Amazon.com",
              "dataset_attribution": "Matthieu Guillaumin; Thomas Dideriksen; Kenan Deng; Himanshu Arora; Arnab Dhua; Xi (Brian) Zhang; Tomas Yago-Vicente; Jasmine Collins; Shubham Goel; Jitendra Malik",
              "dataset_attribution_source": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html#attribution",
              "publication_citation": "Collins et al., ABO: Dataset and Benchmarks for Real-World 3D Object Understanding, CVPR 2022",
              "changes": "Original S3 photos resized preserving aspect ratio and matted using CPU rembg2.0.57/U2NetP; originals retained",
              "redistributable": True, "scope": "The selected original photos and corresponding prepared RGBA derivatives; no model weights",
              "objects": [row["object_id"] for row in retained]}
    (folder / "RIGHTS_AND_ATTRIBUTION.json").write_text(json.dumps(rights, indent=2) + "\n", encoding="utf-8")
    (folder / "source_objects.jsonl").write_text("".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8")
    from stylesphere.datasets import prepare
    manifest = prepare(root, "reconstruction", {"metadata_path": "datasets/reconstruction/source_objects.jsonl", "object_count": 20}, seed=8410)
    receipt = {"source_images_transferred_bytes": total_transferred, "objects_retained": len(retained), "candidates_screened": len(ledger),
               "excluded": [row for row in ledger if row["outcome"] == "excluded"], "selection_plan_sha256": sha(plan_path),
               "all_geometry_and_view_metrics_ineligible": True, "gpu_inference_performed": False,
               "manifest_sha256": sha(folder / "manifest.json"), "case_count": manifest["details"]["object_count"]}
    (root / "environment/abo_reconstruction_preparation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
