"""Bind one blinded human worksheet to both models' genuine benchmark images."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from stylesphere.common.io import canonical_hash, portable_path, read_jsonl, sha256_file, utc_now, write_json
from .review import RATING_FIELDS, export_review
from .provenance import verify_prompt_binding

MODEL_KEYS = {"sdxl_lightning", "flux2_klein"}


def has_human_work(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"review_id", "rater_id", *RATING_FIELDS, "usable", "error_categories", "notes"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("Existing canonical ratings file has an invalid header; refuse replacement")
        return any(any(value is not None and str(value).strip() for key, value in row.items() if key != "review_id") for row in reader)


def export_canonical_review(root: Path, output_dir: Path, seed: int = 8410, protocol_sha256: str | None = None,
                            *, _validate_only: bool = False) -> dict:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("Portable canonical review requires output_dir inside the project root")
    records = read_jsonl(output_dir / "generation" / "records.jsonl")
    candidates = [r for r in records if r.get("status") == "success" and r.get("mode") == "evaluation"
                  and r.get("provenance") == "model_inference" and r.get("model") in MODEL_KEYS and r.get("protocol_sha256")]
    protocols = {}
    for row in candidates:
        protocols.setdefault(row["protocol_sha256"], {}).setdefault(row["model"], set()).add((row["case_id"], row["seed"], row["input_sha256"]))
    matched = [key for key, models in protocols.items() if set(models) == MODEL_KEYS
               and models["sdxl_lightning"] & models["flux2_klein"]]
    if protocol_sha256 is not None:
        if protocol_sha256 not in matched:
            raise ValueError("Requested generation protocol has no matched successful benchmark inputs from both models")
    elif len(matched) != 1:
        raise ValueError("Require one matching generation protocol from both models; use --protocol-sha256 when several exist")
    else:
        protocol_sha256 = matched[0]
    latest = {}
    for row in candidates:
        if row["protocol_sha256"] == protocol_sha256:
            latest[(row["model"], row["case_id"], row["seed"], row["input_sha256"])] = row
    selected, bindings = [], []
    for key, row in sorted(latest.items()):
        input_errors = verify_prompt_binding(row, root, output_dir / "generation")
        if input_errors:
            raise ValueError("Selected review input provenance is invalid: " + "; ".join(input_errors))
        image = portable_path(root, row["artifacts"]["image"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError("A selected source image is absent or outside the project root")
        expected = row.get("artifact_sha256", {}).get("image")
        if not expected or sha256_file(image) != expected:
            raise ValueError("A selected source image does not match its inference-record SHA256")
        bindings.append({"model": row["model"], "case_id": row["case_id"], "seed": row["seed"], "run_id": row["run_id"],
                         "input_sha256": row["input_sha256"], "config_sha256": row["config_sha256"],
                         "source_image": image.relative_to(root).as_posix(), "source_image_sha256": expected})
        selected.append({**row, "artifacts": {**row["artifacts"], "image": image.relative_to(root).as_posix()}})
    selection_sha = canonical_hash({"protocol_sha256": protocol_sha256, "seed": seed, "source_bindings": bindings})
    destination = output_dir / "generation" / "human_review"
    ratings_path, manifest_path = destination / "ratings.csv", destination / "manifest.json"
    human_work = has_human_work(ratings_path)
    if human_work or _validate_only:
        if not manifest_path.is_file():
            raise ValueError("Human work exists without a bound manifest; refuse to overwrite or remap it")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("selection_sha256") != selection_sha:
            raise ValueError("Canonical human work is frozen: source selection changed; refuse overwrite or mapping changes")
        if sha256_file(destination / "private_mapping.json") != manifest["private_mapping_sha256"]:
            raise ValueError("Frozen private mapping changed after human work was entered")
        mapping = json.loads((destination / "private_mapping.json").read_text(encoding="utf-8"))
        with ratings_path.open(newline="", encoding="utf-8-sig") as stream:
            if any(row["review_id"] not in {entry["review_id"] for entry in mapping} for row in csv.DictReader(stream)):
                raise ValueError("Human worksheet contains a review ID absent from its frozen mapping")
        for asset in manifest["blinded_images"]:
            path = root / asset["image"]
            if not path.is_file() or sha256_file(path) != asset["sha256"]:
                raise ValueError("Frozen blinded image changed after human work was entered")
        return {"status": "existing_frozen_review" if human_work else "existing_validated_review", "images": len(bindings), "directory": destination.relative_to(root).as_posix(),
                "protocol_sha256": protocol_sha256, "selection_sha256": selection_sha}
    # Only an entirely blank worksheet may be rebuilt. No input image or
    # inference record is changed. Existing unlisted files are preserved.
    previous_assets = []
    if manifest_path.is_file():
        previous_assets = json.loads(manifest_path.read_text(encoding="utf-8")).get("blinded_images", [])
    if ratings_path.exists():
        ratings_path.unlink()
    export_review(selected, destination, seed, root=root)
    mapping_path = destination / "private_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_lookup = {(b["run_id"], b["model"], b["case_id"], b["seed"]): b for b in bindings}
    blinded = []
    for row in mapping:
        binding = source_lookup[(row["run_id"], row["model"], row["case_id"], row["seed"])]
        asset = destination / "images" / (row["review_id"] + ".png")
        row.update(source_image_sha256=binding["source_image_sha256"], input_sha256=binding["input_sha256"],
                   protocol_sha256=protocol_sha256, blinded_image=asset.relative_to(root).as_posix(), blinded_image_sha256=sha256_file(asset))
        blinded.append({"review_id": row["review_id"], "image": row["blinded_image"], "sha256": row["blinded_image_sha256"]})
    write_json(mapping_path, mapping)
    keep = {entry["image"] for entry in blinded}
    for previous in previous_assets:
        path = (root / previous["image"]).resolve()
        if path.is_relative_to((destination / "images").resolve()) and previous["image"] not in keep and path.is_file():
            path.unlink()
    common_inputs = protocols[protocol_sha256]["sdxl_lightning"] & protocols[protocol_sha256]["flux2_klein"]
    manifest = {"schema_version": 1, "created_at_utc": utc_now(), "protocol_sha256": protocol_sha256,
                "selection_sha256": selection_sha, "seed": seed, "n_images": len(bindings), "n_matched_inputs": len(common_inputs),
                "successful_images_by_model": {model: sum(b["model"] == model for b in bindings) for model in sorted(MODEL_KEYS)},
                "source_bindings": bindings, "blinded_images": blinded, "private_mapping_sha256": sha256_file(mapping_path),
                "usable_rule": "unanimous nonblank image votes; disagreement missing pending documented human adjudication"}
    write_json(manifest_path, manifest)
    return {"status": "exported_blank_review", "images": len(bindings), "directory": destination.relative_to(root).as_posix(),
            "protocol_sha256": protocol_sha256, "selection_sha256": selection_sha}


def validate_canonical_review(root: Path, output_dir: Path) -> dict:
    """Read-only report guard, checking current records, source/copy SHA and mapping."""
    manifest_path = Path(output_dir) / "generation" / "human_review" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return export_canonical_review(root, output_dir, manifest["seed"], manifest["protocol_sha256"], _validate_only=True)
