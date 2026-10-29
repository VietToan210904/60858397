"""Verify retained generation artifacts and exact input provenance on CPU."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylesphere.common.io import portable_path, read_jsonl, sha256_file, utc_now, write_json
from stylesphere.generation.provenance import bind_existing_run, verify_prompt_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--deployment-freeze", type=Path)
    parser.add_argument("--source-prompts", type=Path, default=Path("datasets/generation/prompts.jsonl"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    pair = root / "results" / "generation"
    records = [row for row in read_jsonl(pair / "records.jsonl") if row["run_id"] == args.run_id]
    if not records:
        raise ValueError("No records exist for the requested run")
    before = sha256_file(pair / "records.jsonl")
    sidecar = None
    if args.deployment_freeze:
        freeze = portable_path(root, args.deployment_freeze)
        expected = json.loads(freeze.read_text(encoding="utf-8"))["dataset_sha256"]
        sidecar = bind_existing_run(root, pair, args.run_id, portable_path(root, args.source_prompts),
                                    expected_sha256=expected, deployment_freeze=freeze)
    errors = []
    from PIL import Image
    for row in records:
        prefix = f"{row['model']}/{row['case_id']}/{row['seed']}"
        errors.extend(f"{prefix}: {message}" for message in verify_prompt_binding(row, root, pair))
        if row["status"] != "success":
            continue
        for key, value in row.get("artifacts", {}).items():
            path = portable_path(root, value)
            if not path.is_file() or sha256_file(path) != row.get("artifact_sha256", {}).get(key):
                errors.append(f"{prefix}: artifact SHA256 mismatch ({key})")
        with Image.open(portable_path(root, row["artifacts"]["image"])) as image:
            if image.size != (row["generation_config"]["width"], row["generation_config"]["height"]):
                errors.append(f"{prefix}: unexpected image size")
            image.verify()
    ratings_path = pair / args.run_id / "review" / "ratings.csv"
    with ratings_path.open(newline="", encoding="utf-8-sig") as stream:
        ratings = list(csv.DictReader(stream))
    grades_blank = not any(str(value).strip() for row in ratings for key, value in row.items() if key != "review_id")
    windows = [row["timings"]["inference_s"] for row in records if isinstance(row.get("timings", {}).get("inference_s"), (int, float))]
    summary = {"checked_at_utc": utc_now(), "run_id": args.run_id, "n_records": len(records),
               "success": sum(row["status"] == "success" for row in records),
               "failed": sum(row["status"] == "failed" for row in records),
               "blocked": sum(row["status"] == "blocked" for row in records),
               "n_prompts": len({row["case_id"] for row in records}), "seeds": sorted({row["seed"] for row in records}),
               "protocol_sha256": sorted({row.get("protocol_sha256") for row in records}),
               "all_human_review_fields_blank": grades_blank,
               "raw_records_sha256_before": before, "raw_records_sha256_after": sha256_file(pair / "records.jsonl"),
               "prompt_binding": sidecar.relative_to(root).as_posix() if sidecar else None,
               "prompt_binding_sha256": sha256_file(sidecar) if sidecar else None,
               "inference_s": {"total": sum(windows), "mean": statistics.mean(windows), "median": statistics.median(windows),
                               "min": min(windows), "max": max(windows)} if windows else {},
               "peak_allocated_mb": max((row["metrics"].get("peak_allocated_mb", 0) for row in records), default=None),
               "peak_reserved_mb": max((row["metrics"].get("peak_reserved_mb", 0) for row in records), default=None),
               "errors": errors, "quality_ratings": "No human scores assigned by this audit"}
    if args.summary:
        write_json(portable_path(root, args.summary), summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
