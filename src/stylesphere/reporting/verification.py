import csv
import json
import math
import re
from pathlib import Path
from stylesphere.common.config import PAIRS, dataset_fingerprints, load_config, pair_protocol_sha256
from stylesphere.common.io import canonical_hash, portable_path, read_jsonl, sha256_file, utc_now, write_json
from .reports import load_records

REQUIRED_DOCS = ("EXPERIMENT_PROTOCOL", "DATASET_PROVENANCE", "SOURCES_AND_LICENSES", "METRIC_DEFINITIONS", "REPRODUCIBILITY", "LIMITATIONS", "IMPLEMENTATION_STATUS", "AI_ASSISTANCE_DISCLOSURE")


def execution_coverage(root, config, records):
    """Count current planned cases, without treating historical blockers as attempts."""
    from .reports import latest_records
    fingerprints = dataset_fingerprints(root, config)
    coverage = []
    for pair in PAIRS:
        folder = Path(root) / config.get("evaluation", {}).get(pair, {}).get("dataset_dir", f"datasets/{pair}")
        if pair in {"retrieval", "reconstruction"}:
            file, key = ("queries.csv", "query_id") if pair == "retrieval" else ("objects.csv", "object_id")
            with (folder / file).open(encoding="utf-8-sig", newline="") as stream:
                cases = list(csv.DictReader(stream))
        else:
            file, key = ("prompts.jsonl", "prompt_id") if pair == "generation" else ("scenarios.jsonl", "scenario_id")
            cases = read_jsonl(folder / file)
        repeats = int(config.get("evaluation", {}).get(pair, {}).get("repeats", 3)) if pair == "generation" else 1
        expected = {(row[key], config["seed"] + i) for row in cases for i in range(repeats)}
        protocol = pair_protocol_sha256(config, pair, fingerprints)
        for model, options in config["models"].items():
            if options["pair"] != pair:
                continue
            observed = {}
            for record in latest_records(records):
                identity = (record.get("case_id"), record.get("seed"))
                if record.get("model") == model and record.get("mode") == "evaluation" and record.get("protocol_sha256") == protocol and identity in expected:
                    if observed.get(identity) != "success":
                        observed[identity] = record.get("status")
            success = sum(value == "success" for value in observed.values())
            failed = sum(value == "failed" for value in observed.values())
            coverage.append({"pair": pair, "model": model, "protocol_sha256": protocol,
                             "planned_cases": len(expected), "successful_cases": success,
                             "failed_cases": failed, "unexecuted_cases": len(expected) - success - failed})
    return coverage


def verify(root, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    errors, pending, checked = [], [], 0
    required = ["README.md", "AGENTS.md", "pyproject.toml", "config/models.yaml", "config/evaluation.yaml", "environment/doctor.json"] + [f"docs/{name}.md" for name in REQUIRED_DOCS]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing required file: {name}")
    config = None
    try:
        config = load_config(root)
    except Exception as exc:
        errors.append(f"Configuration: {exc}")
    from stylesphere.datasets import validate
    for pair in PAIRS:
        if not (root / "datasets" / pair).is_dir():
            errors.append(f"Missing dataset directory: {pair}")
        try:
            for issue in validate(root, pair):
                # Empty templates, absent labels and explicitly omitted assets
                # are unfinished work. Invalid schemas/provenance and changed
                # bytes are integrity failures before any model output exists.
                incomplete = ("contains no cases/judgements; template only" in issue
                              or issue.startswith("No human judgements for ")
                              or issue.startswith("Missing ") and " asset:" in issue)
                destination = pending if incomplete else errors
                destination.append(f"Dataset {pair}: {issue}")
        except Exception as exc:
            errors.append(f"Dataset {pair}: {exc}")
    try:
        records = load_records(output_dir)
    except Exception as exc:
        errors.append(f"Result records cannot be parsed: {exc}")
        records = []
    seen_success = {p: 0 for p in PAIRS}
    for record in records:
        checked += 1
        label = f"{record.get('pair')}/{record.get('model')}/{record.get('case_id')}"
        if record.get("status") not in {"success", "failed", "blocked"}:
            errors.append(f"Invalid execution status: {label}")
        if record.get("provenance") != "model_inference" or record.get("synthetic", False):
            errors.append(f"Non-model or synthetic result found: {label}")
        if record.get("mode") not in {"smoke", "evaluation"}:
            errors.append(f"Missing execution mode: {label}")
        if record.get("status") != "success":
            pending.append(f"{label}: {record.get('status')}: {record.get('error')}")
            if not record.get("artifacts"):
                continue
        for key in ("run_id", "timestamp_utc", "config_sha256", "input_sha256", "checkpoint_revision"):
            if not record.get(key):
                errors.append(f"Missing evidence {key}: {label}")
        if record.get("checkpoint_revision") in {"main", "master", "latest"}:
            errors.append(f"Unresolved checkpoint revision: {label}")
        if record.get("status") == "success" and record.get("mode") == "evaluation":
            seen_success[record["pair"]] += 1
        artifacts = record.get("artifacts", {})
        if not artifacts:
            errors.append(f"Successful case has no output artifacts: {label}")
        paths = {}
        for name, value in artifacts.items():
            if not isinstance(value, str):
                errors.append(f"Artifact must be a path: {label}/{name}")
                continue
            path = portable_path(root, value).resolve()
            paths[name] = path
            if not path.is_file():
                errors.append(f"Missing output: {label}/{name}")
            elif name not in record.get("artifact_sha256", {}):
                errors.append(f"Missing artifact hash: {label}/{name}")
            elif sha256_file(path) != record["artifact_sha256"][name]:
                errors.append(f"Changed artifact: {label}/{name}")
        # A retained descriptor must bind to the original case fingerprint even
        # when its bytes and artifact hash have both been replaced. Failed cases
        # retain the same input evidence and receive this check too.
        if record.get("pair") == "reconstruction" and "input_descriptor" in paths and paths["input_descriptor"].is_file():
            try:
                descriptor = json.loads(paths["input_descriptor"].read_text(encoding="utf-8"))
                if canonical_hash(descriptor) != record.get("input_sha256"):
                    errors.append(f"Frozen reconstruction input fingerprint mismatch: {label}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"Invalid reconstruction input descriptor: {label}: {exc}")
        if record.get("pair") == "reconstruction" and "renders_manifest" in paths and paths["renders_manifest"].is_file():
            try:
                manifest = json.loads(paths["renders_manifest"].read_text(encoding="utf-8"))
                for rendering in manifest.get("renders", []):
                    nested = [rendering.get("image"), rendering.get("silhouette")]
                    nested.extend(rendering.get("camera", {}).get(field) for field in ("reference_rgb", "reference_silhouette"))
                    for value in filter(None, nested):
                        if not portable_path(root, value).is_file():
                            errors.append(f"Missing nested reconstruction output/reference: {label}/{value}")
                        if Path(str(value).replace("\\", "/")).is_absolute():
                            errors.append(f"Nonportable nested reconstruction output/reference: {label}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"Invalid reconstruction rendering manifest: {label}: {exc}")
        # Regenerate numerical retrieval metrics from retained raw ranked predictions.
        if record.get("status") == "success" and record.get("pair") == "retrieval" and "prediction" in paths and paths["prediction"].exists():
            from stylesphere.retrieval.metrics import score_query
            prediction = json.loads(paths["prediction"].read_text(encoding="utf-8"))
            try:
                recomputed = score_query(prediction["ranking"], prediction["qrels"], complete=prediction.get("qrels_complete", False), k=int(prediction.get("recall_k", 10)))
                descriptor = json.loads(paths["input_descriptor"].read_text(encoding="utf-8"))
                if canonical_hash(descriptor) != record["input_sha256"]:
                    errors.append(f"Frozen retrieval input fingerprint mismatch: {label}")
                for name, artifact in (("catalogue.csv", "frozen_catalogue"), ("queries.csv", "frozen_queries"), ("qrels.csv", "frozen_qrels")):
                    if sha256_file(paths[artifact]) != descriptor["manifests"][name]:
                        errors.append(f"Frozen retrieval manifest mismatch: {label}/{name}")
                with paths["frozen_qrels"].open(encoding="utf-8-sig", newline="") as stream:
                    frozen_qrels = {r["product_id"]: int(r["relevance"]) for r in csv.DictReader(stream) if r["query_id"] == record["case_id"]}
                if frozen_qrels != prediction["qrels"]:
                    errors.append(f"Saved ranking labels differ from independent frozen qrels: {label}")
                with paths["frozen_catalogue"].open(encoding="utf-8-sig", newline="") as stream:
                    product_ids = {r["product_id"] for r in csv.DictReader(stream)}
                if set(prediction["ranking"]) != product_ids or len(prediction["ranking"]) != len(product_ids):
                    errors.append(f"Saved ranking does not exactly cover the shared catalogue: {label}")
                for metric, value in record.get("metrics", {}).items():
                    expected = recomputed.get(metric)
                    if (value is None) != (expected is None) or (isinstance(value, (int, float)) and not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)):
                        errors.append(f"Metric cannot be regenerated: {label}/{metric}")
            except Exception as exc:
                errors.append(f"Prediction cannot be rescored: {label}: {exc}")
        if record.get("pair") == "assistance" and "trace" in paths:
            try:
                from stylesphere.assistance.verification import verify_trace
                kb = json.loads(paths["frozen_knowledge_base"].read_text(encoding="utf-8"))
                scenarios = read_jsonl(paths["frozen_scenarios"])
                if canonical_hash({"knowledge_base": kb, "scenarios": scenarios}) != record["input_sha256"]:
                    errors.append(f"Frozen assistance input fingerprint mismatch: {label}")
                scenario = next(s for s in scenarios if s["scenario_id"] == record["case_id"])
                trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
                errors.extend(f"{label}: {issue}" for issue in verify_trace(trace, scenario, kb))
                if (trace.get("metrics") or {}) != (record.get("metrics") or {}):
                    errors.append(f"Record metrics differ from replayed assistance trace: {label}")
            except Exception as exc:
                errors.append(f"Assistance trace cannot be rescored: {label}: {exc}")
        if record.get("status") == "success" and record.get("pair") == "reconstruction":
            from stylesphere.reconstruction.metrics import verify_record
            errors.extend(f"{label}: {issue}" for issue in verify_record(record, root))
        if record.get("status") == "success" and record.get("pair") == "generation" and "image" in paths:
            try:
                from stylesphere.generation.provenance import verify_prompt_binding
                errors.extend(f"{label}: {issue}" for issue in verify_prompt_binding(record, root, output_dir / "generation"))
                from PIL import Image
                with Image.open(paths["image"]) as generated:
                    generated.verify()
                if record.get("metrics", {}).get("valid_output") != 1:
                    errors.append(f"Generation validity metric mismatch: {label}")
            except Exception as exc:
                errors.append(f"Generation image cannot be verified: {label}: {exc}")
    for pair, successes in seen_success.items():
        if not successes:
            pending.append(f"No successful full benchmark cases for {pair}")
    coverage = []
    if config is not None:
        try:
            coverage = execution_coverage(root, config, records)
            pending.extend(f"Current protocol {row['protocol_sha256'][:12]}: {row['model']} has {row['unexecuted_cases']} planned cases not executed" for row in coverage if row["unexecuted_cases"])
        except Exception as exc:
            pending.append(f"Planned execution coverage unavailable: {type(exc).__name__}: {exc}")
    manifest_path = output_dir / "reports" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sources = {portable_path(root, value).resolve() for value in manifest.get("source_record_hashes", {})}
        current_sources = {path.resolve() for path in output_dir.glob("*/records.jsonl")}
        if expected_sources != current_sources:
            errors.append("Report source-file set changed; regenerate report")
        if manifest.get("records_digest") != canonical_hash(records):
            errors.append("Report record snapshot differs from current records; regenerate report")
        for value, expected in {**manifest.get("source_record_hashes", {}), **manifest.get("source_review_hashes", {})}.items():
            path = portable_path(root, value)
            if not path.exists() or sha256_file(path) != expected:
                errors.append(f"Report source changed; regenerate report: {value}")
        for name, expected in manifest.get("generated_files", {}).items():
            path = manifest_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                errors.append(f"Report artifact changed: {name}")
    else:
        pending.append("Report has not been generated")
    benchmark_successes = [record for record in records if record.get("mode") == "evaluation" and record.get("status") == "success"]
    if (output_dir / "generation" / "human_review" / "ratings.csv").exists():
        try:
            from stylesphere.generation.canonical_review import validate_canonical_review
            validate_canonical_review(root, output_dir)
        except Exception as exc:
            errors.append(f"Canonical generation review: {exc}")
    generation_successes = [record for record in benchmark_successes if record.get("pair") == "generation"]
    generation_review = output_dir / "reports" / "human_generation_reviews.json"
    if generation_successes and generation_review.is_file():
        reviews = json.loads(generation_review.read_text(encoding="utf-8"))
        decided = sum(review.get("usable_coverage", {}).get("n_images_decided", 0) for review in reviews)
        if decided < len(generation_successes):
            pending.append(f"Independent generation usability decisions unavailable for at least {len(generation_successes) - decided} successful benchmark outputs")
    assistance_successes = [record for record in benchmark_successes if record.get("pair") == "assistance"]
    assistance_review = output_dir / "reports" / "human_assistance_reviews.json"
    if assistance_successes and assistance_review.is_file():
        review = json.loads(assistance_review.read_text(encoding="utf-8"))
        reviewed_cases = [case for case in review.get("per_case", []) if all(name in case.get("metrics", {}) for name in ("task_success", "factual_prose_correct", "unsupported_claim_present"))]
        if len(reviewed_cases) < len(assistance_successes):
            pending.append(f"Independent assistance task/prose assessments unavailable for at least {len(assistance_successes) - len(reviewed_cases)} successful benchmark executions")
    reconstruction_successes = [record for record in benchmark_successes if record.get("pair") == "reconstruction"]
    missing_geometry = sum(record.get("metrics", {}).get("chamfer_squared") is None for record in reconstruction_successes)
    if missing_geometry:
        pending.append(f"Primary reconstruction accuracy unavailable for {missing_geometry} successful outputs: eligible aligned references are required")
    report = {"timestamp_utc": utc_now(), "integrity_passed": not errors, "benchmark_complete": False, "checked_records": checked, "execution_coverage": coverage, "errors": errors, "pending": pending, "note": "Integrity checks validate retained evidence; full benchmark completion requires frozen datasets, all planned executions and independent human assessments."}
    write_json(output_dir / "verification.json", report)
    return report
