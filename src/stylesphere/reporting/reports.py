import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from stylesphere.common.io import canonical_hash, sha256_file, utc_now, write_json
from stylesphere.common.statistics import paired_bootstrap, summary

OUTCOME_METRICS = {"ndcg_at_10", "mrr", "hit_at_5", "valid_output", "structured_task_success", "factual_correctness", "unsupported_claim_rate", "required_fact_coverage", "tool_call_schema_validity", "tool_selection_appropriateness", "tool_call_count", "chamfer_l1", "chamfer_squared", "silhouette_iou", "render_rgb_mae", "render_psnr_db", "mesh_valid", "glb_export_success", "mesh_watertight"}
PAIRED_TIMINGS = {"query_inference_s", "inference_s", "end_to_end_s", "peak_allocated_mb", "peak_reserved_mb"}


def comparable_metric(name):
    return name in OUTCOME_METRICS or name.startswith(("recall_at_", "fscore_"))


def load_records(output_dir, *, with_source_hashes=False):
    """Parse each captured byte snapshot and optionally return its exact SHA256.

    The default remains a records list for existing verification callers. Hash
    keys are the source Paths; report manifests normalize them against root.
    Concurrent appends after capture must invalidate the resulting report.
    """
    records, source_hashes = [], {}
    for file in sorted(Path(output_dir).glob("*/records.jsonl")):
        data = file.read_bytes()
        source_hashes[file] = hashlib.sha256(data).hexdigest()
        for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {file.name}:{line_number}") from exc
    return (records, source_hashes) if with_source_hashes else records


def latest_records(records):
    """Keep configurations distinct; never pool model configurations or smoke/benchmark."""
    latest = {}
    for r in records:
        key = tuple(r.get(k) for k in ("pair", "model", "case_id", "seed", "mode", "protocol_sha256", "config_sha256", "input_sha256"))
        if latest.get(key, {}).get("status") == "success" and r.get("status") != "success":
            continue
        latest[key] = r
    return list(latest.values())


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(root, output_dir, config):
    root, output_dir = Path(root), Path(output_dir)
    raw, captured_source_hashes = load_records(output_dir, with_source_hashes=True)
    records = latest_records(raw)
    folder = output_dir / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    deployments = defaultdict(set)
    for r in raw:
        if r.get("mode") == "evaluation" and r.get("status") in {"success", "failed"}:
            deployments[(r.get("pair"), r.get("model"), r.get("protocol_sha256") or r.get("config_sha256"))].add(r.get("config_sha256"))
    if any(len(configurations) > 1 for configurations in deployments.values()):
        raise ValueError("Several deployment configurations share one model/protocol; use distinct protocol versions before paired analysis")
    rows, failures, paired = [], [], []
    groups, attempt_groups = defaultdict(list), defaultdict(list)
    for r in raw:
        group = (r.get("pair"), r.get("model"), r.get("mode"), r.get("config_sha256"), r.get("protocol_sha256") or r.get("config_sha256"))
        attempt_groups[group].append(r)
        if r.get("status") != "success":
            error = r.get("error", "")
            prefix = str(error).split(":", 1)[0]
            error_type = error.get("type", "unspecified") if isinstance(error, dict) else prefix if prefix.endswith(("Error", "Exception")) else "unspecified"
            failures.append({"pair": r.get("pair"), "model": r.get("model"), "case_id": r.get("case_id"), "seed": r.get("seed"), "mode": r.get("mode"), "run_id": r.get("run_id"), "config_sha256": r.get("config_sha256"), "protocol_sha256": r.get("protocol_sha256"), "status": r.get("status"), "error_type": error_type, "error": str(error)})
    for r in records:
        group = (r.get("pair"), r.get("model"), r.get("mode"), r.get("config_sha256"), r.get("protocol_sha256") or r.get("config_sha256"))
        groups[group].append(r)
    for (pair, model, mode, config_hash, protocol_hash), cases in groups.items():
        input_hash = canonical_hash(sorted({str(r.get("input_sha256")) for r in cases}))
        valid = [r for r in cases if r["status"] == "success" and r.get("provenance") == "model_inference"]
        observations = defaultdict(list)
        singleton_windows = set()
        for r in valid:
            for source in (r.get("metrics", {}), r.get("timings", {})):
                for name, value in source.items():
                    if name in {"model_load_s", "catalogue_indexing_s"}:
                        identity = r.get("run_id") if name == "model_load_s" else r.get("artifacts", {}).get("index_metadata", r.get("input_sha256"))
                        if (name, identity) in singleton_windows:
                            continue
                        singleton_windows.add((name, identity))
                    if isinstance(value, (int, float)) and (not isinstance(value, bool) or comparable_metric(name)):
                        observations[name].append(float(value))
        # Blockers occur before a case can execute; do not call these model failures.
        all_attempts = [r for r in attempt_groups[(pair, model, mode, config_hash, protocol_hash)] if r.get("status") in {"success", "failed"}]
        attempted = len(all_attempts)
        base = {"pair": pair, "model": model, "mode": mode, "config_sha256": config_hash, "protocol_sha256": protocol_hash, "input_sha256": input_hash, "successful_cases": len(valid), "attempted_cases": attempted, "blocked_cases": sum(r["status"] == "blocked" for r in cases)}
        rows.append({**base, "metric": "execution_success_rate", **summary([r["status"] == "success" for r in all_attempts])})
        rows.append({**base, "metric": "execution_failure_rate", **summary([r["status"] == "failed" for r in all_attempts])})
        for name, values in observations.items():
            rows.append({**base, "metric": name, **summary(values)})
        if pair == "generation":
            timed = [r for r in all_attempts if isinstance(r.get("timings", {}).get("inference_s"), (int, float))]
            total_seconds = sum(r["timings"]["inference_s"] for r in timed)
            rows.append({**base, "metric": "valid_outputs_per_inference_second", "n": len(timed),
                         "timed_attempts": len(timed), "untimed_attempts": len(all_attempts) - len(timed),
                         "mean": sum(r.get("metrics", {}).get("valid_output", 0) for r in timed if r.get("status") == "success") / total_seconds if total_seconds > 0 and len(timed) == len(all_attempts) else None,
                         "median": None, "std": None})
    # Match only the same pair, cases, input, config and mode. Generation clusters by prompt.
    matches = defaultdict(dict)
    first_executions = set()
    for r in raw:
        if r.get("mode") != "evaluation" or r.get("status") not in {"success", "failed"} or r.get("provenance") != "model_inference":
            continue
        protocol = r.get("protocol_sha256") or r.get("config_sha256")
        first_key = (r["pair"], r["model"], protocol, r["case_id"], r["seed"], r.get("input_sha256"))
        if first_key in first_executions:
            continue
        first_executions.add(first_key)
        key = (r["pair"], "execution_success_rate", protocol)
        matches[key].setdefault(r["model"], {})[(r["case_id"], r["seed"], r.get("input_sha256"))] = (int(r["status"] == "success"), r.get("prompt_id", r.get("object_id", r["case_id"])))
    for r in records:
        if r.get("mode") != "evaluation" or r.get("status") not in {"success", "failed"} or r.get("provenance") != "model_inference":
            continue
        # Quality/resource comparisons remain conditional on retained successes.
        if r.get("status") != "success":
            continue
        for name, value in {**r.get("metrics", {}), **{k: v for k, v in r.get("timings", {}).items() if k in PAIRED_TIMINGS}}.items():
            if not comparable_metric(name) and name not in PAIRED_TIMINGS:
                continue
            if isinstance(value, (int, float)) and (not isinstance(value, bool) or comparable_metric(name)):
                key = (r["pair"], name, r.get("protocol_sha256") or r.get("config_sha256"))
                matches[key].setdefault(r["model"], {})[(r["case_id"], r["seed"], r.get("input_sha256"))] = (value, r.get("prompt_id", r.get("object_id", r["case_id"])))
    stats = config.get("evaluation", {}).get("statistics", {})
    for (pair, metric, config_hash), models in matches.items():
        if len(models) != 2:
            continue
        a, b = sorted(models)
        keys = sorted(models[a].keys() & models[b].keys())
        if not keys:
            continue
        result = paired_bootstrap([models[a][k][0] for k in keys], [models[b][k][0] for k in keys], clusters=[models[a][k][1] for k in keys], seed=config["seed"], iterations=stats.get("bootstrap_iterations", 2000), confidence=stats.get("confidence", .95))
        paired.append({"pair": pair, "metric": metric, "model_a": a, "model_b": b, "protocol_sha256": config_hash,
                       "matched_cases": len(keys), "eligible_model_a": len(models[a]), "eligible_model_b": len(models[b]),
                       "population": "matched first executed attempt per case" if metric == "execution_success_rate" else "matched successful cases with the metric available", **result})
    write_csv(folder / "model_summary.csv", rows, ["pair", "model", "mode", "config_sha256", "protocol_sha256", "input_sha256", "successful_cases", "attempted_cases", "blocked_cases", "metric", "n", "timed_attempts", "untimed_attempts", "mean", "median", "std"])
    write_csv(folder / "failures.csv", failures, ["pair", "model", "case_id", "seed", "mode", "run_id", "config_sha256", "protocol_sha256", "status", "error_type", "error"])
    error_fields = ["pair", "model", "mode", "config_sha256", "protocol_sha256", "status", "error_type"]
    error_counts = Counter(tuple(failure.get(field) for field in error_fields) for failure in failures)
    write_csv(folder / "execution_error_summary.csv", [{**dict(zip(error_fields, key)), "count": value} for key, value in error_counts.items()], [*error_fields, "count"])
    write_json(folder / "paired_comparisons.json", paired)
    write_csv(folder / "paired_comparisons.csv", paired, ["pair", "metric", "model_a", "model_b", "protocol_sha256", "population", "matched_cases", "eligible_model_a", "eligible_model_b", "n_clusters", "mean_difference_a_minus_b", "ci_low", "ci_high", "confidence", "iterations", "seed", "relative_difference"])
    # Ratings are independent human inputs. Empty worksheets produce no scores.
    from stylesphere.generation.review import aggregate_ratings
    human_reviews = []
    benchmark_generation_keys = {(r.get("run_id"), r.get("model"), r.get("case_id"), r.get("seed")) for r in records if r.get("pair") == "generation" and r.get("mode") == "evaluation" and r.get("status") == "success"}
    canonical_ratings = output_dir / "generation" / "human_review" / "ratings.csv"
    if canonical_ratings.exists():
        from stylesphere.generation.canonical_review import validate_canonical_review
        validate_canonical_review(root, output_dir)
    rating_paths = [canonical_ratings] if canonical_ratings.exists() else sorted((output_dir / "generation").rglob("ratings.csv"))
    for ratings_path in rating_paths:
        mapping_path = ratings_path.with_name("private_mapping.json")
        if not mapping_path.exists():
            continue
        with ratings_path.open(encoding="utf-8-sig", newline="") as stream:
            assessments = list(csv.DictReader(stream))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = [entry for entry in mapping if (entry.get("run_id"), entry.get("model"), entry.get("case_id"), entry.get("seed")) in benchmark_generation_keys]
        if not mapping:
            continue  # Development/smoke human reviews never become benchmark findings.
        review_ids = {entry["review_id"] for entry in mapping}
        assessments = [row for row in assessments if row.get("review_id") in review_ids]
        aggregate = aggregate_ratings(assessments, mapping)
        review_comparisons = []
        for field in ("usable_concept_rate", "usable_vote_rate", "prompt_adherence", "visual_coherence", "composition", "usability_rating"):
            by_model = defaultdict(dict)
            for prompt in aggregate["per_prompt"]:
                if field in prompt["metrics"]:
                    by_model[prompt["model"]][prompt["case_id"]] = prompt["metrics"][field]
            if len(by_model) == 2:
                a, b = sorted(by_model)
                keys = sorted(by_model[a].keys() & by_model[b].keys())
                if keys:
                    review_comparisons.append({"metric": field, "model_a": a, "model_b": b, **paired_bootstrap([by_model[a][k] for k in keys], [by_model[b][k] for k in keys], seed=config["seed"], iterations=stats.get("bootstrap_iterations", 2000))})
        human_reviews.append({"worksheet": str(ratings_path), "ratings_sha256": sha256_file(ratings_path), "mapping_sha256": sha256_file(mapping_path), **aggregate, "paired_prompt_comparisons": review_comparisons})
    write_json(folder / "human_generation_reviews.json", human_reviews)
    from stylesphere.assistance.review import review_assistance
    write_json(folder / "human_assistance_reviews.json", review_assistance(records, output_dir / "assistance" / "human_assessments.csv"))
    captions = []
    metrics = sorted({(r["pair"], r["metric"]) for r in rows if r["mode"] == "evaluation" and r["mean"] is not None and (comparable_metric(r["metric"]) or r["metric"] in PAIRED_TIMINGS or r["metric"] in {"execution_success_rate", "valid_outputs_per_inference_second", "catalogue_indexing_s", "model_load_s"})})
    if metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for pair, metric in metrics:
            selected = [r for r in rows if r["mode"] == "evaluation" and r["pair"] == pair and r["metric"] == metric and r["mean"] is not None]
            fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
            labels = [str(r["model"]) + "\nprotocol " + str(r["protocol_sha256"])[:8] + f" (n={r['n']})" for r in selected]
            palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
            positions = list(range(len(selected)))
            ax.bar(positions, [r["mean"] for r in selected], color=[palette[i % 4] for i in range(len(selected))])
            ax.set_xticks(positions, labels)
            ax.set_ylabel(metric + (" (seconds)" if metric.endswith("_s") else " (MiB)" if metric.endswith("_mb") else ""))
            ax.set_title(f"{pair.title()}: observed {metric}")
            ax.spines[["top", "right"]].set_visible(False)
            name = f"{pair}_{metric}"
            for extension in ("png", "svg"):
                fig.savefig(folder / f"{name}.{extension}", dpi=180)
            plt.close(fig)
            captions.append(f"- {name}: arithmetic means over eligible observed cases; configurations identified by hash; no inference across unmatched cases.")
    text = "# Observed execution report\n\n" + f"Raw records: {len(raw)}. Distinct case/configuration records: {len(records)}.\n\n"
    text += "For the confirmed four-pair component's completed quality analyses, real-reference extension and recommendations, read [the final evaluation](../final_evaluation/REPORT.md). This file preserves the original execution and human-review reporting scope.\n\n"
    text += "Blocked counts include preflight sentinel records when no dataset could be loaded; these are not executed model cases or model-failure observations. Model-load and catalogue-indexing windows are counted once per actual load/build.\n\n"
    text += "Execution rates retain all logged attempts, including failed retries. Throughput is available only when every executed attempt has a measured inference window; timed/untimed coverage is explicit in model_summary.csv. failures.csv retains every raw failure/blocker. The case table shows retained resumable case outcomes; paired reliability uses the first executed attempt per matched case so a later retry cannot erase the original failure. Quality/resource comparisons use matched retained successful outputs.\n\n"
    text += "Execution success and resource measurements do not establish task quality. Quality outcomes with missing labels or human assessments remain unavailable. No model winner or business recommendation is automatically produced.\n\n"
    text += "Execution exception counts are in execution_error_summary.csv, separated by mode, protocol and blocked/failed status. These are observed execution diagnostics; semantic error categories still require independent review.\n\n"
    if not any(r.get("status") == "success" and r.get("mode") == "evaluation" for r in records):
        text += "No successful benchmark evaluation records are available. No benchmark winner or quality recommendation is supported. Smoke outputs are excluded from benchmark charts and pairwise inference.\n\n"
    text += "| Pair | Model | Mode | Protocol | Successful | Attempted | Blocked |\n|---|---|---|---|---:|---:|---:|\n"
    for group, cases in groups.items():
        text += f"| {group[0]} | {group[1]} | {group[2]} | {str(group[4])[:12]} | {sum(r['status']=='success' for r in cases)} | {sum(r['status']!='blocked' for r in cases)} | {sum(r['status']=='blocked' for r in cases)} |\n"
    if paired:
        text += "\n## Matched comparisons\n\nValues are A minus B; sign alone is not a quality recommendation. CI resamples the stated independent case/prompt/object clusters. Reliability includes executed failures; other comparisons are conditional on matched successful cases with that metric. See paired_comparisons.csv for eligible counts and confidence levels.\n\n| Pair | A / B | Metric | Matched cases | Clusters | Mean difference | CI |\n|---|---|---|---:|---:|---:|---|\n"
        for comparison in paired:
            interval = f"[{comparison['ci_low']:.6g}, {comparison['ci_high']:.6g}]" if comparison["ci_low"] is not None else "unavailable"
            text += f"| {comparison['pair']} | {comparison['model_a']} / {comparison['model_b']} | {comparison['metric']} | {comparison['matched_cases']} | {comparison['n_clusters']} | {comparison['mean_difference_a_minus_b']:.6g} | {interval} |\n"
    text += "\n## Figure captions\n\n" + ("\n".join(captions) if captions else "No eligible benchmark values; no numerical charts generated.") + "\n"
    (folder / "REPORT.md").write_text(text, encoding="utf-8")
    review_inputs = list((output_dir / "generation").rglob("ratings.csv")) + list((output_dir / "generation").rglob("private_mapping.json")) + [output_dir / "generation" / "human_review" / "manifest.json", output_dir / "assistance" / "human_assessments.csv"]
    manifest = {"generated_at": utc_now(), "source_record_hashes": {p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p): digest for p, digest in captured_source_hashes.items()}, "source_review_hashes": {p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p): sha256_file(p) for p in review_inputs if p.is_file()}, "records_digest": canonical_hash(raw), "generated_files": {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.iterdir() if p.is_file() and p.name != "manifest.json"}}
    write_json(folder / "manifest.json", manifest)
    return {"records": len(raw), "summary_rows": len(rows), "paired_comparisons": len(paired), "report": str(folder / "REPORT.md")}
