"""Strict completion of the authorized four-pair Codes+Data evaluation scope.

AI-assisted quality assessment is authorized for this scope. It remains separate
from the legacy benchmark's independent-human-primary completion requirement.
No inference, rendering, network access, human ratings or report tables are made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path

from stylesphere.common.io import canonical_hash, portable_path, sha256_file, utc_now, write_json

RECEIPT = "results/final_evaluation/completion.json"
EXPECTED_MODELS = {"fashion_siglip": 120, "siglip2": 120, "sdxl_lightning": 72,
                   "flux2_klein": 72, "phi4_mini": 80, "qwen35_4b": 80, "sf3d": 20, "spar3d": 20}
ORIGINAL_RAW_SHA256 = {
    "results/assistance/records.jsonl": "a4de4c4880f399e3aa9218f8e1084dce920557f93c7ad8ab5369a42561c01dd1",
    "results/generation/records.jsonl": "a5e220ed946c09b85e4f32f49d0dd932f68cb7c3381143e3cd594a3ac707d443",
    "results/reconstruction/records.jsonl": "c90b97c30eb1149626e941e3457e5313f87c5191b02280feec70c6b40d6de001",
    "results/retrieval/records.jsonl": "a6a04f3a52e6080f5cfa8060a0fe9d4b5a50d0f1c04c14c8177c74be93fe3d23"}
REQUIRED_COUNTS = {"original_raw_records": 928, "planned_technical_attempts": 584,
                   "technical_successes": 511, "technical_protocol_failures": 73,
                   "ai_visual_outputs": 184, "ai_visual_generation_outputs": 144,
                   "ai_visual_reconstruction_outputs": 40, "assistance_assessments": 160,
                   "assistance_failed_attempts_assessed": 73, "retrieval_product_profiles": 600,
                   "retrieval_proxy_pairs": 72000, "retrieval_query_scores": 240,
                   "retrieval_queries_per_model": 120, "reference_development_attempts": 8,
                   "reference_test_attempts": 40, "reference_test_scored": 40,
                   "reference_test_paired_objects": 20}
REQUIRED_CHECKS = {"visual_ai_quality", "assistance_ai_quality", "retrieval_ai_proxy_quality",
                   "authorized_technical_scope", "human_ai_separation", "normalized_reference_extension"}


class Evidence:
    """Bind parsed byte snapshots, check stable source bytes again at the end."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, value):
        candidate = Path(str(value).replace("\\", "/"))
        # A fresh extraction may itself be nested inside an original-workspace
        # alias. Such current absolute paths must not be rebased a second time.
        resolved = candidate.resolve() if candidate.is_absolute() else None
        path = resolved if resolved is not None and resolved.is_relative_to(self.root) else portable_path(self.root, str(value)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Completion evidence must remain inside the project")
        return path

    def bind(self, value, expected=None):
        path = self.path(value)
        name = path.relative_to(self.root).as_posix()
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Evidence SHA256 differs: {name}")
        if name in self.hashes and self.hashes[name] != actual:
            raise ValueError(f"Evidence changed during validation: {name}")
        self.hashes[name] = actual
        return path

    def data(self, value):
        path = self.path(value)
        data = path.read_bytes()
        name = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        if name in self.hashes and self.hashes[name] != digest:
            raise ValueError(f"Evidence changed during validation: {name}")
        self.hashes[name] = digest
        return data.decode("utf-8-sig")

    def json(self, value):
        return json.loads(self.data(value))

    def lines(self, value):
        return [json.loads(line) for line in self.data(value).splitlines() if line.strip()]

    def csv(self, value):
        return list(csv.DictReader(io.StringIO(self.data(value))))

    def mapping(self, base, values):
        for name, expected in values.items():
            self.bind(Path(base) / name, expected)

    def artifacts(self, record):
        if set(record.get("artifacts", {})) != set(record.get("artifact_sha256", {})):
            raise ValueError("Raw artifact paths/hash keys differ")
        for name, value in record.get("artifacts", {}).items():
            self.bind(value, record["artifact_sha256"][name])

    def stable(self):
        return [f"Evidence changed during completion validation: {name}" for name, digest in self.hashes.items()
                if not (self.root / name).is_file() or sha256_file(self.root / name) != digest]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(canonical_hash(actual) == canonical_hash(expected), message)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Evidence chronology requires timezone-aware timestamps")
    return result


def _visual(e, counts):
    from .ai_visual import aggregate, select_verified_records, validate_export
    from .ai_visual_coverage import decision_coverage
    base = "results/ai_visual_assessment"
    manifest = e.json(f"{base}/manifest.json")
    e.mapping(base, manifest["files_sha256"])
    e.bind("src/stylesphere/reporting/ai_visual.py")
    e.bind("src/stylesphere/reporting/ai_visual_coverage.py")
    e.bind("environment/reconstruction-benchmark-validation.json")
    e.bind("results/generation/human_review/manifest.json")
    mapping = e.json(f"{base}/private_mapping.json")
    cards = e.json(f"{base}/public_cards.json")
    require(len(mapping) == len(cards) == 184, "Visual supplement must cover184 actual outputs")
    require(len({x["assessment_id"] for x in mapping}) == 184, "Duplicate visual source identity")
    assessments = [row for path in sorted((e.root / base / "assessments").glob("*.jsonl")) for row in e.lines(path)]
    require(len(assessments) == 184 and {r["assessment_id"] for r in assessments} == {r["assessment_id"] for r in cards}, "Missing/extra visual assessments")
    require(all(r["review_status"] in {"assessed", "abstained"} for r in assessments), "Pending visual assessment")
    for path in (e.root / "results/generation").rglob("prompt_binding.json"):
        e.bind(path)
    # The source selector validates exact raw/input/artifact bindings, independently
    # of AI scores. All selected artifact bytes are then receipt-bound too.
    source = select_verified_records(e.root, e.root / "results")
    for record in source:
        e.artifacts(record)
    require(len(source) == 184, "Visual source selector does not cover184 originals")
    source_by_hash = {canonical_hash(record): record for record in source}
    require(len(source_by_hash) == 184, "Repeated visual raw source")
    require({row["source_record_sha256"] for row in mapping} == set(source_by_hash), "Visual packets no longer bind selected original outputs")
    for row in mapping:
        record = source_by_hash[row["source_record_sha256"]]
        for field in ("model", "case_id", "seed", "run_id", "input_sha256"):
            require(row[field] == record[field], "Visual mapping identity differs from actual raw record")
    validate_export(e.root / base)
    equal(e.json(f"{base}/summary.json"), aggregate(e.root / base), "Visual summary is stale or cannot be regenerated")
    diagnostic = e.json(f"{base}/decision_coverage.json")
    recomputed = decision_coverage(mapping, assessments)
    equal({key: diagnostic[key] for key in recomputed}, recomputed, "Visual decision-denominator diagnostic differs")
    require(diagnostic["protocol_sha256"] == manifest["protocol_sha256"] and diagnostic["selection_sha256"] == manifest["selection_sha256"], "Visual diagnostic protocol/source differs")
    e.bind("src/stylesphere/reporting/ai_visual_coverage.py", diagnostic["helper_source_sha256"])
    e.mapping(base, diagnostic["assessment_files_sha256"])
    counts.update(ai_visual_outputs=len(assessments), ai_visual_generation_outputs=sum(r["pair"] == "generation" for r in mapping),
                  ai_visual_reconstruction_outputs=sum(r["pair"] == "reconstruction" for r in mapping))
    return source


def _assistance(e, counts):
    from stylesphere.assistance.ai_assessment import PROTOCOL, collect_packets, materialize_decisions, validate_assessments
    base = "results/assistance/ai_assessment/v1"
    manifest = e.json(f"{base}/source_manifest.json")
    e.bind("src/stylesphere/assistance/ai_assessment.py", manifest["implementation_sha256"])
    protocol = e.json(f"{base}/assessment_protocol.json")
    equal(protocol, PROTOCOL, "Assistance AI rubric differs")
    require(canonical_hash(protocol) == manifest["protocol_sha256"], "Assistance protocol hash differs")
    e.bind(f"{base}/evidence_packets.jsonl", manifest["packets_sha256"])
    retained = e.lines(f"{base}/evidence_packets.jsonl")
    # Register all trace/input bytes before replay to detect concurrent changes.
    for packet in retained:
        for key, value in packet["source_artifacts"].items():
            e.bind(value, packet["source_artifact_sha256"][key])
    packets = collect_packets(e.root)
    equal(retained, packets, "Assistance evidence differs from all actual final traces")
    require(manifest["packet_count"] == 160, "Assistance manifest case count differs")
    equal(manifest["source_record_sha256"], {p["assessment_id"]: p["source_record_sha256"] for p in packets}, "Assistance manifest source set differs")
    reviews = e.lines(f"{base}/assessments.jsonl")
    ledger = e.data(f"{base}/authored_decisions.txt")
    validate_assessments(packets, reviews)
    equal(reviews, materialize_decisions(packets, ledger), "Assistance decisions differ from explicit authored ledger")
    require(len(packets) == len(reviews) == 160, "All160 assistance attempts require assessment")
    require(Counter(p["model"] for p in packets) == {"phi4_mini": 80, "qwen35_4b": 80}, "Assistance model/case coverage differs")
    for model in ("phi4_mini", "qwen35_4b"):
        require({p["case_id"] for p in packets if p["model"] == model} == {f"assist-{i:03d}" for i in range(1, 81)}, "Missing assistance case")
    require(sum(p["original_status"] == "failed" for p in packets) == 73, "Failed assistance traces were excluded or rewritten")
    summary = e.json(f"{base}/summary.json")
    e.mapping(base, summary["source_hashes"])
    require(summary["n_executions"] == 160 and summary["assessor"]["human"] is False, "Assistance summary provenance/count differs")
    rows = {(r["model"], r["category"]): r for r in summary["rows"]}
    grouped = defaultdict(list)
    lookup = {p["assessment_id"]: p for p in packets}
    for review in reviews:
        grouped[(review["model"], "all")].append(review)
        grouped[(review["model"], lookup[review["assessment_id"]]["scenario"]["evaluation_category"])].append(review)
    require(set(rows) == set(grouped), "Assistance summary category coverage differs")
    for key, subset in grouped.items():
        row = rows[key]
        require(row["n_executions"] == len(subset), "Assistance summary execution denominator differs")
        expected_counts = {
            "n_customer_prose": sum(lookup[r["assessment_id"]]["response"]["text"] is not None for r in subset),
            "original_protocol_completions": sum(lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in subset),
            "deterministic_exact_executed_tool_set_match": sum(lookup[r["assessment_id"]]["deterministic"]["exact_executed_tool_set_match"] for r in subset),
            "requested_fact_entailment_counts": dict(Counter(v for r in subset for v in r["requested_fact_entailment"].values())),
            "deterministic_required_facts_observed": sum(lookup[r["assessment_id"]]["deterministic"]["required_facts_observed"] for r in subset),
            "deterministic_required_facts_total": sum(lookup[r["assessment_id"]]["deterministic"]["required_fact_count"] for r in subset),
            "ai_semantically_satisfied_original_failures": sum(r["ai_semantic_task_verdict"] == "satisfied" and not lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in subset),
            "ai_operational_and_grounded_semantic_success": sum(r["ai_semantic_task_verdict"] == "satisfied" and r["ai_ungrounded_claim_present"] is False and lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in subset)}
        equal({name: row[name] for name in expected_counts}, expected_counts, "Assistance diagnostic/intersection counts differ")
        for metric, field, positive, excluded in (
            ("ai_semantic_task_success", "ai_semantic_task_verdict", "satisfied", {"unassessable"}),
            ("ai_factual_prose_supported", "ai_factual_prose_verdict", "supported", {"no_factual_claims", "no_customer_prose"}),
            ("ai_unsupported_claim_present", "ai_unsupported_claim_present", True, {None}),
            ("ai_ungrounded_claim_present", "ai_ungrounded_claim_present", True, {None}),
            ("ai_tool_selection_appropriate", "ai_tool_selection_verdict", "appropriate", {"unassessable"})):
            values = [review[field] for review in subset if review[field] not in excluded]
            n = sum(value == positive for value in values)
            equal(row[metric], {"positive": n, "n_assessable": len(values), "n_missing_or_not_applicable": len(subset) - len(values), "rate": n / len(values) if values else None}, "Assistance summary rate cannot be regenerated")
        equal(row["error_categories"], dict(Counter(value for review in subset for value in review["error_categories"])), "Assistance error diagnostic differs")
    counts.update(assistance_assessments=len(reviews), assistance_failed_attempts_assessed=73)
    records = e.lines("results/assistance/records.jsonl")
    selected = {p["source_record_sha256"] for p in packets}
    return [r for r in records if canonical_hash(r) in selected]


def _retrieval(e, counts):
    from stylesphere.retrieval.ai_assessment import PROTOCOL, TRAIT_KEYS
    from stylesphere.retrieval.proxy_metrics import check_evidence, clustered_paired_bootstrap, proxy_grade, ranking_metrics, validate_profiles
    base = "results/retrieval/ai_assessment/v1"
    evidence_manifest = e.json(f"{base}/evidence_manifest.json")
    e.bind("src/stylesphere/retrieval/ai_assessment.py", evidence_manifest["implementation_sha256"])
    e.bind(f"{base}/product_evidence.jsonl", evidence_manifest["product_evidence_sha256"])
    e.mapping(f"{base}/sheets", evidence_manifest["sheet_sha256"])
    evidence = e.lines(f"{base}/product_evidence.jsonl")
    for packet in evidence:
        e.bind(packet["image_path"], packet["image_sha256"])
    for name in ("assessment_protocol.json", "protocol_freeze.json"):
        e.bind(f"{base}/{name}")
    check_evidence(e.root, e.root / base)
    freeze = e.json(f"{base}/profiles_freeze.json")
    require(freeze["rankings_read"] is False and freeze["protocol_sha256"] == canonical_hash(PROTOCOL), "Retrieval labels not frozen independently of rankings")
    e.bind("src/stylesphere/retrieval/proxy_metrics.py", freeze["implementation_sha256"])
    e.bind(f"{base}/evidence_manifest.json", freeze["evidence_manifest_sha256"])
    e.mapping(f"{base}/annotations", freeze["annotation_source_sha256"])
    actual_annotation_files = {p.name for p in (e.root / base / "annotations").glob("*.jsonl")}
    require(actual_annotation_files == set(freeze["annotation_source_sha256"]), "Retrieval annotation source-file set changed")
    e.bind(f"{base}/frozen_product_profiles.jsonl", freeze["profiles_sha256"])
    profiles = e.lines(f"{base}/frozen_product_profiles.jsonl")
    validate_profiles(evidence, profiles)
    protocol_freeze = e.json(f"{base}/protocol_freeze.json")
    require(all(timestamp(protocol_freeze["frozen_at_utc"]) <= timestamp(p["assessed_at_utc"]) <= timestamp(freeze["frozen_at_utc"]) for p in profiles), "Retrieval annotations were not recorded between rubric and profile freeze")
    authored = [row for name in sorted(actual_annotation_files) for row in e.lines(f"{base}/annotations/{name}")]
    equal(sorted(authored, key=lambda p: p["annotation_index"]), profiles, "Frozen profiles differ from authored annotations")
    require(len(profiles) == 600, "Retrieval requires600 product profiles")
    catalogue = e.csv("datasets/retrieval/catalogue.csv")
    queries = e.csv("datasets/retrieval/queries.csv")
    require(len(catalogue) == 600 and len(queries) == 120 and len({q["query_id"] for q in queries}) == 120, "Retrieval source case coverage differs")
    e.bind("datasets/retrieval/qrels.csv", evidence_manifest["source_sha256"]["original_human_qrels"])
    require(not e.csv("datasets/retrieval/qrels.csv"), "Human qrels changed; never substitute AI labels into human-primary files")
    qmanifest = e.json(f"{base}/qrels_manifest.json")
    require(qmanifest["profiles_sha256"] == freeze["profiles_sha256"] and qmanifest["human_ground_truth"] is False and qmanifest["rankings_used_for_labels"] is False, "Proxy qrels provenance/freeze differs")
    e.bind(f"{base}/proxy_qrels.jsonl", qmanifest["qrels_sha256"])
    e.bind(f"{base}/query_coverage.json", qmanifest["query_coverage_sha256"])
    qrels = e.lines(f"{base}/proxy_qrels.jsonl")
    expected, expected_coverage, by_query = [], [], {}
    for query in queries:
        kind = query["query_category"].replace(" ", "_")
        trait = TRAIT_KEYS[(int(query["query_id"].rsplit("q", 1)[-1]) - 1) % 10]
        grades = {p["product_id"]: proxy_grade(p, kind, trait) for p in profiles}
        by_query[query["query_id"]] = grades
        expected.extend({"query_id": query["query_id"], "product_id": p["product_id"], "grade": grades[p["product_id"]], "query_type": kind, "trait": trait, "assessment_kind": "AI_proxy_not_human_qrels"} for p in profiles)
        values = list(grades.values())
        expected_coverage.append({"query_id": query["query_id"], "query_text": query["query_text"], "n_catalogue": 600,
            "n_judged": sum(v is not None for v in values), "n_unknown": sum(v is None for v in values),
            "known_useful_relevant": sum(v is not None and v >= 2 for v in values), "all_catalogue_complete": all(v is not None for v in values),
            "grade_counts": dict(Counter("unknown" if v is None else str(v) for v in values)), "physical_weight_measured": False if trait == "travel_appearance" else None})
    equal(qrels, expected, "All72000 proxy grades cannot be regenerated from frozen independent profiles")
    equal(e.json(f"{base}/query_coverage.json"), {"schema_version": 1, "n_queries": 120, "n_pairs": 72000, "rows": expected_coverage}, "Proxy coverage summary differs")
    summary = e.json(f"{base}/summary.json")
    require(timestamp(summary["generated_at_utc"]) >= timestamp(freeze["frozen_at_utc"]), "Ranking scores predate the independent annotation freeze")
    e.bind(f"{base}/query_scores.jsonl", summary["score_sha256"])
    e.bind(f"{base}/qrels_manifest.json", summary["qrels_manifest_sha256"])
    e.bind(f"{base}/profiles_freeze.json", summary["profiles_freeze_sha256"])
    require(summary["implementation_sha256"] == freeze["implementation_sha256"], "Retrieval scoring source changed after profile freeze")
    source = e.lines("results/retrieval/records.jsonl")
    selected = [r for r in source if r["run_id"] in summary["run_ids"]]
    require(len(selected) == 240 and Counter(r["model"] for r in selected) == {"fashion_siglip": 120, "siglip2": 120}, "Retrieval requires120 source attempts per model")
    require(len({(r["model"], r["case_id"]) for r in selected}) == 240 and all(r["status"] == "success" for r in selected), "Duplicate/missing/failed retrieval source query")
    scores = e.lines(f"{base}/query_scores.jsonl")
    lookup = {(r["model"], r["query_id"]): r for r in scores}
    require(len(scores) == len(lookup) == 240, "Retrieval requires240 unique query scores")
    for record in selected:
        e.artifacts(record)
        prediction = e.json(record["artifacts"]["prediction"])
        require(prediction["query_id"] == record["case_id"], "Retrieval prediction query differs")
        row = lookup[(record["model"], record["case_id"])]
        require(row["original_run_id"] == record["run_id"] and row["source_record_sha256"] == canonical_hash(record), "Proxy score lost original raw source anchor")
        require(row["source_prediction_sha256"] == record["artifact_sha256"]["prediction"] and row["profiles_sha256"] == freeze["profiles_sha256"] and row["proxy_qrels_sha256"] == qmanifest["qrels_sha256"], "Proxy score evidence anchor differs")
        equal(row["metrics"], ranking_metrics(prediction["ranking"], by_query[record["case_id"]]), "Proxy query scores cannot be regenerated")
    equal(summary["paired_comparisons"], clustered_paired_bootstrap(scores), "Retrieval paired family bootstrap differs")
    require(summary["n_queries"] == 120 and summary["n_model_queries"] == 240, "Retrieval summary denominator differs")
    require(len(summary["rows"]) == 2 and {r["model"] for r in summary["rows"]} == {"fashion_siglip", "siglip2"}, "Retrieval model aggregate is missing/duplicated")
    for row in summary["rows"]:
        subset = [r for r in scores if r["model"] == row["model"]]
        require(row["n_queries"] == len(subset) == 120, "Retrieval model summary denominator differs")
        require(set(row["metrics"]) == {"known_labels_unknown_zero_gain", "all_unknown_grade3_sensitivity", "conservative_bounds"}, "Retrieval sensitivity scenario aggregate is missing")
        for scenario, metrics in row["metrics"].items():
            require(set(metrics) == set(subset[0]["metrics"][scenario]), "Retrieval aggregate metric set differs")
            for name, value in metrics.items():
                values = [r["metrics"][scenario][name] for r in subset if r["metrics"][scenario][name] is not None]
                equal(value, {"mean": sum(values) / len(values) if values else None, "n_assessable": len(values), "n_missing": 120 - len(values)}, "Retrieval aggregate metric differs")
        require(math.isclose(row["mean_judged_coverage_at_10"], sum(r["metrics"]["judged_coverage_at_10"] for r in subset) / 120), "Retrieval top10 coverage differs")
        require(row["n_queries_all_catalogue_complete"] == sum(r["metrics"]["all_catalogue_complete"] for r in subset), "Retrieval complete-catalogue count differs")
    counts.update(retrieval_product_profiles=600, retrieval_proxy_pairs=len(qrels), retrieval_query_scores=len(scores), retrieval_queries_per_model=120)
    return selected


def _technical(e, counts, selected):
    from stylesphere.common.config import load_config
    from .verification import execution_coverage
    all_raw = []
    for name, digest in ORIGINAL_RAW_SHA256.items():
        e.bind(name, digest)
        all_raw.extend(e.lines(name))
    require(len(all_raw) == 928, "Original928 raw record set changed")
    require(len(selected) == len({canonical_hash(r) for r in selected}) == 584, "All584 authorized first attempts must be identified exactly once")
    require(Counter(r["model"] for r in selected) == EXPECTED_MODELS, "Authorized eight-model technical case counts differ")
    require(Counter(r["status"] for r in selected) == {"success": 511, "failed": 73}, "Technical outcomes must retain all73 protocol failures")
    require(all(r.get("mode") == "evaluation" and r.get("provenance") == "model_inference" for r in selected), "Synthetic/preflight/smoke records cannot substitute for authorized evaluations")
    for record in selected:
        e.artifacts(record)
    for path in (e.root / "config").glob("*.yaml"):
        e.bind(path)
    for path in ("src/stylesphere/common/config.py", "src/stylesphere/reporting/verification.py"):
        e.bind(path)
    coverage = execution_coverage(e.root, load_config(e.root), all_raw)
    require(len(coverage) == 8 and {r["model"] for r in coverage} == set(EXPECTED_MODELS), "Current authorized scope is not exactly eight models")
    require(all(r["planned_cases"] == EXPECTED_MODELS[r["model"]] and r["unexecuted_cases"] == 0 for r in coverage), "Current inputs/settings have missing planned technical cases")
    counts.update(original_raw_records=len(all_raw), planned_technical_attempts=len(selected), technical_successes=511, technical_protocol_failures=73)


def _human_separation(e, counts):
    # These are currently uncollected templates. Explicitly forbid silently
    # copying supplementary AI values into columns labelled as human ratings.
    files = list((e.root / "results/generation").rglob("ratings.csv"))
    files.append(e.root / "results/assistance/human_assessments.csv")
    for path in files:
        rows = e.csv(path)
        identity = {"review_id", "run_id", "model", "case_id", "seed", "record_sha256"}
        require(all(not str(value or "").strip() for row in rows for key, value in row.items() if key not in identity), "Human-labelled worksheet contains values; independent provenance needs a separate validation protocol")
    counts["human_rating_values_created"] = 0


def _reference(e, counts):
    from stylesphere.reconstruction.reference_benchmark import check_freeze, validate_raw
    freeze = e.json("environment/reference_benchmark_freeze.json")
    e.mapping("", freeze["source_sha256"])
    e.mapping("", freeze["asset_sha256"])
    e.mapping(freeze["source_snapshot"], freeze["source_sha256"])
    design = check_freeze(e.root)
    authorization = e.json("environment/reference_benchmark_test_authorization.json")
    require(authorization["design_freeze_sha256"] == e.hashes["environment/reference_benchmark_freeze.json"] and authorization["design_freeze_id"] == design["freeze_id"], "Reference test authorization lost frozen design binding")
    e.mapping("", authorization["development_evidence_sha256"])
    all_scores, all_raw = [], []
    for split, expected in (("development", 4), ("test", 20)):
        folder = "datasets/reconstruction_reference" + ("/development" if split == "development" else "")
        cases = e.csv(f"{folder}/objects.csv")
        require(len(cases) == len({r["object_id"] for r in cases}) == expected, "Reference split case coverage differs")
        raw = e.lines(f"results/reference_{split}/reconstruction/records.jsonl")
        scores = e.lines(f"results/reference_{split}/reference_metrics.jsonl")
        identities = {(m, row["object_id"]) for m in ("sf3d", "spar3d") for row in cases}
        require(len(raw) == len(identities) and {(r["model"], r["case_id"]) for r in raw} == identities, "Missing/duplicate reference raw model/case execution")
        require(all(r["status"] == "success" and r.get("mode") == "evaluation" and r.get("provenance") == "model_inference" for r in raw), "Reference computational runs are unfinished or nongenuine")
        require(len(scores) == len(identities) and {(r["model"], r["case_id"]) for r in scores} == identities and all(r["status"] == "success" for r in scores), "Reference scores are pending, duplicated or failed")
        lookup = {canonical_hash(row): row for row in raw}
        for record in raw:
            e.artifacts(record)
            manifest = validate_raw(e.root, split, record, design)
            expected_row = next(r for r in cases if r["object_id"] == record["case_id"])
            equal(manifest, expected_row, "Reference raw descriptor is not the frozen split row")
        for score in scores:
            require(score["raw_record_sha256"] in lookup, "Reference score missing exact raw record anchor")
            raw_record = lookup[score["raw_record_sha256"]]
            require(score["model"] == raw_record["model"] and score["case_id"] == raw_record["case_id"] and score["input_sha256"] == raw_record["input_sha256"] and score["split"] == split, "Reference score/raw identity differs")
            require(score["freeze_id"] == design["freeze_id"] and score["protocol_sha256"] == design["metric_protocol_sha256"], "Reference score design/protocol differs")
            require(score["metrics"] and all(type(value) in (int, float) and math.isfinite(value) for value in score["metrics"].values()), "Reference metrics absent/nonfinite")
            e.mapping("", score["artifact_sha256"])
        summary = e.json(f"results/reference_{split}/reference_summary.json")
        require(summary["raw_attempts"] == expected * 2 and summary["paired_objects"] == expected, "Reference summary case denominator differs")
        for model in ("sf3d", "spar3d"):
            subset = [row for row in scores if row["model"] == model]
            require(summary["models"][model]["attempts"] == summary["models"][model]["success"] == summary["models"][model]["scored_objects"] == expected, "Reference model summary denominator differs")
            for name, value in summary["models"][model]["mean_metrics"].items():
                require(math.isclose(value, sum(row["metrics"][name] for row in subset) / expected, rel_tol=1e-12, abs_tol=1e-12), "Reference mean metric differs from retained per-case scores")
        all_scores.extend(scores)
        all_raw.extend(raw)
    # The independent audit verifies actual meshes/sampling/transforms/cameras,
    # beyond this validator's case, hash, provenance and aggregation checks.
    audit = e.json("environment/reference_geometry_audit.json")
    require(audit.get("integrity_passed") is True and not audit.get("errors"), "Independent reference geometry audit is pending or failed")
    require(audit.get("freeze_id") == design["freeze_id"] and audit.get("freeze_precedes_development") is True
            and audit.get("test_authorization_precedes_all_test_records") is True, "Reference audit design/authorization chronology differs")
    audit_required = {"environment/reference_benchmark_freeze.json", "environment/reference_benchmark_test_authorization.json",
                      "config/reconstruction_reference_protocol.json", "datasets/reconstruction_reference/objects.csv",
                      "datasets/reconstruction_reference/development/objects.csv", "datasets/reconstruction_reference/preparation_manifest.json",
                      "scripts/verify_reference_geometry.py", "src/stylesphere/reconstruction/reference_benchmark.py"}
    audit_required.update(f"results/reference_{split}/{name}" for split in ("development", "test")
                          for name in ("reconstruction/records.jsonl", "reference_metrics.jsonl", "reference_summary.json", "reference_per_case.csv"))
    require(audit_required <= set(audit.get("source_sha256", {})), "Independent reference audit omits required source bindings")
    e.mapping("", audit["source_sha256"])
    for split, expected in (("development", 8), ("test", 40)):
        section = audit["splits"][split]
        require(section.get("coverage_complete") is True and section.get("raw_attempts") == section.get("scored_objects_checked") == section.get("expected_executions") == expected, "Independent geometry audit case coverage differs")
    integrity = e.json("environment/reference_benchmark_integrity.json")
    require(integrity.get("integrity_passed") is True and not integrity.get("errors") and integrity.get("scored_executions_checked") == 48 and integrity.get("freeze_id") == design["freeze_id"], "Reference metric recomputation receipt is pending/stale")
    counts.update(reference_development_attempts=8, reference_test_attempts=40, reference_test_scored=40, reference_test_paired_objects=20)


def inspect_completion(root: Path) -> dict:
    """Read-only current validation; never accept receipt booleans as evidence."""
    e, counts, errors, checks = Evidence(root), {}, [], []
    e.bind(Path(__file__))
    selected = []
    for name, function in (("visual_ai_quality", _visual), ("assistance_ai_quality", _assistance), ("retrieval_ai_proxy_quality", _retrieval)):
        try:
            selected.extend(function(e, counts))
            checks.append({"name": name, "passed": True})
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            checks.append({"name": name, "passed": False, "error": message})
    for name, function in (("authorized_technical_scope", lambda: _technical(e, counts, selected)),
                           ("human_ai_separation", lambda: _human_separation(e, counts)),
                           ("normalized_reference_extension", lambda: _reference(e, counts))):
        try:
            function()
            checks.append({"name": name, "passed": True})
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            checks.append({"name": name, "passed": False, "error": message})
    for key, expected in REQUIRED_COUNTS.items():
        if counts.get(key) != expected:
            errors.append(f"Required completion count differs: {key}={counts.get(key)!r}, expected{expected}")
    errors.extend(e.stable())
    return {"schema_version": 1, "validated_at_utc": utc_now(),
            "scope": "User-authorized four pairs/eight models, Codes+Data, AI-assisted quality evaluation and genuine normalized scanned-reference extension",
            "requested_component_complete": not errors, "legacy_benchmark_complete": False,
            "legacy_reason": "The legacy primary protocol requires independent human outcomes and absolute/aligned primary geometry; supplementary AI judgements and explicitly normalized/reference-assisted geometry remain separately labelled.",
            "counts": counts, "required_counts": REQUIRED_COUNTS, "checks": checks,
            "errors": errors, "source_sha256": dict(sorted(e.hashes.items())),
            "qualification": "Completion means all authorized attempts and assessments are accounted for, including73 genuine protocol failures and retained AI uncertainty; it does not mean every model answered correctly or every subjective judgement is certain."}


def finalize_completion(root: Path) -> dict:
    root = Path(root).resolve()
    receipt = inspect_completion(root)
    write_json(root / RECEIPT, receipt)
    return receipt


def validate_completion_receipt(root: Path) -> list[str]:
    """Packaging API: recheck hashes, counts and complete current evidence."""
    root = Path(root).resolve()
    errors = []
    try:
        receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
        require(receipt.get("schema_version") == 1 and receipt.get("requested_component_complete") is True and receipt.get("legacy_benchmark_complete") is False, "Completion receipt is pending or has invalid completion semantics")
        require(receipt.get("errors") == [], "Completion receipt retains validation errors")
        require(receipt.get("required_counts") == REQUIRED_COUNTS and all(receipt.get("counts", {}).get(key) == value for key, value in REQUIRED_COUNTS.items()), "Completion receipt count requirements differ")
        require({check.get("name") for check in receipt.get("checks", [])} == REQUIRED_CHECKS and len(receipt["checks"]) == len(REQUIRED_CHECKS)
                and all(check.get("passed") is True for check in receipt["checks"]), "Completion receipt has missing or failed checks")
        e = Evidence(root)
        e.mapping("", receipt["source_sha256"])
        current = inspect_completion(root)
        errors.extend(current["errors"])
        require(current["requested_component_complete"] is True, "Current requested component is incomplete")
        equal(current["counts"], receipt["counts"], "Completion receipt counts are stale")
        equal(current["source_sha256"], receipt["source_sha256"], "Completion receipt evidence set/hashes are stale")
    except Exception as exc:
        errors.append(f"Completion receipt: {type(exc).__name__}: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "finalize", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "verify":
        errors = validate_completion_receipt(args.root)
        result = {"integrity_passed": not errors, "errors": errors}
    else:
        result = finalize_completion(args.root) if args.command == "finalize" else inspect_completion(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
