"""Supplementary, explicitly AI-authored semantic review of immutable traces.

This module prepares evidence and validates/aggregates supplied judgements. It
never derives free-form semantic grades from keyword matches, calls a model,
repairs a benchmark action, or writes the human assessment worksheet.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from stylesphere.common.io import canonical_hash, portable_path, sha256_file, utc_now, write_json
from .tools import knowledge_facts
from .verification import verify_trace

RUNS = {"phi4_mini": "d9923b2e32d54eb0955cbc11cc801808",
        "qwen35_4b": "ae13ded36915404e929fa8428f956235"}
PROTOCOL = {
    "schema_version": 1, "protocol_id": "supplementary-ai-assistance-v1",
    "purpose": "Post-hoc AI semantic/prose review; separate from the original frozen benchmark and human feedback.",
    "assessor": {"kind": "AI-assisted review by Codex coding assistant", "human": False,
                 "blinded": False, "model_version": "not exposed in this session",
                 "limitations": "The assessor helped implement the controller and had model identities available. No independent human validation or inter-rater reliability is claimed."},
    "units": "All 160 first final-run outputs, including 73 strict-protocol failures. Prose is missing for tool-only or structured-only terminal output.",
    "extraction": "Decode an actual quoted answer scalar when present, even in malformed outer JSON; otherwise retain plain visible prose. Never accept/execute/rewrite the original action.",
    "rubric": {
        "semantic_task": "Ignoring JSON syntax, does the actual response provide the requested correct information or appropriately explain/refuse an unavailable or unauthorized request? Missing requested information and unnecessary refusal fail. No customer answer fails delivery. Judge the actual customer request; the predefined checklist can contain ancillary facts not explicitly requested. Extra assertions are separately reviewed on factual and grounding axes.",
        "factual_prose": "Review actual assertions against frozen KB, controller capabilities and returned observations. A truthful refusal without factual business assertions is not a perfect factual-content score.",
        "unsupported": "An asserted material fact is contradicted by, or cannot be established from, the frozen KB/controller. Missing knowledge is not proof an external-world claim is false.",
        "ungrounded": "A business assertion lacks support in actually returned authorized observations, including a true KB fact that was never retrieved. Controller capability statements may use the system/tool policy.",
        "tool_selection": "Review executed calls and identifiable tool-action proposals in context, including missing necessary calls and scenario visibility restrictions. Correct tool choices that could not execute because the output batches multiple actions remain distinct from successful tool use. A prose promise to check is not an executed call.",
        "requested_facts": "An AI judgement of whether each predefined fact is conveyed in actual customer prose; not a string-presence or structured-path proxy."},
    "invariants": ["No human ratings are created.", "Raw records, traces, frozen runtime and input files remain unchanged.",
                   "No network, external inference API or GPU model call is made by this pipeline.",
                   "A semantically correct malformed response remains an original protocol failure.",
                   "Unknown/unassessable judgements remain missing; no invented ground truth."]}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def extract_prose(visible: str) -> dict:
    """Extract retained text only; this does not repair or re-accept an action."""
    for match in re.finditer(r'"answer"\s*:\s*', visible):
        try:
            value, end = json.JSONDecoder().raw_decode(visible, match.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            return {"text": value, "method": "retained_json_answer_scalar",
                    "raw_span": [match.end(), end]}
    text = visible.strip()
    if text and not text.startswith(("{", "[", "```")):
        return {"text": text, "method": "plain_visible_terminal", "raw_span": [visible.index(text), visible.index(text) + len(text)]}
    return {"text": None, "method": "no_customer_prose", "raw_span": None}


def collect_packets(root: Path, runs: dict | None = None) -> list[dict]:
    root = Path(root)
    runs = runs or RUNS
    packets = []
    for record in read_jsonl(root / "results/assistance/records.jsonl"):
        if runs.get(record.get("model")) != record.get("run_id"):
            continue
        paths = {key: portable_path(root, value) for key, value in record["artifacts"].items()}
        for key, path in paths.items():
            if sha256_file(path) != record["artifact_sha256"][key]:
                raise ValueError(f"Changed source artifact: {record['case_id']}/{key}")
        trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
        kb = json.loads(paths["frozen_knowledge_base"].read_text(encoding="utf-8"))
        scenario = next(s for s in read_jsonl(paths["frozen_scenarios"]) if s["scenario_id"] == record["case_id"])
        errors = verify_trace(trace, scenario, kb)
        if errors:
            raise ValueError(f"Source trace replay failed: {record['case_id']}: {errors}")
        visible = trace["turns"][-1].get("visible_model_output", "")
        digest = canonical_hash(record)
        valid = [call["validated_tool"]["name"] for call in trace["tool_trace"] if call.get("schema_valid")]
        expected = scenario.get("expected_tool_names", [])
        required = scenario["required_facts"]
        packets.append({"assessment_id": "ai-assist-" + digest[:20], "model": record["model"],
                        "case_id": record["case_id"], "run_id": record["run_id"],
                        "source_record_sha256": digest, "trace_sha256": record["artifact_sha256"]["trace"],
                        "input_sha256": record["input_sha256"], "protocol_sha256": record.get("protocol_sha256"),
                        "original_status": record["status"], "original_error": record.get("error"),
                        "source_artifacts": record["artifacts"], "source_artifact_sha256": record["artifact_sha256"],
                        "scenario": scenario, "knowledge_base": kb, "observed_facts": trace["observed_facts"],
                        "terminal_visible_output": visible, "response": extract_prose(visible),
                        "turns": trace["turns"], "tool_trace": trace["tool_trace"],
                        "deterministic": {"protocol_completed": record["status"] == "success",
                                          "tool_call_count": len(trace["tool_trace"]),
                                          "validated_tool_names": valid, "expected_tool_names": expected,
                                          "exact_executed_tool_set_match": set(valid) == set(expected) and all(c.get("schema_valid") for c in trace["tool_trace"]),
                                          "required_fact_count": len(required),
                                          "required_facts_observed": sum(f["path"] in trace["observed_facts"] for f in required)}})
    identities = [(p["model"], p["case_id"]) for p in packets]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate selected source execution")
    return sorted(packets, key=lambda p: (p["case_id"], p["model"]))


def export(root: Path, directory: Path, *, runs=None) -> dict:
    packets = collect_packets(root, runs)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evidence_packets.jsonl"
    content = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("Preserve existing packets; use a new supplementary assessment version")
    path.write_text(content, encoding="utf-8")
    write_json(directory / "assessment_protocol.json", PROTOCOL)
    manifest = {"schema_version": 1, "protocol_id": PROTOCOL["protocol_id"],
                "created_at_utc": utc_now(), "packet_count": len(packets),
                "packets_sha256": sha256_file(path), "protocol_sha256": canonical_hash(PROTOCOL),
                "source_records_snapshot_sha256": sha256_file(Path(root) / "results/assistance/records.jsonl"),
                "source_record_sha256": {p["assessment_id"]: p["source_record_sha256"] for p in packets},
                "implementation_sha256": sha256_file(Path(__file__))}
    write_json(directory / "source_manifest.json", manifest)
    return manifest


def validate_assessments(packets: list[dict], assessments: list[dict]) -> None:
    lookup = {p["assessment_id"]: p for p in packets}
    seen = set()
    for review in assessments:
        key = review["assessment_id"]
        if key not in lookup or key in seen:
            raise ValueError("Unknown or duplicate assessment identifier")
        seen.add(key)
        packet = lookup[key]
        for field in ("source_record_sha256", "trace_sha256", "case_id", "model"):
            if review.get(field) != packet[field]:
                raise ValueError(f"Assessment/source binding differs: {key}/{field}")
        if review.get("ai_semantic_task_verdict") not in {"satisfied", "not_satisfied", "unassessable"}:
            raise ValueError("Invalid semantic task verdict")
        if review.get("ai_factual_prose_verdict") not in {"supported", "contradicted", "unsupported", "no_factual_claims", "no_customer_prose"}:
            raise ValueError("Invalid factual prose verdict")
        if review.get("ai_tool_selection_verdict") not in {"appropriate", "inappropriate", "unassessable"}:
            raise ValueError("Invalid tool selection verdict")
        for field in ("ai_unsupported_claim_present", "ai_ungrounded_claim_present"):
            if review.get(field) is not None and type(review[field]) is not bool:
                raise ValueError(f"Invalid Boolean/missing judgement: {field}")
        required = {f["path"] for f in packet["scenario"]["required_facts"]}
        entailment = review.get("requested_fact_entailment", {})
        if set(entailment) != required or any(v not in {"entailed", "omitted", "contradicted", "uncertain"} for v in entailment.values()):
            raise ValueError("Requested-fact checklist differs from frozen scenario")
        if packet["response"]["text"] is None and review["ai_factual_prose_verdict"] != "no_customer_prose":
            raise ValueError("A prose score requires actual retained customer prose")
        if packet["response"]["text"] is None and review["ai_semantic_task_verdict"] == "satisfied":
            raise ValueError("Missing customer response cannot satisfy delivery")
        if review.get("confidence") not in {"high", "medium", "low"} or not str(review.get("rationale", "")).strip():
            raise ValueError("Explicit confidence and authored rationale are required")
        if not isinstance(review.get("error_categories"), list) or not isinstance(review.get("evidence"), list):
            raise ValueError("Explicit error categories and evidence are required")
        facts = knowledge_facts(packet["knowledge_base"])
        for evidence in review["evidence"]:
            excerpt = evidence.get("excerpt", "")
            if not excerpt or excerpt not in packet["terminal_visible_output"] and excerpt not in (packet["response"]["text"] or ""):
                raise ValueError("Assessment evidence is not an exact retained-output excerpt")
            for path in evidence.get("kb_paths", []):
                if path not in facts:
                    raise ValueError("Assessment cites a nonexistent KB path")
        if not review["evidence"]:
            raise ValueError("At least one actual-output evidence excerpt is required")
    if seen != set(lookup):
        raise ValueError("Every selected execution, including protocol failures, needs an explicit assessment")


def materialize_decisions(packets: list[dict], ledger: str) -> list[dict]:
    """Expand explicit authored decisions; perform no semantic inference."""
    lookup = {(p["model"], p["case_id"]): p for p in packets}
    mappings = [{"S": "satisfied", "F": "not_satisfied", "U": "unassessable"},
                {"S": "supported", "C": "contradicted", "U": "unsupported", "N": "no_factual_claims", "X": "no_customer_prose"},
                {"0": False, "1": True, "-": None}, {"0": False, "1": True, "-": None},
                {"A": "appropriate", "I": "inappropriate", "U": "unassessable"}]
    entailment = {"E": "entailed", "O": "omitted", "C": "contradicted", "U": "uncertain"}
    fields = ["ai_semantic_task_verdict", "ai_factual_prose_verdict", "ai_unsupported_claim_present",
              "ai_ungrounded_claim_present", "ai_tool_selection_verdict"]
    model = None
    reviews = []
    for line in ledger.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            model = line[1:-1]
            continue
        parts = line.split("|")
        if len(parts) != 10 or model is None:
            raise ValueError("Malformed explicit decision ledger")
        number, *codes, checklist, confidence, categories, rationale = parts
        packet = lookup[(model, "assist-" + number)]
        required = packet["scenario"]["required_facts"]
        if len(checklist) != len(required):
            raise ValueError(f"Authored checklist length differs: {model}/{number}")
        review = {field: packet[field] for field in ("assessment_id", "model", "case_id", "source_record_sha256", "trace_sha256")}
        review.update({field: mapping[code] for field, mapping, code in zip(fields, mappings, codes)})
        errors = categories.split(",") if categories else []
        if packet["original_status"] != "success":
            errors.append("original_protocol_failure")
        review.update({"schema_version": 1, "assessment_kind": "supplementary_ai_assessment_not_human_feedback",
                       "requested_fact_entailment": {f["path"]: entailment[c] for f, c in zip(required, checklist)},
                       "confidence": confidence, "error_categories": errors, "rationale": rationale,
                       "evidence": [{"excerpt": packet["response"]["text"] or packet["terminal_visible_output"],
                                     "kb_paths": [f["path"] for f in required],
                                     "basis": "Actual retained customer prose, frozen KB and replayed authorized observations; controller policy for capability statements."}]})
        reviews.append(review)
    validate_assessments(packets, reviews)
    return reviews


def summarize(root: Path, directory: Path) -> dict:
    packets = collect_packets(root)
    retained = read_jsonl(directory / "evidence_packets.jsonl")
    if canonical_hash(packets) != canonical_hash(retained):
        raise ValueError("Retained assessment packets differ from actual immutable evidence")
    manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "assessment_protocol.json").read_text(encoding="utf-8"))
    if canonical_hash(protocol) != manifest["protocol_sha256"] or protocol != PROTOCOL:
        raise ValueError("Supplementary assessment protocol differs")
    if sha256_file(Path(__file__)) != manifest["implementation_sha256"]:
        raise ValueError("Supplementary assessment implementation differs; export a new validated manifest")
    if sha256_file(directory / "evidence_packets.jsonl") != manifest["packets_sha256"]:
        raise ValueError("Supplementary evidence packet hash differs")
    assessments = read_jsonl(directory / "assessments.jsonl")
    validate_assessments(packets, assessments)
    authored = materialize_decisions(packets, (directory / "authored_decisions.txt").read_text(encoding="utf-8"))
    if canonical_hash(authored) != canonical_hash(assessments):
        raise ValueError("Assessment records differ from the explicit authored decision ledger")
    lookup = {p["assessment_id"]: p for p in packets}
    groups = defaultdict(list)
    for review in assessments:
        groups[(review["model"], "all")].append(review)
        groups[(review["model"], lookup[review["assessment_id"]]["scenario"]["evaluation_category"])].append(review)
    rows = []
    for (model, category), reviews in sorted(groups.items()):
        def verdict(field, positive, excluded=()):
            eligible = [r[field] for r in reviews if r[field] not in excluded]
            return {"positive": sum(v == positive for v in eligible), "n_assessable": len(eligible),
                    "n_missing_or_not_applicable": len(reviews) - len(eligible),
                    "rate": sum(v == positive for v in eligible) / len(eligible) if eligible else None}
        rows.append({"model": model, "category": category, "n_executions": len(reviews),
                     "n_customer_prose": sum(lookup[r["assessment_id"]]["response"]["text"] is not None for r in reviews),
                     "original_protocol_completions": sum(lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in reviews),
                     "ai_semantic_task_success": verdict("ai_semantic_task_verdict", "satisfied", {"unassessable"}),
                     "ai_factual_prose_supported": verdict("ai_factual_prose_verdict", "supported", {"no_factual_claims", "no_customer_prose"}),
                     "ai_unsupported_claim_present": verdict("ai_unsupported_claim_present", True, {None}),
                     "ai_ungrounded_claim_present": verdict("ai_ungrounded_claim_present", True, {None}),
                     "ai_tool_selection_appropriate": verdict("ai_tool_selection_verdict", "appropriate", {"unassessable"}),
                     "deterministic_exact_executed_tool_set_match": sum(lookup[r["assessment_id"]]["deterministic"]["exact_executed_tool_set_match"] for r in reviews),
                     "requested_fact_entailment_counts": dict(Counter(v for r in reviews for v in r["requested_fact_entailment"].values())),
                     "deterministic_required_facts_observed": sum(lookup[r["assessment_id"]]["deterministic"]["required_facts_observed"] for r in reviews),
                     "deterministic_required_facts_total": sum(lookup[r["assessment_id"]]["deterministic"]["required_fact_count"] for r in reviews),
                     "ai_semantically_satisfied_original_failures": sum(r["ai_semantic_task_verdict"] == "satisfied" and not lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in reviews),
                     "ai_operational_and_grounded_semantic_success": sum(r["ai_semantic_task_verdict"] == "satisfied" and r["ai_ungrounded_claim_present"] is False and lookup[r["assessment_id"]]["deterministic"]["protocol_completed"] for r in reviews),
                     "error_categories": dict(Counter(error for r in reviews for error in r["error_categories"]))})
    report = {"schema_version": 1, "generated_at_utc": utc_now(), "assessment_kind": "supplementary_ai_assessment_not_human_feedback",
              "protocol_id": PROTOCOL["protocol_id"], "assessor": PROTOCOL["assessor"], "n_executions": len(assessments),
              "rows": rows, "source_hashes": {p.name: sha256_file(p) for p in [directory / "assessment_protocol.json", directory / "source_manifest.json", directory / "evidence_packets.jsonl", directory / "assessments.jsonl", directory / "authored_decisions.txt"]},
              "limitations": ["Post-hoc, nonblind AI review by an implementation contributor; no human validation or independent rater agreement.",
                              "Response-level semantic judgements do not replace original operational status or independently collected human outcomes.",
                              "Unknown facts are not invented; missing prose and not-applicable factual content have explicit denominators.",
                              "The original human worksheet and frozen benchmark records are unchanged."]}
    write_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "materialize", "summarize"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--directory", type=Path, default=Path("results/assistance/ai_assessment/v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.directory if args.directory.is_absolute() else root / args.directory
    if args.command == "export":
        result = export(root, directory)
    elif args.command == "materialize":
        packets = read_jsonl(directory / "evidence_packets.jsonl")
        reviews = materialize_decisions(packets, (directory / "authored_decisions.txt").read_text(encoding="utf-8"))
        (directory / "assessments.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews), encoding="utf-8")
        result = {"n_executions": len(reviews)}
    else:
        result = summarize(root, directory)
    print(json.dumps({"directory": str(directory), "n_executions": result.get("n_executions", result.get("packet_count"))}))


if __name__ == "__main__":
    main()
