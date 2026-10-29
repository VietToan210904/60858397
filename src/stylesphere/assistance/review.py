"""Explicit human adjudication of prose, separate from deterministic claim checks."""
import csv
from collections import defaultdict
from pathlib import Path
from stylesphere.common.io import canonical_hash

FIELDS = ["run_id", "model", "case_id", "seed", "record_sha256", "rater_id", "task_success", "factual_prose_correct", "unsupported_claim_present", "error_categories", "notes"]
OUTCOMES = ("task_success", "factual_prose_correct", "unsupported_claim_present")


def review_assistance(records, path):
    """Append blank rows for new successes, validate actual ratings, summarize by case.

    A completed row requires rater identity and a hash matching retained execution.
    No phrase matching or automatic prose rating is performed.
    """
    path = Path(path)
    eligible = [r for r in records if r.get("pair") == "assistance" and r.get("status") == "success" and r.get("mode") == "evaluation"]
    lookup = {canonical_hash(r): r for r in eligible}
    existing = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(FIELDS) - set(reader.fieldnames or []):
                raise ValueError("Assistance review worksheet is missing required columns")
            existing = list(reader)
    present = {row["record_sha256"] for row in existing}
    appended = [{**{field: "" for field in FIELDS}, **{field: r[field] for field in ("run_id", "model", "case_id", "seed")}, "record_sha256": digest} for digest, r in lookup.items() if digest not in present]
    if appended or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing + appended)
    cases = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in existing:
        if not any(str(row.get(field, "")).strip() for field in OUTCOMES):
            continue
        digest, rater = row["record_sha256"], row["rater_id"].strip()
        if digest not in lookup or not rater:
            raise ValueError("Human assessments must reference a retained execution and identify the rater")
        record = lookup[digest]
        if any(str(row[field]) != str(record[field]) for field in ("run_id", "model", "case_id", "seed")):
            raise ValueError("Assessment identifiers differ from the referenced execution")
        if (digest, rater) in seen:
            raise ValueError("Duplicate assistance execution/rater assessment")
        seen.add((digest, rater))
        for field in OUTCOMES:
            value = str(row.get(field, "")).strip()
            if value:
                if value not in {"0", "1"}:
                    raise ValueError(f"Human {field} must be0,1 or blank")
                key = (record["model"], record["case_id"], record.get("protocol_sha256"), record.get("input_sha256"))
                cases[key][field].append(int(value))
    per_case = [{"model": key[0], "case_id": key[1], "protocol_sha256": key[2], "input_sha256": key[3],
                 "metrics": {field: values[0] for field, values in ratings.items() if len(set(values)) == 1},
                 "supplemental_vote_rates": {field: sum(values) / len(values) for field, values in ratings.items()},
                 "disagreement_fields": [field for field, values in ratings.items() if len(set(values)) != 1],
                 "n_votes": {field: len(values) for field, values in ratings.items()}}
                for key, ratings in cases.items()]
    models = defaultdict(lambda: defaultdict(list))
    for case in per_case:
        for field, value in case["metrics"].items():
            models[(case["model"], case["protocol_sha256"])][field].append(value)
        for field, value in case["supplemental_vote_rates"].items():
            models[(case["model"], case["protocol_sha256"])][field + "_vote_rate"].append(value)
    summaries = [{"model": key[0], "protocol_sha256": key[1], "metrics": {field: {"mean": sum(values) / len(values), "n_scenarios": len(values)} for field, values in ratings.items()}} for key, ratings in models.items()]
    return {"source": "independent human assessments; blank rows are not observations", "worksheet": str(path),
            "binary_decision_rule": "unanimous nonblank votes per scenario; disagreement missing pending documented human adjudication; vote rates supplementary",
            "eligible_executions": len(lookup), "n_assessments": len(seen), "models": summaries, "per_case": per_case}
