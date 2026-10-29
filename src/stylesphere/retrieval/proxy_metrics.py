"""Validate authored product profiles and derive explicitly AI-labelled qrels.

Profile preparation/freezing never reads retrieval predictions. Only the separate
score command opens saved rankings, after all 600 authored profiles are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from stylesphere.common.io import canonical_hash, portable_path, sha256_file, utc_now, write_json
from .ai_assessment import PROTOCOL, SOURCE_FIELDS, TRAIT_KEYS, TYPE_KEYS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_profiles(evidence: list[dict], profiles: list[dict], *, complete=True) -> None:
    lookup = {p["product_id"]: p for p in evidence}
    seen = set()
    for profile in profiles:
        key = profile["product_id"]
        if key not in lookup or key in seen:
            raise ValueError("Unknown or duplicate annotated product")
        seen.add(key)
        packet = lookup[key]
        if profile.get("image_sha256") != packet["image_sha256"] or profile.get("annotation_index") != packet["annotation_index"]:
            raise ValueError("Profile/image identity differs from evidence")
        for field in ("matched_types", "uncertain_types"):
            values = profile.get(field)
            if not isinstance(values, list) or len(set(values)) != len(values) or not set(values) <= set(TYPE_KEYS):
                raise ValueError("Invalid authored type labels")
        if set(profile["matched_types"]) & set(profile["uncertain_types"]):
            raise ValueError("Matched and uncertain types must be disjoint")
        traits = profile.get("traits", {})
        if set(traits) != set(TRAIT_KEYS) or any(value is not None and (type(value) is not int or value not in {0, 1, 2}) for value in traits.values()):
            raise ValueError("All ten visual traits require explicit 0/1/2/null labels")
        if profile.get("confidence") not in {"high", "medium", "low"} or not str(profile.get("visual_notes", "")).strip():
            raise ValueError("Authored visual evidence and confidence required")
        lightweight = profile.get("explicit_lightweight_source")
        if lightweight is not None and type(lightweight) is not bool:
            raise ValueError("Lightweight support must be Boolean or null")
        citations = profile.get("source_support")
        if not isinstance(citations, list):
            raise ValueError("Source citations must be an explicit list")
        for citation in citations:
            field = citation.get("field")
            if field not in SOURCE_FIELDS:
                raise ValueError("Invalid source citation field")
            values = citation.get("exact_source_values", [citation])
            if not isinstance(values, list) or not values:
                raise ValueError("Source citations require exact original values")
            for cited in values:
                exact = {k: cited[k] for k in ("language_tag", "value") if k in cited}
                if "value" not in exact or not any(all(value.get(k) == v for k, v in exact.items()) for value in packet["source_facts"][field]):
                    raise ValueError("Source citation is not an exact retained multilingual value")
        if lightweight is True and not citations:
            raise ValueError("Explicit lightweight claim requires an authored exact source citation")
    if complete and seen != set(lookup):
        raise ValueError("Every catalogue product requires an explicit authored profile")


def materialize_ledger(evidence: list[dict], ledger: str, *, assessed_at_utc: str) -> list[dict]:
    """Expand this assessor's explicit per-image ledger; never infer trait grades."""
    lookup = {p["annotation_index"]: p for p in evidence}
    profiles = []
    lightweight_indices = set()
    for line in ledger.splitlines():
        if line.startswith("# explicit_lightweight_source_indices:"):
            lightweight_indices.update(int(value) for value in line.split(":", 1)[1].split(",") if value.strip())
    for line in ledger.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 6:
            raise ValueError("Malformed authored visual ledger")
        index, matched, uncertain, labels, confidence, notes = parts
        packet = lookup[int(index)]
        traits = dict.fromkeys(TRAIT_KEYS, 0)
        for token in labels.split():
            name, value = token[:-1], token[-1]
            if name not in TRAIT_KEYS:
                raise ValueError(f"Unknown explicitly authored trait: {name}")
            traits[name] = None if value == "?" else int(value)
        names = packet["source_facts"]["item_name"]
        english = [n for n in names if n.get("language_tag", "").startswith("en_")]
        selected = (english or names)[0] if names else None
        citation = [{"field": "item_name", **selected, "role": "Exact source context read beside photograph; visible traits remain separately authored."}] if selected else []
        profiles.append({"annotation_index": packet["annotation_index"], "product_id": packet["product_id"],
                         "image_sha256": packet["image_sha256"],
                         "matched_types": [] if matched == "-" else matched.split(","),
                         "uncertain_types": [] if uncertain == "-" else uncertain.split(","),
                         "traits": traits, "explicit_lightweight_source": True if int(index) in lightweight_indices else None,
                         "confidence": confidence, "visual_notes": notes,
                         "source_support": citation, "assessor_id": "codex-retrieval-assistance",
                         "assessed_at_utc": assessed_at_utc, "assessment_kind": "AI_visual_source_proxy_not_human_ground_truth"})
    validate_profiles(evidence, profiles, complete=False)
    return profiles


def check_evidence(root: Path, directory: Path) -> list[dict]:
    manifest = json.loads((directory / "evidence_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "assessment_protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((directory / "protocol_freeze.json").read_text(encoding="utf-8"))
    if protocol != PROTOCOL or freeze["protocol_sha256"] != canonical_hash(protocol):
        raise ValueError("Frozen rating protocol differs")
    path = directory / "product_evidence.jsonl"
    if sha256_file(path) != manifest["product_evidence_sha256"]:
        raise ValueError("Product evidence hash differs")
    evidence = read_jsonl(path)
    if canonical_hash(evidence) != manifest["product_evidence_canonical_sha256"]:
        raise ValueError("Canonical product evidence differs")
    paths = {"catalogue": "datasets/retrieval/catalogue.csv", "queries": "datasets/retrieval/queries.csv",
             "original_human_qrels": "datasets/retrieval/qrels.csv", "multilingual_source": ".cache/abo/listings_0.json.gz"}
    for key, location in paths.items():
        if key == "multilingual_source" and not (root / location).exists():
            # Portable exports retain every assessed exact listing in the hashed
            # evidence packets, while the original compressed download is cache.
            if not all(packet.get("full_source_listings") for packet in evidence):
                raise ValueError("Neither original listing cache nor retained full listings are available")
            continue
        if sha256_file(root / location) != manifest["source_sha256"][key]:
            raise ValueError(f"Original proxy source changed: {key}")
    for packet in evidence:
        if sha256_file(root / packet["image_path"]) != packet["image_sha256"]:
            raise ValueError("Actual catalogue photo changed")
    for name, digest in manifest["sheet_sha256"].items():
        if sha256_file(directory / "sheets" / name) != digest:
            raise ValueError("Visual assessment sheet changed")
    return evidence


def freeze_profiles(root: Path, directory: Path) -> dict:
    evidence = check_evidence(root, directory)
    paths = sorted((directory / "annotations").glob("*.jsonl"))
    profiles = [row for path in paths for row in read_jsonl(path)]
    validate_profiles(evidence, profiles)
    profiles.sort(key=lambda p: p["annotation_index"])
    target = directory / "frozen_product_profiles.jsonl"
    content = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in profiles)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("Profiles already frozen differently; use a new version")
    target.write_text(content, encoding="utf-8")
    manifest = {"schema_version": 1, "frozen_at_utc": utc_now(), "n_products": len(profiles), "rankings_read": False,
                "protocol_sha256": canonical_hash(PROTOCOL), "profiles_sha256": sha256_file(target),
                "annotation_source_sha256": {p.name: sha256_file(p) for p in paths},
                "evidence_manifest_sha256": sha256_file(directory / "evidence_manifest.json"),
                "implementation_sha256": sha256_file(Path(__file__))}
    receipt = directory / "profiles_freeze.json"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if any(old.get(key) != manifest[key] for key in ("profiles_sha256", "protocol_sha256", "annotation_source_sha256", "evidence_manifest_sha256")):
            raise ValueError("Prior frozen annotation receipt differs")
        return old
    write_json(receipt, manifest)
    return manifest


def proxy_grade(profile: dict, query_type: str, trait: str) -> int | None:
    if query_type in profile["uncertain_types"]:
        return None
    if query_type not in profile["matched_types"]:
        return 0
    value = profile["traits"][trait]
    if value is None:
        return None
    if value == 2:
        return 2 if trait == "travel_appearance" and profile["explicit_lightweight_source"] is not True else 3
    return value


def derive_qrels(root: Path, directory: Path) -> dict:
    evidence = check_evidence(root, directory)
    receipt = json.loads((directory / "profiles_freeze.json").read_text(encoding="utf-8"))
    path = directory / "frozen_product_profiles.jsonl"
    if sha256_file(path) != receipt["profiles_sha256"]:
        raise ValueError("Frozen product profile hash differs")
    profiles = read_jsonl(path)
    validate_profiles(evidence, profiles)
    with (root / "datasets/retrieval/queries.csv").open(encoding="utf-8-sig", newline="") as stream:
        queries = list(csv.DictReader(stream))
    rows, coverage = [], []
    for query in queries:
        kind = query["query_category"].replace(" ", "_")
        if kind not in TYPE_KEYS:
            raise ValueError("Query type is outside frozen rubric")
        index = int(query["query_id"].rsplit("q", 1)[-1])
        trait = TRAIT_KEYS[(index - 1) % 10]
        expected = [f"black {query['query_category']}", f"white {query['query_category']}", f"red {query['query_category']}",
                    f"simple {query['query_category']} for everyday wear", f"{query['query_category']} with a metallic finish",
                    f"{query['query_category']} for formal occasions", f"minimalist {query['query_category']}",
                    f"lightweight {query['query_category']} for travel", f"{query['query_category']} with a bold decorative design",
                    f"{query['query_category']} in neutral colours"][(index - 1) % 10]
        if query["query_text"] != expected:
            raise ValueError("Query text differs from fixed modifier mapping")
        grades = [proxy_grade(p, kind, trait) for p in profiles]
        rows.extend({"query_id": query["query_id"], "product_id": p["product_id"], "grade": grade,
                     "query_type": kind, "trait": trait, "assessment_kind": "AI_proxy_not_human_qrels"}
                    for p, grade in zip(profiles, grades))
        counts = Counter("unknown" if grade is None else str(grade) for grade in grades)
        coverage.append({"query_id": query["query_id"], "query_text": query["query_text"], "n_catalogue": len(profiles),
                         "n_judged": sum(g is not None for g in grades), "n_unknown": sum(g is None for g in grades),
                         "known_useful_relevant": sum(g is not None and g >= 2 for g in grades),
                         "all_catalogue_complete": all(g is not None for g in grades), "grade_counts": dict(counts),
                         "physical_weight_measured": False if trait == "travel_appearance" else None})
    target = directory / "proxy_qrels.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(directory / "query_coverage.json", {"schema_version": 1, "n_queries": len(queries), "n_pairs": len(rows), "rows": coverage})
    manifest = {"schema_version": 1, "generated_at_utc": utc_now(), "profiles_sha256": receipt["profiles_sha256"],
                "protocol_sha256": canonical_hash(PROTOCOL), "qrels_sha256": sha256_file(target),
                "query_coverage_sha256": sha256_file(directory / "query_coverage.json"), "n_pairs": len(rows),
                "human_ground_truth": False, "rankings_used_for_labels": False}
    write_json(directory / "qrels_manifest.json", manifest)
    return manifest


def _scenario_metrics(ranking: list[str], grades: dict[str, int], k: int) -> dict:
    top = ranking[:k]
    relevant = {key for key, grade in grades.items() if grade >= 2}
    hits = sum(key in relevant for key in top)
    dcg = sum((2 ** grades[key] - 1) / math.log2(rank + 2) for rank, key in enumerate(top))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return {"ndcg_at_10": dcg / ideal if ideal else None,
            "mrr_at_10": next((1 / (rank + 1) for rank, key in enumerate(top) if key in relevant), 0.0),
            "hit_at_10": int(hits > 0), "precision_at_10": hits / len(top) if top else None,
            "recall_at_10": hits / len(relevant) if relevant else None,
            "n_relevant": len(relevant)}


def ranking_metrics(ranking: list[str], grades: dict[str, int | None], *, k=10) -> dict:
    if k != 10:
        raise ValueError("The frozen proxy protocol fixes k=10")
    if len(ranking) != len(set(ranking)) or set(ranking) != set(grades):
        raise ValueError("Saved ranking must be an exact catalogue permutation")
    known = {key: value if value is not None else 0 for key, value in grades.items()}
    potential = {key: value if value is not None else 3 for key, value in grades.items()}
    low = _scenario_metrics(ranking, known, k)
    high = _scenario_metrics(ranking, potential, k)
    top = ranking[:k]
    known_dcg = sum((2 ** known[key] - 1) / math.log2(i + 2) for i, key in enumerate(top))
    potential_dcg = sum((2 ** potential[key] - 1) / math.log2(i + 2) for i, key in enumerate(top))
    known_ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(known.values(), reverse=True)[:k]))
    potential_ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(potential.values(), reverse=True)[:k]))
    unknown = sum(v is None for v in grades.values())
    unknown_top = sum(grades[key] is None for key in top)
    return {"known_labels_unknown_zero_gain": low,
            "all_unknown_grade3_sensitivity": high,
            "conservative_bounds": {"ndcg_at_10_lower": known_dcg / potential_ideal if potential_ideal else None,
                                    "ndcg_at_10_upper": min(1.0, potential_dcg / known_ideal) if known_ideal else (1.0 if potential_ideal else None),
                                    "hit_at_10_lower": low["hit_at_10"], "hit_at_10_upper": high["hit_at_10"],
                                    "mrr_at_10_lower": low["mrr_at_10"], "mrr_at_10_upper": high["mrr_at_10"]},
            "n_unknown_catalogue": unknown, "n_unknown_top10": unknown_top,
            "judged_coverage_at_10": (len(top) - unknown_top) / len(top) if top else None,
            "all_catalogue_complete": unknown == 0,
            "recall_denominator": "known AI-proxy useful relevant items only; not objective or human recall"}


def clustered_paired_bootstrap(scored: list[dict], *, seed=8410, repetitions=2000) -> list[dict]:
    """Resample related query variants together in their12 type families."""
    import random
    lookup = {(row["model"], row["query_id"]): row for row in scored}
    queries = sorted({row["query_id"] for row in scored})
    comparisons = []
    for scenario in ("known_labels_unknown_zero_gain", "all_unknown_grade3_sensitivity"):
        for metric in ("ndcg_at_10", "mrr_at_10", "hit_at_10", "precision_at_10", "recall_at_10"):
            families = defaultdict(list)
            for query in queries:
                first = lookup[("fashion_siglip", query)]["metrics"][scenario][metric]
                second = lookup[("siglip2", query)]["metrics"][scenario][metric]
                if first is not None and second is not None:
                    families[(int(query.rsplit("q", 1)[-1]) - 1) // 10].append(second - first)
            keys = sorted(families)
            actual = [value for key in keys for value in families[key]]
            rng = random.Random(seed)
            samples = []
            for _ in range(repetitions):
                drawn = [value for key in rng.choices(keys, k=len(keys)) for value in families[key]] if keys else []
                if drawn:
                    samples.append(sum(drawn) / len(drawn))
            samples.sort()
            def percentile(fraction):
                if not samples:
                    return None
                position = fraction * (len(samples) - 1)
                lower = int(position)
                upper = min(lower + 1, len(samples) - 1)
                return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)
            comparisons.append({"scenario": scenario, "metric": metric, "difference": "siglip2_minus_fashion_siglip",
                                "paired_mean_difference": sum(actual) / len(actual) if actual else None,
                                "cluster_bootstrap_95_percentile_interval": [percentile(.025), percentile(.975)],
                                "n_paired_queries": len(actual), "n_assessable_query_family_clusters": len(keys),
                                "n_planned_query_family_clusters": 12, "bootstrap_repetitions": repetitions,
                                "bootstrap_seed": seed,
                                "interpretation": "Conditional on these AI proxy labels and this query design;12 related query families are the resampling units, not120 independent users."})
    return comparisons


def score_outputs(root: Path, directory: Path, run_ids: list[str]) -> dict:
    """Read rankings only after independently authored profiles are frozen."""
    check_evidence(root, directory)
    freeze = json.loads((directory / "profiles_freeze.json").read_text(encoding="utf-8"))
    if sha256_file(directory / "frozen_product_profiles.jsonl") != freeze["profiles_sha256"]:
        raise ValueError("Profiles changed after rating freeze")
    qrels_manifest = json.loads((directory / "qrels_manifest.json").read_text(encoding="utf-8"))
    if qrels_manifest["profiles_sha256"] != freeze["profiles_sha256"] or sha256_file(directory / "proxy_qrels.jsonl") != qrels_manifest["qrels_sha256"]:
        raise ValueError("Derived proxy qrels changed")
    qrels = defaultdict(dict)
    for row in read_jsonl(directory / "proxy_qrels.jsonl"):
        if row["product_id"] in qrels[row["query_id"]]:
            raise ValueError("Duplicate proxy relevance pair")
        qrels[row["query_id"]][row["product_id"]] = row["grade"]
    if len(qrels) != 120 or any(len(rows) != 600 for rows in qrels.values()):
        raise ValueError("Expected complete120-query by600-product proxy matrix including unknowns")
    source = root / "results/retrieval/records.jsonl"
    snapshot = source.read_bytes()
    records = [json.loads(line) for line in snapshot.decode("utf-8-sig").splitlines() if line.strip()]
    selected = [r for r in records if r.get("run_id") in set(run_ids)]
    if not selected or {r["run_id"] for r in selected} != set(run_ids):
        raise ValueError("Explicit source run identifiers do not exist")
    if len(selected) != 240 or any(r.get("status") != "success" or r.get("case_id") not in qrels for r in selected):
        raise ValueError("Select exactly240 successful frozen final-run query attempts")
    identity = [(r["model"], r["case_id"]) for r in selected]
    if len(set(identity)) != 240 or set(r["model"] for r in selected) != {"fashion_siglip", "siglip2"}:
        raise ValueError("Each model requires exactly the same120 queries")
    protocols = {r.get("protocol_sha256") for r in selected}
    inputs = {r.get("input_sha256") for r in selected}
    if len(protocols) != 1 or len(inputs) != 1:
        raise ValueError("Source models do not share the same original protocol and inputs")
    scored = []
    for record in sorted(selected, key=lambda r: (r["model"], r["case_id"])):
        path = portable_path(root, record["artifacts"]["prediction"])
        digest = sha256_file(path)
        if digest != record["artifact_sha256"]["prediction"]:
            raise ValueError("Original saved ranking hash differs")
        frozen_catalogue = portable_path(root, record["artifacts"]["frozen_catalogue"])
        frozen_queries = portable_path(root, record["artifacts"]["frozen_queries"])
        if sha256_file(frozen_catalogue) != sha256_file(root / "datasets/retrieval/catalogue.csv") or sha256_file(frozen_queries) != sha256_file(root / "datasets/retrieval/queries.csv"):
            raise ValueError("Original predictions use different catalogue/query inputs")
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if prediction["query_id"] != record["case_id"]:
            raise ValueError("Original prediction/query identity differs")
        scored.append({"schema_version": 1, "assessment_kind": "supplementary_AI_proxy_not_human_ground_truth",
                       "model": record["model"], "query_id": record["case_id"], "query_text": record["query_text"],
                       "original_run_id": record["run_id"], "source_record_sha256": canonical_hash(record),
                       "source_prediction_sha256": digest, "profiles_sha256": freeze["profiles_sha256"],
                       "proxy_qrels_sha256": qrels_manifest["qrels_sha256"],
                       "metrics": ranking_metrics(prediction["ranking"], qrels[record["case_id"]])})
    target = directory / "query_scores.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8")
    summary_rows = []
    for model in sorted({row["model"] for row in scored}):
        subset = [row for row in scored if row["model"] == model]
        metrics = {}
        for scenario in ("known_labels_unknown_zero_gain", "all_unknown_grade3_sensitivity", "conservative_bounds"):
            metrics[scenario] = {}
            keys = subset[0]["metrics"][scenario]
            for key in keys:
                values = [row["metrics"][scenario][key] for row in subset if row["metrics"][scenario][key] is not None]
                metrics[scenario][key] = {"mean": sum(values) / len(values) if values else None,
                                         "n_assessable": len(values), "n_missing": len(subset) - len(values)}
        coverage = [row["metrics"]["judged_coverage_at_10"] for row in subset]
        summary_rows.append({"model": model, "n_queries": len(subset), "metrics": metrics,
                             "mean_judged_coverage_at_10": sum(coverage) / len(coverage),
                             "n_queries_all_catalogue_complete": sum(row["metrics"]["all_catalogue_complete"] for row in subset)})
    import hashlib
    report = {"schema_version": 1, "generated_at_utc": utc_now(), "assessment_kind": "supplementary_AI_proxy_not_human_ground_truth",
              "n_queries": 120, "n_model_queries": len(scored), "run_ids": run_ids,
              "rows": summary_rows, "source_records_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
              "paired_comparisons": clustered_paired_bootstrap(scored),
              "score_sha256": sha256_file(target), "qrels_manifest_sha256": sha256_file(directory / "qrels_manifest.json"),
              "profiles_freeze_sha256": sha256_file(directory / "profiles_freeze.json"),
              "implementation_sha256": sha256_file(Path(__file__)),
              "limitations": [PROTOCOL["assessor"], "Scores are AI visual/source proxies, not human relevance or objective ground truth.",
                              "Lightweight/travel is an appearance proxy with explicit-source exception; weight/performance were not measured.",
                              "Unknowns remain null in qrels; main scores use known labels with zero unknown gain and disclose top10 coverage.",
                              "All-unknown0 and all-unknown3 are sensitivity scenarios, while separately reported conservative nDCG bounds allow independent unknown assignments.",
                              "Recall uses only known AI-proxy relevant items; no original primary human quality field is populated."]}
    write_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["materialize-assistant", "freeze", "qrels", "score"])
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--directory", type=Path, default=Path("results/retrieval/ai_assessment/v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.directory if args.directory.is_absolute() else root / args.directory
    if args.command == "materialize-assistant":
        evidence = check_evidence(root, directory)
        ledger = directory / "annotations/assistant-13-24.txt"
        profiles = materialize_ledger(evidence, ledger.read_text(encoding="utf-8"), assessed_at_utc=utc_now())
        target = directory / "annotations/assistant-13-24.jsonl"
        if target.exists():
            raise ValueError("Preserve authored profile timestamp; do not overwrite an existing materialization")
        target.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in profiles), encoding="utf-8")
        result = {"n_products": len(profiles)}
    elif args.command == "freeze":
        result = freeze_profiles(root, directory)
    elif args.command == "qrels":
        result = derive_qrels(root, directory)
    else:
        if not args.run_id:
            parser.error("score requires explicit --run-id (repeat for multiple source runs)")
        result = score_outputs(root, directory, args.run_id)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
