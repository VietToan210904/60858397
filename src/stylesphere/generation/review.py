"""Blinded human review. No automatic human-quality judgements are made."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from stylesphere.common.io import portable_path

RATING_FIELDS = ("prompt_adherence", "visual_coherence", "composition", "usability_rating")
ANCHORS = {1: "Unacceptable: major defects", 2: "Poor: substantial revision needed",
           3: "Adequate: some revision needed", 4: "Good: minor revision needed",
           5: "Excellent: ready for intended brief"}


def export_review(records: list[dict], output_dir: Path, seed: int = 8410, root: Path | None = None) -> dict:
    """Export copied, anonymously named images plus blank CSV and static HTML.

    Keep private_mapping.json away from evaluators until scoring is frozen. The
    order is randomized reproducibly; the image filename reveals no model key.
    Existing ratings are never overwritten.
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "images"
    assets.mkdir(exist_ok=True)
    successful = [r for r in records if r.get("status") == "success" and r.get("artifacts", {}).get("image")]
    random.Random(seed).shuffle(successful)
    mapping, rows, cards = [], [], []
    for index, record in enumerate(successful, 1):
        opaque = hashlib.sha256(f"{seed}|{record['run_id']}|{record['model']}|{record['case_id']}|{record['seed']}".encode()).hexdigest()[:16]
        review_id = f"image_{opaque}"
        source = Path(record["artifacts"]["image"])
        if root is not None:
            source = portable_path(root, record["artifacts"]["image"])
        target = assets / f"{review_id}.png"
        # Re-encode without model-bearing PNG metadata.
        with Image.open(source) as im:
            im.convert("RGB").save(target)
        mapping.append({"review_id": review_id, "run_id": record["run_id"], "model": record["model"],
                        "case_id": record["case_id"], "seed": record["seed"], "image": source.relative_to(root).as_posix() if root is not None else str(source)})
        rows.append({"review_id": review_id, "rater_id": "", **{f: "" for f in RATING_FIELDS}, "usable": "", "error_categories": "", "notes": ""})
        prompt = html.escape(record.get("prompt", record["case_id"]))
        criteria = html.escape("; ".join(record.get("mandatory_criteria", [])))
        cards.append(f'<article><h2>{index}. {review_id}</h2><p>{prompt}</p><p>Mandatory criteria: {criteria}</p><img src="images/{review_id}.png" alt="Generated campaign image"><p>Record ratings in ratings.csv; usable is a separate 0/1 decision.</p></article>')
    csv_path = output_dir / "ratings.csv"
    if not csv_path.exists():
        fields = ["review_id", "rater_id", *RATING_FIELDS, "usable", "error_categories", "notes"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "private_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    anchors = "; ".join(f"{key}: {value}" for key, value in ANCHORS.items())
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Blinded campaign review</title><style>body{font-family:system-ui;max-width:960px;margin:auto;padding:24px}article{border-top:1px solid #aaa;padding:20px 0}img{max-width:100%;max-height:650px}</style><h1>Blinded campaign review</h1>'
    page += f"<p>Five-point scales: {html.escape(anchors)}.</p><p>Rate adherence against the brief and mandatory criteria; coherence against structure and artefacts; composition against visual balance; usability against suitability for the campaign. No ratings are prefilled. Usable: 1 means acceptable as a promotional concept, 0 means unacceptable. Keep rater identity consistent. The primary concept decision requires unanimous nonblank usable votes per image (one rater counts). Disagreement remains missing pending documented human adjudication; it is never converted to a failure or half a usable image. Mean reviewer votes are supplementary. This worksheet does not imply clearance for commercial publication.</p>"
    page += "".join(cards) or "<p>No successfully generated images are available for review.</p>"
    (output_dir / "review.html").write_text(page + "</html>", encoding="utf-8")
    return {"images": len(mapping), "ratings_csv": csv_path.relative_to(root).as_posix() if root is not None else str(csv_path),
            "review_html": (output_dir / "review.html").relative_to(root).as_posix() if root is not None else str(output_dir / "review.html")}


def aggregate_ratings(ratings: list[dict], mapping: list[dict]) -> dict:
    """Average image scores within prompt, then prompts within model.

    Blank scores remain missing. This avoids weighting prompts by how many images
    or raters happen to have been reviewed. One row per (image,rater) is allowed.
    Primary usable_concept_rate requires unanimous binary image votes. Disagreeing
    images remain missing; usable_vote_rate is a supplementary reviewer average.
    """
    lookup = {r["review_id"]: r for r in mapping}
    if len(lookup) != len(mapping):
        raise ValueError("Duplicate review IDs in mapping")
    values = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in ratings:
        nonblank = any(row.get(f) is not None and str(row.get(f, "")).strip() for f in (*RATING_FIELDS, "usable"))
        if not nonblank:
            continue
        rid, rater = row.get("review_id"), str(row.get("rater_id", "")).strip()
        if rid not in lookup or not rater:
            raise ValueError("Every rated image needs a mapped review_id and nonblank rater_id")
        key = (rid, rater)
        if key in seen:
            raise ValueError("Duplicate image/rater assessment")
        seen.add(key)
        for field in (*RATING_FIELDS, "usable"):
            raw = row.get(field, "")
            if raw is None or str(raw).strip() == "":
                continue
            number = float(raw)
            allowed = (0, 1) if field == "usable" else (1, 2, 3, 4, 5)
            if not math.isfinite(number) or number not in allowed:
                raise ValueError(f"Invalid {field}: expected {allowed}")
            values[rid][field].append(number)
    clustered = defaultdict(lambda: defaultdict(list))
    coverage_models = defaultdict(lambda: {"n_images_total": 0, "n_images_with_votes": 0, "n_images_decided": 0,
                                          "n_images_disagreement": 0, "n_images_missing_votes": 0})
    coverage_prompts = defaultdict(lambda: {"n_images_total": 0, "n_images_with_votes": 0, "n_images_decided": 0,
                                           "n_images_disagreement": 0, "n_images_missing_votes": 0})
    for rid, info in lookup.items():
        votes = values.get(rid, {}).get("usable", [])
        decided = bool(votes) and len(set(votes)) == 1
        for counts in (coverage_models[info["model"]], coverage_prompts[(info["model"], info["case_id"])]):
            counts["n_images_total"] += 1
            counts["n_images_with_votes"] += int(bool(votes))
            counts["n_images_decided"] += int(decided)
            counts["n_images_disagreement"] += int(bool(votes) and not decided)
            counts["n_images_missing_votes"] += int(not votes)
        if decided:
            clustered[(info["model"], info["case_id"])]["usable_concept_rate"].append(votes[0])
    for rid, fields in values.items():
        info = lookup[rid]
        for field, scores in fields.items():
            output_field = "usable_vote_rate" if field == "usable" else field
            clustered[(info["model"], info["case_id"])][output_field].append(sum(scores) / len(scores))
    by_model = defaultdict(lambda: defaultdict(list))
    per_prompt = []
    for (model, prompt), fields in sorted(clustered.items()):
        metrics = {f: sum(v) / len(v) for f, v in fields.items()}
        per_prompt.append({"model": model, "case_id": prompt, "metrics": metrics, "usable_coverage": coverage_prompts[(model, prompt)]})
        for field, score in metrics.items():
            by_model[model][field].append(score)
    summaries = {model: {field: {"mean": sum(scores) / len(scores), "n_prompts": len(scores)}
                         for field, scores in fields.items()} for model, fields in by_model.items()}
    global_coverage = {name: sum(counts[name] for counts in coverage_models.values()) for name in
                       ("n_images_total", "n_images_with_votes", "n_images_decided", "n_images_disagreement", "n_images_missing_votes")}
    return {"n_assessments": len(seen), "n_images_rated": len(values), "per_prompt": per_prompt, "models": summaries,
            "missing_review_images": len(mapping) - len(values), "aggregation": "image->prompt->model; supplementary reviewer votes average within image",
            "usable_rule": "unanimous nonblank per-image votes; disagreement missing pending documented adjudication",
            "usable_coverage": global_coverage, "usable_coverage_by_model": dict(coverage_models)}
