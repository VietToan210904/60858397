"""Separate post-hoc AI visual annotations; never human ratings or model outputs."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml

from stylesphere.common.io import canonical_hash, portable_path, read_jsonl, sha256_file, utc_now, write_json
from stylesphere.common.statistics import paired_bootstrap

GEN_SCORES = ("ai_prompt_adherence", "ai_visual_coherence", "ai_composition", "ai_usability_rating")
RECON_SCORES = ("ai_visible_input_fidelity", "ai_shape_coherence", "ai_texture_coherence", "ai_multiview_consistency")
RECON_CRITERIA = ("visible_object_type_and_major_parts", "visible_colour_and_texture_similarity",
                  "cross_view_structural_coherence", "unseen_surface_correctness")
ERRORS = {
    "generation": {"none_observed", "prompt_mismatch", "missing_mandatory_attributes", "visual_artefacts",
                   "incorrect_object_structure", "inappropriate_or_misleading_content", "other"},
    "reconstruction": {"none_observed", "missing_geometry", "distorted_shape", "incorrect_texture", "other"},
}


def frozen_protocol(root: Path, config_path="config/ai_visual_assessment.yaml") -> dict:
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    if config.get("study_kind") != "post_hoc_supplementary_ai_visual_assessment":
        raise ValueError("AI assessment protocol must explicitly be a post-hoc supplement")
    docs = root / "docs/AI_VISUAL_ASSESSMENT_PROTOCOL.md"
    core = {"config": config, "config_sha256": sha256_file(root / config_path),
            "instructions_sha256": sha256_file(docs), "exporter_source_sha256": sha256_file(Path(__file__))}
    return {**core, "protocol_sha256": canonical_hash(core)}


def select_verified_records(root: Path, results: Path) -> list[dict]:
    """Bind exact completed runs to previously audited frozen input evidence."""
    from stylesphere.generation.canonical_review import validate_canonical_review
    from stylesphere.generation.provenance import verify_prompt_binding
    validate_canonical_review(root, results)
    gen_manifest = json.loads((results / "generation/human_review/manifest.json").read_text())
    generation = read_jsonl(results / "generation/records.jsonl")
    selected = []
    for binding in gen_manifest["source_bindings"]:
        matches = [r for r in generation if all(r.get(k) == binding[k] for k in ("model", "case_id", "seed", "run_id", "input_sha256", "config_sha256"))]
        if len(matches) != 1 or matches[0].get("status") != "success":
            raise ValueError("Generation selection must identify one actual successful record")
        row = matches[0]
        if verify_prompt_binding(row, root, results / "generation"):
            raise ValueError("Generation frozen prompt binding failed")
        selected.append(row)
    audit_path = root / "environment/reconstruction-benchmark-validation.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("integrity_passed") or audit.get("selected_record_count") != 40:
        raise ValueError("A passing 40-object reconstruction audit is required")
    reconstruction = [r for r in read_jsonl(results / "reconstruction/records.jsonl")
                      if r.get("model") in audit["models"] and r.get("run_id") == audit["models"][r["model"]]["run_id"]]
    if canonical_hash(reconstruction) != audit["selected_records_digest"]:
        raise ValueError("Audited reconstruction record subset changed")
    selected.extend(reconstruction)
    if Counter(r["pair"] for r in selected) != {"generation": 144, "reconstruction": 40}:
        raise ValueError("Expected exactly 144 generation and 40 reconstruction outputs")
    for record in selected:
        if record.get("provenance") != "model_inference" or record.get("mode") != "evaluation" or record.get("synthetic"):
            raise ValueError("Only genuine final evaluation outputs are eligible")
        if set(record["artifacts"]) != set(record["artifact_sha256"]):
            raise ValueError("Missing source artifact hashes")
        for name, value in record["artifacts"].items():
            if sha256_file(portable_path(root, value)) != record["artifact_sha256"][name]:
                raise ValueError("Source artifact changed before AI packet export")
    return selected


def _display(source: Path, destination: Path, size: int) -> None:
    with Image.open(source) as original:
        if original.mode == "RGBA":
            canvas = Image.new("RGBA", original.size, "white")
            canvas.alpha_composite(original)
            original = canvas.convert("RGB")
        else:
            original = original.convert("RGB")
        image = Image.new("RGB", (size, size), "white")
        original.thumbnail((size, size), Image.Resampling.LANCZOS)
        image.paste(original, ((size - original.width) // 2, (size - original.height) // 2))
        image.save(destination)


def export_packets(root: Path, results: Path, destination: Path, *, records: list[dict] | None = None) -> dict:
    """Freeze label-masked review packets, without creating any assessment values."""
    root, results, destination = Path(root).resolve(), Path(results).resolve(), Path(destination).resolve()
    protocol = frozen_protocol(root)
    records = select_verified_records(root, results) if records is None else records
    selection = [{"record_sha256": canonical_hash(r), "artifacts": r["artifact_sha256"]} for r in records]
    identity = canonical_hash({"protocol_sha256": protocol["protocol_sha256"], "selection": selection})
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = validate_export(destination)
        if manifest["selection_sha256"] != identity:
            raise ValueError("Existing AI packet selection/protocol differs; refuse remapping")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refuse to overwrite a nonempty AI assessment directory")
    (destination / "images").mkdir(parents=True)
    (destination / "batches").mkdir()
    (destination / "assessments").mkdir()
    (destination / "templates").mkdir()
    write_json(destination / "protocol.json", protocol)
    (destination / "instructions.md").write_bytes((root / "docs/AI_VISUAL_ASSESSMENT_PROTOCOL.md").read_bytes())
    (destination / "config_frozen.yaml").write_bytes((root / "config/ai_visual_assessment.yaml").read_bytes())
    shuffled = list(records)
    random.Random(protocol["config"]["seed"]).shuffle(shuffled)
    public, private = [], []
    for record in shuffled:
        opaque = "ai_" + canonical_hash({"identity": identity, "record": canonical_hash(record)})[:16]
        pair = record["pair"]
        card = {"assessment_id": opaque, "pair": pair, "brief": record.get("prompt", "Compare the source photograph with four model-generated object previews."),
                "criteria": record.get("mandatory_criteria", list(RECON_CRITERIA)), "display_assets": {}}
        sources = {"image": record["artifacts"]["image"]} if pair == "generation" else {
            "input": record["artifacts"]["input_descriptor"], **{f"view_{i:02d}": record["artifacts"][f"render_{i:02d}"] for i in range(4)}}
        if pair == "reconstruction":
            descriptor = json.loads(portable_path(root, sources["input"]).read_text())
            if canonical_hash(descriptor) != record["input_sha256"]:
                raise ValueError("Reconstruction descriptor input binding failed")
            sources["input"] = descriptor["manifest"]["source_image"]
        for asset, value in sources.items():
            target = destination / "images" / f"{opaque}_{asset}.png"
            _display(portable_path(root, value), target, 512)
            card["display_assets"][asset] = {"path": target.relative_to(destination).as_posix(), "sha256": sha256_file(target)}
        card["evidence_sha256"] = {key: item["sha256"] for key, item in card["display_assets"].items()}
        if pair == "reconstruction":
            mosaic = Image.new("RGB", (1536, 1088), "white")
            draw = ImageDraw.Draw(mosaic)
            for i, (asset, item) in enumerate(card["display_assets"].items()):
                x, y = (i % 3) * 512, (i // 3) * 544
                draw.text((x + 8, y + 8), f"{opaque} | {asset}", fill="black")
                with Image.open(destination / item["path"]) as im: mosaic.paste(im, (x, y + 32))
            card_path = destination / "images" / f"{opaque}_card.png"
            mosaic.save(card_path)
            card["card_image"] = {"path": card_path.relative_to(destination).as_posix(), "sha256": sha256_file(card_path)}
        public.append(card)
        private.append({"assessment_id": opaque, "pair": pair, "model": record["model"], "case_id": record["case_id"],
                        "seed": record["seed"], "run_id": record["run_id"], "input_sha256": record["input_sha256"],
                        "source_record_sha256": canonical_hash(record), "source_artifact_sha256": record["artifact_sha256"]})
    batch_files = []
    for pair in ("generation", "reconstruction"):
        cards = [c for c in public if c["pair"] == pair]
        for start in range(0, len(cards), 12):
            number = start // 12 + 1
            batch = {"kind": "post_hoc_supplementary_ai_visual_packet", "protocol_sha256": protocol["protocol_sha256"],
                     "selection_sha256": identity, "pair": pair, "batch_number": number, "cards": cards[start:start + 12]}
            filename = f"batches/{pair}_{number:02d}.json"
            if pair == "generation":
                contact = Image.new("RGB", (2048, 1632), "white")
                draw = ImageDraw.Draw(contact)
                for i, card in enumerate(batch["cards"]):
                    x, y = (i % 4) * 512, (i // 4) * 544
                    draw.text((x + 8, y + 8), f"{i+1}. {card['assessment_id']}", fill="black")
                    with Image.open(destination / card["display_assets"]["image"]["path"]) as im: contact.paste(im, (x, y + 32))
                contact_path = destination / "batches" / f"{pair}_{number:02d}.png"
                contact.save(contact_path)
                batch["contact_sheet"] = {"path": contact_path.relative_to(destination).as_posix(), "sha256": sha256_file(contact_path),
                                          "layout": "4 columns, 3 rows; each image 512x512 beneath its ID"}
            write_json(destination / filename, batch)
            template_path = destination / "templates" / f"{pair}_{number:02d}.template.jsonl"
            with template_path.open("w", encoding="utf-8") as stream:
                for card in batch["cards"]:
                    stream.write(json.dumps(assessment_template(card, protocol["protocol_sha256"])) + "\n")
            batch_files.append(filename)
    write_json(destination / "private_mapping.json", private)
    write_json(destination / "public_cards.json", public)
    files = {path.relative_to(destination).as_posix(): sha256_file(path) for path in destination.rglob("*") if path.is_file()}
    manifest = {"schema_version": 1, "kind": "post_hoc_supplementary_ai_visual_export", "created_at_utc": utc_now(),
                "protocol_sha256": protocol["protocol_sha256"], "selection_sha256": identity, "n_outputs": len(public),
                "counts": dict(Counter(c["pair"] for c in public)), "batch_files": batch_files, "files_sha256": files,
                "source_selection": selection, "human_rating_files_modified": False,
                "masking_limit": protocol["config"]["masking"]}
    write_json(manifest_path, manifest)
    return manifest


def validate_export(destination: Path) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    protocol = json.loads((destination / "protocol.json").read_text())
    core = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if canonical_hash(core) != protocol["protocol_sha256"] or protocol["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("AI assessment protocol binding changed")
    if (sha256_file(destination / "instructions.md") != protocol["instructions_sha256"]
            or sha256_file(destination / "config_frozen.yaml") != protocol["config_sha256"]):
        raise ValueError("Frozen AI instructions/configuration changed")
    if canonical_hash({"protocol_sha256": manifest["protocol_sha256"], "selection": manifest["source_selection"]}) != manifest["selection_sha256"]:
        raise ValueError("AI selection identity changed")
    for name, expected in manifest["files_sha256"].items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"AI packet artifact changed: {name}")
    return manifest


def assessment_template(card: dict, protocol_sha256: str) -> dict:
    """Pending collection template, explicitly invalid as a completed assessment."""
    fields = GEN_SCORES if card["pair"] == "generation" else RECON_SCORES
    return {"schema_version": 1, "assessment_id": card["assessment_id"], "protocol_sha256": protocol_sha256,
            "assessor_kind": "ai_assistant", "assessor_id": "", "assessor_model": "", "assessed_at_utc": "",
            "review_status": "pending", "scores": {name: None for name in fields},
            "criteria": [{"criterion": name, "verdict": "uncertain", "evidence": ""} for name in card["criteria"]],
            "ai_usable_concept": None, "confidence": None, "uncertainty_reasons": [], "observations": [],
            "error_categories": [], "evidence_sha256": card["evidence_sha256"]}


def validate_assessment(row: dict, card: dict, protocol_sha256: str) -> None:
    if row.get("schema_version") != 1:
        raise ValueError("AI assessment schema_version must be 1")
    if row.get("assessment_id") != card["assessment_id"] or row.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Assessment identity/protocol mismatch")
    if row.get("assessor_kind") != "ai_assistant" or not row.get("assessor_id") or not row.get("assessor_model"):
        raise ValueError("AI assessor identity must be explicit; unavailable model ID must say not_exposed")
    try:
        timestamp = datetime.fromisoformat(row["assessed_at_utc"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None: raise ValueError("Timezone required")
    except (KeyError, TypeError, ValueError):
        raise ValueError("Actual assessment timestamp with timezone required") from None
    if row.get("evidence_sha256") != card["evidence_sha256"]:
        raise ValueError("Assessment image evidence hashes differ from frozen packet")
    if row.get("review_status") not in {"assessed", "abstained"} or row.get("confidence") not in {"low", "medium", "high"}:
        raise ValueError("Explicit assessment status and confidence required")
    if not isinstance(row.get("uncertainty_reasons"), list):
        raise ValueError("Uncertainty reasons must be a list")
    fields = GEN_SCORES if card["pair"] == "generation" else RECON_SCORES
    if set(row.get("scores", {})) != set(fields):
        raise ValueError("Wrong AI score fields; human-rating column names are not accepted")
    for value in row["scores"].values():
        if value is not None and (isinstance(value, bool) or value not in (1, 2, 3, 4, 5)):
            raise ValueError("AI ordinal scores must be 1..5 or null")
    if isinstance(row.get("ai_usable_concept"), bool) or row.get("ai_usable_concept") not in (None, 0, 1):
        raise ValueError("AI usable concept must be 0/1/null")
    if card["pair"] == "reconstruction" and row.get("ai_usable_concept") is not None:
        raise ValueError("Reconstruction cannot receive a promotional usable-concept rating")
    criteria = row.get("criteria", [])
    if len(criteria) != len(card["criteria"]) or [c.get("criterion") for c in criteria] != card["criteria"]:
        raise ValueError("Each frozen mandatory criterion needs exactly one ordered assessment")
    for criterion in criteria:
        if criterion.get("verdict") not in {"met", "absent", "uncertain"} or not str(criterion.get("evidence", "")).strip():
            raise ValueError("Every criterion requires a verdict and visible evidence/uncertainty reason")
    if card["pair"] == "reconstruction" and criteria[-1]["verdict"] != "uncertain":
        raise ValueError("Unseen surface correctness has no reference and must stay uncertain")
    if row.get("ai_usable_concept") == 1 and any(c["verdict"] != "met" for c in criteria):
        raise ValueError("Positive AI shortlist requires every mandatory criterion observed met")
    if row.get("ai_usable_concept") == 1 and any(row["scores"].get(name) is not None and row["scores"][name] <= 2
                                                  for name in ("ai_visual_coherence", "ai_usability_rating")):
        raise ValueError("Positive AI shortlist contradicts a major-defect coherence/usability score")
    observations = row.get("observations", [])
    if row["review_status"] == "assessed" and not observations:
        raise ValueError("An assessed image requires actual visual observations")
    if row["review_status"] == "abstained" and (any(v is not None for v in row["scores"].values()) or row.get("ai_usable_concept") is not None):
        raise ValueError("Abstention preserves missing scores, never invented zero ratings")
    for evidence in observations:
        if evidence.get("asset") not in card["display_assets"] or not str(evidence.get("description", "")).strip():
            raise ValueError("Observation needs a provided asset and visible description")
        region = evidence.get("region")
        if region is not None and (len(region) != 4 or not all(isinstance(v, (int,float)) and math.isfinite(v) and 0 <= v <= 1 for v in region)
                                   or region[0] >= region[2] or region[1] >= region[3]):
            raise ValueError("Optional observation region must be a normalized positive rectangle")
    errors = row.get("error_categories", [])
    if not isinstance(errors, list) or not set(errors) <= ERRORS[card["pair"]] or "none_observed" in errors and len(errors) != 1:
        raise ValueError("Invalid observed error categories")


def aggregate(destination: Path, assessment_files: list[Path] | None = None) -> dict:
    """Summarize AI annotations with prompt/object clustering and explicit missingness."""
    destination = Path(destination)
    manifest = validate_export(destination)
    cards = {x["assessment_id"]: x for x in json.loads((destination / "public_cards.json").read_text())}
    mapping = {x["assessment_id"]: x for x in json.loads((destination / "private_mapping.json").read_text())}
    files = sorted(assessment_files if assessment_files is not None else (destination / "assessments").glob("*.jsonl"))
    assessed = {}
    for file in files:
        for row in read_jsonl(file):
            identity = row.get("assessment_id")
            if identity not in cards: raise ValueError("Unknown AI assessment output ID")
            validate_assessment(row, cards[identity], manifest["protocol_sha256"])
            if identity in assessed: raise ValueError("One primary AI assessment per output; duplicate IDs are not consensus")
            assessed[identity] = row
    clustered = defaultdict(lambda: defaultdict(list))
    coverage = defaultdict(Counter)
    paired = defaultdict(dict)
    for identity, info in mapping.items():
        key = (info["pair"], info["model"])
        coverage[key]["planned_outputs"] += 1
        row = assessed.get(identity)
        if not row:
            coverage[key]["missing_outputs"] += 1
            continue
        coverage[key][row["review_status"] + "_outputs"] += 1
        coverage[key]["confidence_" + row["confidence"]] += 1
        values = {**row["scores"], "ai_usable_concept_rate": row.get("ai_usable_concept")}
        for metric, value in values.items():
            if value is not None:
                clustered[(info["pair"], info["model"], info["case_id"])][metric].append(float(value))
                paired[(info["pair"], info["case_id"], info["seed"], info["input_sha256"], metric)][info["model"]] = float(value)
        for category in row["error_categories"]:
            coverage[key]["observed_error_" + category] += 1
        coverage[key]["criteria_" + "total"] += len(row["criteria"])
        for criterion in row["criteria"]: coverage[key]["criteria_" + criterion["verdict"]] += 1
    by_model = defaultdict(lambda: defaultdict(list))
    clusters = []
    for (pair, model, case), metrics in sorted(clustered.items()):
        means = {name: sum(values)/len(values) for name,values in metrics.items()}
        clusters.append({"pair":pair,"model":model,"case_id":case,"metrics":means,
                         "n_scored_outputs":{name:len(values) for name,values in metrics.items()}})
        for name,value in means.items(): by_model[(pair,model)][name].append(value)
    models = [{"pair":pair,"model":model,"coverage":dict(coverage[(pair,model)]),
               "metrics":{name:{"mean":sum(values)/len(values),"n_clusters":len(values)} for name,values in by_model[(pair,model)].items()}}
              for pair,model in sorted(coverage)]
    comparisons = []
    pair_models = {pair:sorted({v["model"] for v in mapping.values() if v["pair"]==pair}) for pair in ("generation","reconstruction")}
    for pair, names in pair_models.items():
        if len(names)!=2: continue
        metrics = sorted({key[-1] for key in paired if key[0]==pair})
        for metric in metrics:
            matches=[(key,value) for key,value in paired.items() if key[0]==pair and key[-1]==metric and all(name in value for name in names)]
            if not matches: continue
            result=paired_bootstrap([v[names[0]] for _,v in matches],[v[names[1]] for _,v in matches],
                                    clusters=[k[1] for k,_ in matches],seed=8410,iterations=2000)
            comparisons.append({"pair":pair,"metric":metric,"model_a":names[0],"model_b":names[1],"matched_outputs":len(matches),**result})
    return {"kind":"post_hoc_supplementary_ai_visual_summary","protocol_sha256":manifest["protocol_sha256"],
            "selection_sha256":manifest["selection_sha256"],"assessment_files_sha256":{p.relative_to(destination).as_posix():sha256_file(p) for p in files},
            "n_outputs":len(cards),"n_assessments":len(assessed),"n_missing":len(cards)-len(assessed),
            "models":models,"per_prompt_or_object":clusters,"paired_comparisons":comparisons,
            "interpretation":"AI observations and ordinal summaries only; not human feedback, primary usable-concept results, calibrated quality, reference-geometry accuracy or commercial approval. Confidence intervals cover sampled prompt/object variability, not AI assessor bias."}
