"""Ranking-independent photo/source evidence for supplementary AI proxy qrels.

This exporter never reads model scores or rankings, modifies human qrels, or
automatically assigns subjective visual ratings. Product judgements are authored
separately after the rubric is fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path

from stylesphere.common.io import canonical_hash, sha256_file, utc_now, write_json

TYPE_KEYS = ["running_shoes", "ankle_boots", "sandals", "flat_shoes", "handbags", "backpacks",
             "wallets", "hats", "earrings", "necklaces", "rings", "casual_shoes"]
TRAIT_KEYS = ["black", "white", "red", "simple_everyday", "metallic_finish", "formal",
              "minimalist", "travel_appearance", "bold_decorative", "neutral"]
SOURCE_FIELDS = ["item_name", "color", "material", "style", "pattern", "bullet_point", "item_keywords", "product_type"]
PROTOCOL = {
    "schema_version": 1, "protocol_id": "supplementary-ai-retrieval-proxy-v1",
    "frozen_before_ratings": True, "human_ground_truth": False,
    "assessor": "Nonblind Codex AI coding assistants, implementation contributors; identities available but predicted rankings withheld until product annotations are fixed. No human validation or inter-rater reliability claim.",
    "purpose": "Complete 600-product by 120-query AI proxy relevance matrix, separate from unchanged human qrels and original benchmark.",
    "evidence": "One actual catalogue photograph and exact cached multilingual ABO listing values per product. No model predictions/scores/rankings, cloud API or GPU inference used for labels.",
    "type_keys": TYPE_KEYS, "trait_keys": TRAIT_KEYS,
    "type_definitions": {
        "running_shoes": "Visibly athletic trainer/running-shoe appearance or explicit source identification; no performance guarantee.",
        "ankle_boots": "Boot with shaft at/just above ankle, below mid-calf; uncertain if crop prevents shaft judgement.",
        "sandals": "Open construction with sandal straps/open toe or heel; not a closed dress shoe.",
        "flat_shoes": "Dress flats, low/no-heel loafers or ballet-style shoes; athletic trainers are not classified as dress flats.",
        "casual_shoes": "Everyday leisure trainers, slip-ons or informal closed lace-ups; may overlap running-shoe appearance.",
        "handbags": "Handheld, shoulder, crossbody or tote bag; excludes backpack and wallet.",
        "backpacks": "Back-carried bag with shoulder straps.", "wallets": "Wallet/cardholder/purse for money or cards, rather than a handbag.",
        "hats": "Headwear including cap, beanie and brimmed hat.", "earrings": "Ear jewellery, including studs, hoops and drops.",
        "necklaces": "Necklace/pendant chain; exclude visually identified bracelet or anklet even in a mixed source category.",
        "rings": "Finger ring, including plain bands and decorative rings."},
    "trait_scale": {"0": "Absent or conflicting with requested visual attribute", "1": "Weak/partial match, such as an accent colour or limited style resemblance", "2": "Clear dominant/strong visual or explicit-source match", "null": "Cannot assess; retain uncertainty rather than inventing a negative"},
    "trait_definitions": {
        "black": "Visible black principal areas; accent only is weak.", "white": "Visible white/ivory principal areas; accent only is weak.",
        "red": "Visible red/burgundy principal areas; orange/pink are not automatically red.",
        "neutral": "Subdued black, white, grey, beige, tan, brown, navy or conventional gold/silver tones dominate.",
        "simple_everyday": "Uncomplicated practical everyday styling for the product class; not a durability/function measurement.",
        "metallic_finish": "Visible reflective metallic finish; metal material alone does not prove visible sheen.",
        "formal": "Dressy/elegant occasion appearance appropriate to its product class; subjective appearance proxy.",
        "minimalist": "Restrained clean design with limited ornamentation; plain does not imply measured quality.",
        "travel_appearance": "Compact/simple or visibly portable/practical appearance for its class. Does not establish physical weight, comfort, airline acceptance or travel performance.",
        "bold_decorative": "Prominent pattern, sculptural form, conspicuous embellishment or decorative motif; mere brightness alone can be weak."},
    "qrels_rules": {"definite_wrong_type": 0, "uncertain_type": None,
                    "matched_type_trait_0": 0, "matched_type_trait_1": 1, "matched_type_trait_2": 3,
                    "matched_type_unknown_trait": None,
                    "travel_exception": "For lightweight-for-travel queries, trait2 gives2 unless an exact source citation explicitly supports lightweight, when3 is permitted. Unknown physical weight is always disclosed; ordinary traits have no grade2 by design."},
    "metrics": {"useful_relevance_threshold": 2, "graded_gain": "2**grade - 1", "k": 10,
                "metrics": "AI-proxy nDCG@10, MRR@10, Hit@10, precision@10, recall@10 using only known AI-proxy relevant denominator; coverage@10 and all-catalogue-complete flag. Never call this objective/human Recall.",
                "unknowns": "Report sensitivity using all unknown grades0 and all unknown grades3; these are two scenario estimates, not necessarily tight nDCG bounds."},
    "invariants": ["Do not change catalogue, queries, original human qrels, original model outputs or frozen runtime.",
                   "Annotate products independently of predicted rankings; freeze all profiles before score access.",
                   "Every product has an authored visual note, confidence, exact image hash, and explicit uncertainty.",
                   "Exact multilingual source citations retain language tags; an unknown source property stays unknown."]}


def freeze_protocol(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "assessment_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != PROTOCOL:
        raise ValueError("Frozen proxy rubric differs; use a new version")
    write_json(path, PROTOCOL)
    receipt = {"schema_version": 1, "protocol_sha256": canonical_hash(PROTOCOL),
               "frozen_at_utc": utc_now(), "accepted_by_project_coordinator_before_ratings": True}
    receipt_path = directory / "protocol_freeze.json"
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    return receipt


def source_facts(root: Path, catalogue: list[dict]) -> tuple[list[dict], dict]:
    """Retain exact multilingual values; do not machine-translate or infer gaps."""
    wanted = {r["source_id"] for r in catalogue}
    matches = defaultdict(list)
    path = root / ".cache/abo/listings_0.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["item_id"] in wanted:
                matches[row["item_id"]].append(row)
    packets = []
    for index, row in enumerate(sorted(catalogue, key=lambda r: r["product_id"]), 1):
        records = matches.get(row["source_id"], [])
        packets.append({"annotation_index": index, "batch": (index - 1) // 25 + 1,
                        "product_id": row["product_id"], "source_id": row["source_id"],
                        "image_path": row["image_path"], "image_sha256": row["image_sha256"],
                        "source_category": row["category"], "original_catalogue_attributes": row["attributes"],
                        "source_facts": {field: [value for record in records for value in record.get(field, [])]
                                         for field in SOURCE_FIELDS},
                        "full_source_listings": records})
    coverage = {"n_products": len(packets), "n_matched_products": len(matches),
                "n_source_records": sum(map(len, matches.values())),
                "source_sha256": sha256_file(path),
                "field_coverage_any_language": {field: sum(bool(p["source_facts"][field]) for p in packets) for field in SOURCE_FIELDS},
                "field_coverage_english": {field: sum(any(v.get("language_tag", "").startswith("en_") and str(v.get("value", "")).strip()
                                                                      for v in p["source_facts"][field]) for p in packets) for field in SOURCE_FIELDS},
                "item_name_language_counts": dict(Counter(v.get("language_tag", "unspecified") for p in packets for v in p["source_facts"]["item_name"]))}
    return packets, coverage


def export(root: Path, directory: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    root, directory = Path(root), Path(directory)
    with (root / "datasets/retrieval/catalogue.csv").open(encoding="utf-8-sig", newline="") as stream:
        catalogue = list(csv.DictReader(stream))
    packets, coverage = source_facts(root, catalogue)
    directory.mkdir(parents=True, exist_ok=True)
    freeze_protocol(directory)
    (directory / "sheets").mkdir(exist_ok=True)
    (directory / "source_sidecars").mkdir(exist_ok=True)
    evidence_path = directory / "product_evidence.jsonl"
    evidence = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets)
    if evidence_path.exists() and evidence_path.read_text(encoding="utf-8") != evidence:
        raise ValueError("Evidence changed; preserve the prior version and use a new directory")
    evidence_path.write_text(evidence, encoding="utf-8")
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    blank = []
    for offset in range(0, len(packets), 25):
        batch = packets[offset:offset + 25]
        sheet = Image.new("RGB", (1500, 1750), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, packet in enumerate(batch):
            image_path = root / packet["image_path"]
            if sha256_file(image_path) != packet["image_sha256"]:
                raise ValueError(f"Catalogue image hash differs: {packet['product_id']}")
            with Image.open(image_path) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (294, 296))
                x, y = slot % 5 * 300, slot // 5 * 350
                sheet.paste(thumbnail, (x + (300 - thumbnail.width) // 2, y + (300 - thumbnail.height) // 2))
            draw.text((x + 5, y + 302), f"{packet['annotation_index']:03d} {packet['source_id']}", font=font, fill="black")
            draw.text((x + 5, y + 324), packet["product_id"] + " " + packet["source_category"], font=font, fill="black")
            blank.append({"annotation_index": packet["annotation_index"], "product_id": packet["product_id"],
                          "image_sha256": packet["image_sha256"], "matched_types": None, "uncertain_types": None,
                          "traits": {key: None for key in TRAIT_KEYS}, "explicit_lightweight_source": None,
                          "confidence": None, "visual_notes": None, "source_support": []})
        number = offset // 25 + 1
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=94, subsampling=0)
        sheet_path = directory / "sheets" / f"batch-{number:02d}.jpg"
        if sheet_path.exists():
            if sheet_path.read_bytes() != buffer.getvalue():
                raise ValueError("Existing evidence sheet differs; use a new version")
        else:
            sheet_path.write_bytes(buffer.getvalue())
        (directory / "source_sidecars" / f"batch-{number:02d}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in batch), encoding="utf-8")
    template = directory / "annotation_template.jsonl"
    if not template.exists():
        template.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in blank), encoding="utf-8")
    write_json(directory / "source_coverage.json", coverage)
    manifest = {"schema_version": 1, "created_at_utc": utc_now(), "purpose": "Supplementary AI visual/source assessment, not human qrels",
                "rankings_read": False, "order": "lexicographic product_id, 25 images per sheet", "n_products": len(packets),
                "source_sha256": {"catalogue": sha256_file(root / "datasets/retrieval/catalogue.csv"),
                                  "queries": sha256_file(root / "datasets/retrieval/queries.csv"),
                                  "original_human_qrels": sha256_file(root / "datasets/retrieval/qrels.csv"),
                                  "multilingual_source": coverage["source_sha256"]},
                "product_evidence_sha256": sha256_file(evidence_path), "product_evidence_canonical_sha256": canonical_hash(packets),
                "sheet_sha256": {p.name: sha256_file(p) for p in sorted((directory / "sheets").glob("*.jpg"))},
                "implementation_sha256": sha256_file(Path(__file__))}
    write_json(directory / "evidence_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "freeze-protocol"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--directory", type=Path, default=Path("results/retrieval/ai_assessment/v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.directory if args.directory.is_absolute() else root / args.directory
    result = export(root, directory) if args.command == "export" else freeze_protocol(directory)
    print(json.dumps({"directory": str(directory), "n_products": result.get("n_products"), "protocol_sha256": result.get("protocol_sha256")}))


if __name__ == "__main__":
    main()
