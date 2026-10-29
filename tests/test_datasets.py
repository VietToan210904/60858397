"""Synthetic unit-test assets live only in pytest temporary directories."""
import csv
import json
from pathlib import Path

import pytest

from stylesphere.datasets import (assistance_scenarios, business_knowledge_base,
                                 campaign_briefs, prepare, validate)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_controlled_inputs_are_explicitly_hypothetical_and_varied(tmp_path):
    prepare(tmp_path, "generation", {})
    prepare(tmp_path, "assistance", {})
    assert validate(tmp_path, "generation") == []
    assert validate(tmp_path, "assistance") == []
    briefs = campaign_briefs()
    assert len(briefs) == 24
    assert len({row["prompt"] for row in briefs}) == 24
    assert {category: sum(row["category"] == category for row in briefs) for category in {row["category"] for row in briefs}} == {
        "product_focused": 6, "editorial": 6, "seasonal": 6, "multi_attribute": 6}
    kb = business_knowledge_base()
    scenarios = assistance_scenarios(kb)
    assert len(scenarios) == len({row["customer_request"] for row in scenarios}) == 80
    assert all(row["hypothetical"] for row in scenarios)
    assert {row["expected_outcome"]["type"] for row in scenarios} == {"answer", "refuse", "escalate"}


def test_generation_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="three distinct"):
        campaign_briefs([1, 1, 2])


def test_prepare_preserves_edited_inputs_and_human_labels(tmp_path):
    prepare(tmp_path, "generation", {})
    prompts = tmp_path / "datasets/generation/prompts.jsonl"
    rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    rows[0]["prompt"] = "Human-edited frozen creative brief"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = prompts.read_bytes()
    prepare(tmp_path, "generation", {})
    assert prompts.read_bytes() == before
    prepare(tmp_path, "retrieval", {})
    qrels = tmp_path / "datasets/retrieval/qrels.csv"
    qrels.write_text("query_id,product_id,relevance,annotator_id,annotation_date\nq1,p1,3,rater,2026-01-01\n")
    before = qrels.read_bytes()
    prepare(tmp_path, "retrieval", {})
    assert qrels.read_bytes() == before


def test_empty_manifests_are_incomplete(tmp_path):
    for pair in ("retrieval", "reconstruction"):
        manifest = prepare(tmp_path, pair, {})
        assert manifest["model_inference_performed"] is False
        assert any("template" in error for error in validate(tmp_path, pair))


def test_retrieval_import_stable_ids_human_qrels_and_completeness(tmp_path):
    # These text bytes deliberately stand in for image files only for schema tests.
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"SYNTHETIC UNIT TEST IMAGE PLACEHOLDER")
    source = tmp_path / "source.csv"
    write_csv(source, [{"item_ID": "a", "image_path": "a.jpg", "category1": "shoe", "text": "white shoe"},
                       {"item_ID": "b", "image_path": "b.jpg", "category1": "bag", "text": "red bag"}])
    queries = tmp_path / "queries.csv"
    write_csv(queries, [{"query_id": "q1", "query_text": "white shoe", "query_category": "colour", "qrels_complete": "true"}])
    cfg = {"metadata_path": "source.csv", "image_root": ".", "source": "synthetic-unit-test-only",
           "source_revision": "test-revision", "licence": "test-only", "queries_path": "queries.csv"}
    result = prepare(tmp_path, "retrieval", cfg)
    assert result["details"]["catalogue_count"] == 2
    catalogue = list(csv.DictReader((tmp_path / "datasets/retrieval/catalogue.csv").open()))
    annotations = tmp_path / "annotations.csv"
    write_csv(annotations, [{"query_id": "q1", "product_id": row["product_id"], "relevance": "3" if row["source_id"] == "a" else "0",
                            "annotator_id": "synthetic-test-rater", "annotation_date": "2026-01-01"} for row in catalogue])
    cfg["annotations_path"] = "annotations.csv"
    prepare(tmp_path, "retrieval", cfg)
    assert validate(tmp_path, "retrieval") == []
    ids = [row["product_id"] for row in catalogue]
    retained = prepare(tmp_path, "retrieval", cfg, seed=123)
    assert retained["details"]["selection_seed"] == 8410
    assert [row["product_id"] for row in csv.DictReader((tmp_path / "datasets/retrieval/catalogue.csv").open())] == ids
    (tmp_path / "a.jpg").write_bytes(b"changed")
    assert any("hash differs" in error for error in validate(tmp_path, "retrieval"))
    source.write_text(source.read_text() + "new,not-an-image,shoe,changed\n")
    with pytest.raises(ValueError, match="Frozen source"):
        prepare(tmp_path, "retrieval", cfg)


def test_invalid_annotation_and_revision_are_rejected(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"UNIT TEST ONLY")
    write_csv(tmp_path / "source.csv", [{"product_id": "a", "image_path": "a.jpg", "category": "shoe"}])
    with pytest.raises(ValueError, match="source_revision"):
        prepare(tmp_path, "retrieval", {"metadata_path": "source.csv", "source": "test", "licence": "test"})


def test_kb_fact_changes_invalidate_expected_answers(tmp_path):
    prepare(tmp_path, "assistance", {})
    kb_path = tmp_path / "datasets/assistance/knowledge_base.json"
    kb = json.loads(kb_path.read_text())
    kb["products"][0]["price"] = 999
    kb_path.write_text(json.dumps(kb))
    assert any("Reference fact differs" in error for error in validate(tmp_path, "assistance"))


def test_geometry_requires_documented_alignment(tmp_path):
    (tmp_path / "a.png").write_bytes(b"UNIT TEST ONLY")
    (tmp_path / "a.glb").write_bytes(b"UNIT TEST ONLY")
    write_csv(tmp_path / "objects-source.csv", [{"object_id": "test-a", "source_image": "a.png", "category": "bag",
        "source_dataset": "synthetic-unit-test", "licence": "test", "source_revision": "test", "reference_mesh": "a.glb", "geometry_eligible": "true"}])
    with pytest.raises(ValueError, match="preparation_metadata"):
        prepare(tmp_path, "reconstruction", {"metadata_path": "objects-source.csv"})
    write_csv(tmp_path / "objects-source.csv", [{"object_id": "test-a", "source_image": "a.png", "category": "bag",
        "source_dataset": "synthetic-unit-test", "licence": "test", "source_revision": "test", "reference_mesh": "a.glb", "geometry_eligible": "false"}])
    prepare(tmp_path, "reconstruction", {"metadata_path": "objects-source.csv"})
    assert validate(tmp_path, "reconstruction") == []
    (tmp_path / "a.glb").write_bytes(b"CHANGED SYNTHETIC UNIT TEST MESH")
    assert any("reference_mesh hash differs" in issue for issue in validate(tmp_path, "reconstruction"))
