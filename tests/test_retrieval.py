"""Synthetic, hand-calculated fixtures only; never benchmark observations."""

import csv
import math
import sys

import pytest

from stylesphere.retrieval.data import load_dataset
from stylesphere.retrieval.metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k, recall_at_k, score_query


def test_hand_calculated_ranking_metrics():
    labels = {"a": 3, "b": 2, "c": 0}
    ranking = ["c", "b", "a"]
    expected_dcg = 3 / math.log2(3) + 7 / math.log2(4)
    ideal = 7 + 3 / math.log2(3)
    assert ndcg_at_k(ranking, labels, 3) == pytest.approx(expected_dcg / ideal)
    assert mean_reciprocal_rank(ranking, labels) == 0.5
    assert hit_at_k(ranking, labels, 1) == 0
    assert hit_at_k(ranking, labels, 2) == 1
    assert recall_at_k(ranking, labels, 2, complete=True) == 0.5
    assert recall_at_k(ranking, labels, 2) is None


def test_queries_without_judged_relevant_items_are_ineligible():
    result = score_query(["a"], {"a": 0}, complete=True)
    assert result["ndcg_at_10"] is None
    assert result["mrr"] is None
    assert result["hit_at_5"] is None
    assert result["recall_at_10"] is None
    assert result["ranking_metrics_eligible"] is False


def test_unjudged_documents_have_zero_gain_and_are_counted():
    result = score_query(["unknown", "a"], {"a": 2})
    assert result["ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    assert result["mrr"] == 0.5
    assert result["unjudged_retrieved_count"] == 1


def test_invalid_ranking_and_grades_fail():
    with pytest.raises(ValueError, match="unique"):
        ndcg_at_k(["a", "a"], {"a": 1})
    with pytest.raises(ValueError, match="nonnegative"):
        ndcg_at_k(["a"], {"a": -1})
    with pytest.raises(ValueError, match="positive"):
        hit_at_k(["a"], {"a": 1}, 0)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def retrieval_dataset(tmp_path):
    # Bytes stand in for files solely for manifest validation; never encoded.
    (tmp_path / "test_image.png").write_bytes(b"synthetic fixture, not a real image")
    write_csv(tmp_path / "catalogue.csv", ["product_id", "image_path", "category", "source"],
              [{"product_id": "a", "image_path": "test_image.png", "category": "shoe", "source": "synthetic unit fixture"},
               {"product_id": "b", "image_path": "test_image.png", "category": "bag", "source": "synthetic unit fixture"}])
    write_csv(tmp_path / "queries.csv", ["query_id", "query_text", "qrels_complete"], [{"query_id": "q", "query_text": "red shoe", "qrels_complete": "false"}])
    write_csv(tmp_path / "qrels.csv", ["query_id", "product_id", "relevance"], [{"query_id": "q", "product_id": "a", "relevance": 2}])
    return tmp_path


def test_dataset_references_and_completeness(retrieval_dataset):
    directory = retrieval_dataset
    catalogue, queries, qrels = load_dataset(directory, directory)
    assert len(catalogue) == 2 and queries[0]["qrels_complete"] is False
    assert qrels == {"q": {"a": 2}}
    write_csv(directory / "queries.csv", ["query_id", "query_text", "qrels_complete"], [{"query_id": "q", "query_text": "red shoe", "qrels_complete": "true"}])
    with pytest.raises(ValueError, match="every catalogue product"):
        load_dataset(directory, directory)
    write_csv(directory / "qrels.csv", ["query_id", "product_id", "relevance"], [{"query_id": "q", "product_id": "missing", "relevance": 2}])
    with pytest.raises(ValueError, match="unknown"):
        load_dataset(directory, directory)


def test_cuda_blocker_creates_no_performance_metrics(retrieval_dataset, monkeypatch):
    from stylesphere.common import runtime
    from stylesphere.retrieval.runner import run
    def blocked():
        raise RuntimeError("Synthetic CUDA-unavailable test condition")
    monkeypatch.setattr(runtime, "require_cuda", blocked)
    config = {"models": {"fashion_siglip": {"revision": "test"}, "siglip2": {"revision": "test"}},
              "evaluation": {"retrieval": {"dataset_dir": "."}}}
    result = run(retrieval_dataset, config, retrieval_dataset / "results", 42, smoke=True)
    assert result["blocked"] == 2
    import json
    records = [json.loads(line) for line in (retrieval_dataset / "results/retrieval/records.jsonl").read_text().splitlines()]
    assert all(r["status"] == "blocked" and "metrics" not in r and "timings" not in r for r in records)


def test_resume_rejects_missing_or_modified_saved_predictions(tmp_path):
    from stylesphere.retrieval.runner import _file_hash, _valid_artifacts
    prediction = tmp_path / "prediction.json"
    prediction.write_text('{"synthetic_unit_fixture":true}', encoding="utf-8")
    record = {"artifacts": {"prediction": "prediction.json"}, "artifact_sha256": {"prediction": _file_hash(prediction)}}
    assert _valid_artifacts(record, tmp_path)
    prediction.write_text('{"synthetic_unit_fixture":"changed"}', encoding="utf-8")
    assert not _valid_artifacts(record, tmp_path)
    prediction.unlink()
    assert not _valid_artifacts(record, tmp_path)
