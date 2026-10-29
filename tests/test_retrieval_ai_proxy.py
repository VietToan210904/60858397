import copy
import itertools
from pathlib import Path

import pytest

from stylesphere.retrieval.ai_assessment import TRAIT_KEYS
from stylesphere.retrieval.proxy_metrics import clustered_paired_bootstrap, proxy_grade, ranking_metrics, read_jsonl, validate_profiles


def fixture():
    evidence = {"product_id": "P1", "annotation_index": 1, "image_sha256": "a" * 64,
                "source_facts": {"item_name": [{"language_tag": "en_GB", "value": "A shoe"}]}}
    profile = {"product_id": "P1", "annotation_index": 1, "image_sha256": "a" * 64,
               "matched_types": ["casual_shoes"], "uncertain_types": [],
               "traits": dict.fromkeys(TRAIT_KEYS, 0), "confidence": "high",
               "visual_notes": "An explicitly reviewed photograph.", "source_support": [],
               "explicit_lightweight_source": None}
    return evidence, profile


def test_uncertainty_is_retained_and_wrong_type_is_zero():
    _, profile = fixture()
    profile["uncertain_types"] = ["running_shoes"]
    profile["traits"]["black"] = None
    assert proxy_grade(profile, "running_shoes", "black") is None
    assert proxy_grade(profile, "casual_shoes", "black") is None
    assert proxy_grade(profile, "earrings", "black") == 0


def test_proxy_grade_rules_and_travel_limit():
    _, profile = fixture()
    for value, expected in [(0, 0), (1, 1), (2, 3)]:
        profile["traits"]["formal"] = value
        assert proxy_grade(profile, "casual_shoes", "formal") == expected
    profile["traits"]["travel_appearance"] = 2
    assert proxy_grade(profile, "casual_shoes", "travel_appearance") == 2
    profile["explicit_lightweight_source"] = True
    assert proxy_grade(profile, "casual_shoes", "travel_appearance") == 3


def test_exact_source_citation_arrays_and_scalar_values_supported():
    evidence, profile = fixture()
    for citation in [{"field": "item_name", "language_tag": "en_GB", "value": "A shoe"},
                     {"field": "item_name", "exact_source_values": [{"language_tag": "en_GB", "value": "A shoe"}]}]:
        profile["source_support"] = [citation]
        validate_profiles([evidence], [profile])
    profile["source_support"][0]["exact_source_values"][0]["value"] = "Invented lightweight claim"
    with pytest.raises(ValueError, match="exact retained multilingual"):
        validate_profiles([evidence], [profile])


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(image_sha256="changed"), "identity differs"),
    (lambda p: p["traits"].update(black=True), "ten visual traits"),
    (lambda p: p.update(explicit_lightweight_source=True), "exact source citation"),
])
def test_profile_tampering_rejected(mutation, match):
    evidence, profile = fixture()
    mutation(profile)
    with pytest.raises(ValueError, match=match):
        validate_profiles([evidence], [profile])


def test_duplicates_and_missing_catalogue_items_rejected():
    evidence, profile = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        validate_profiles([evidence], [profile, copy.deepcopy(profile)])
    with pytest.raises(ValueError, match="Every catalogue"):
        validate_profiles([evidence], [])


def test_conservative_ndcg_bounds_contain_every_unknown_assignment():
    ranking = ["b", "a", "c", "d"]
    grades = {"a": 3, "b": None, "c": 1, "d": None}
    result = ranking_metrics(ranking, grades)
    bounds = result["conservative_bounds"]
    for b, d in itertools.product(range(4), repeat=2):
        actual = ranking_metrics(ranking, {"a": 3, "b": b, "c": 1, "d": d})["known_labels_unknown_zero_gain"]["ndcg_at_10"]
        assert bounds["ndcg_at_10_lower"] <= actual <= bounds["ndcg_at_10_upper"]
    assert result["n_unknown_top10"] == 2
    assert result["judged_coverage_at_10"] == .5
    assert result["all_catalogue_complete"] is False


def test_unknown_qrels_never_mutated_into_negative_labels():
    grades = {"a": None, "b": 0}
    result = ranking_metrics(["a", "b"], grades)
    assert grades["a"] is None
    assert result["known_labels_unknown_zero_gain"]["ndcg_at_10"] is None
    assert result["all_unknown_grade3_sensitivity"]["hit_at_10"] == 1


def test_actual_600_authored_profiles_bind_to_original_images_and_sources():
    directory = Path(__file__).resolve().parents[1] / "results/retrieval/ai_assessment/v1"
    evidence = read_jsonl(directory / "product_evidence.jsonl")
    profiles = [row for path in (directory / "annotations").glob("*.jsonl") for row in read_jsonl(path)]
    validate_profiles(evidence, profiles)
    assert len(profiles) == 600
    assert len({p["product_id"] for p in profiles}) == 600


def test_paired_bootstrap_resamples_twelve_related_query_families():
    rows = []
    for index in range(1, 121):
        for model in ("fashion_siglip", "siglip2"):
            value = 0 if model == "fashion_siglip" else (1 if (index - 1) // 10 < 6 else -1)
            values = dict.fromkeys(["ndcg_at_10", "mrr_at_10", "hit_at_10", "precision_at_10", "recall_at_10"], value)
            rows.append({"model": model, "query_id": f"abo-q{index:03d}",
                         "metrics": {"known_labels_unknown_zero_gain": values, "all_unknown_grade3_sensitivity": values}})
    result = clustered_paired_bootstrap(rows, repetitions=100)
    assert all(row["n_paired_queries"] == 120 for row in result)
    assert all(row["n_assessable_query_family_clusters"] == 12 for row in result)
    assert all(row["paired_mean_difference"] == 0 for row in result)
