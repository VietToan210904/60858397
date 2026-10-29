"""Synthetic unit fixtures only; these are never experiment observations."""
import json
import csv
from pathlib import Path

import pytest

from stylesphere.generation.review import aggregate_ratings, export_review
from stylesphere.generation.runner import append, file_digest, safe_id, invoke_worker, successful_prior


def test_rating_aggregation_preserves_prompt_clusters_and_missingness():
    mapping = [{"review_id": "a", "model": "A", "case_id": "p1"},
               {"review_id": "b", "model": "A", "case_id": "p1"},
               {"review_id": "c", "model": "A", "case_id": "p2"},
               {"review_id": "d", "model": "A", "case_id": "p3"}]
    ratings = [{"review_id": "a", "rater_id": "r1", "prompt_adherence": "1", "usable": "0"},
               {"review_id": "a", "rater_id": "r2", "prompt_adherence": "3", "usable": "1"},
               {"review_id": "b", "rater_id": "r1", "prompt_adherence": "4", "usable": "1"},
               {"review_id": "c", "rater_id": "r1", "prompt_adherence": "5", "usable": "1"},
               {"review_id": "d", "rater_id": "", "prompt_adherence": "", "usable": ""}]
    summary = aggregate_ratings(ratings, mapping)
    # Raters a=(1+3)/2=2; p1=(2+4)/2=3; p2=5; model=(3+5)/2=4.
    assert summary["models"]["A"]["prompt_adherence"] == {"mean": 4.0, "n_prompts": 2}
    assert summary["models"]["A"]["usable_vote_rate"]["mean"] == pytest.approx(.875)
    assert summary["models"]["A"]["usable_concept_rate"]["mean"] == 1
    assert summary["usable_coverage"] == {"n_images_total": 4, "n_images_with_votes": 3, "n_images_decided": 2,
                                          "n_images_disagreement": 1, "n_images_missing_votes": 1}
    assert summary["missing_review_images"] == 1
    assert summary["n_assessments"] == 4


@pytest.mark.parametrize("value", ["0", "6", "2.5", "nan"])
def test_invalid_likert_ratings_rejected(value):
    with pytest.raises(ValueError):
        aggregate_ratings([{"review_id": "a", "rater_id": "r", "composition": value}], [{"review_id": "a", "model": "A", "case_id": "p"}])


def test_duplicate_rater_rejected():
    row = {"review_id": "a", "rater_id": "r", "usable": "1"}
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_ratings([row, row], [{"review_id": "a", "model": "A", "case_id": "p"}])


def test_empty_ratings_are_missing_not_zero():
    assert aggregate_ratings([], [{"review_id": "a", "model": "A", "case_id": "p"}])["models"] == {}


def test_unanimous_concepts_cluster_images_then_prompts_and_keep_disagreements_missing():
    mapping = [{"review_id": key, "model": "A", "case_id": prompt} for key, prompt in [("a", "p1"), ("b", "p1"), ("c", "p2"), ("d", "p3")]]
    ratings = [{"review_id": "a", "rater_id": "r1", "usable": 0}, {"review_id": "a", "rater_id": "r2", "usable": 0},
               {"review_id": "b", "rater_id": "r1", "usable": 1}, {"review_id": "c", "rater_id": "r1", "usable": 1},
               {"review_id": "d", "rater_id": "r1", "usable": 1}, {"review_id": "d", "rater_id": "r2", "usable": 0}]
    result = aggregate_ratings(ratings, mapping)
    # p1=(0+1)/2=.5, p2=1, p3 unresolved: model=(.5+1)/2=.75.
    assert result["models"]["A"]["usable_concept_rate"] == {"mean": .75, "n_prompts": 2}
    assert "usable_concept_rate" not in next(p["metrics"] for p in result["per_prompt"] if p["case_id"] == "p3")
    assert result["usable_coverage"]["n_images_disagreement"] == 1


def test_failed_measurement_retains_actual_window(monkeypatch):
    from stylesphere.generation import worker
    from types import SimpleNamespace
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append("sync"), reset_peak_memory_stats=lambda: calls.append("reset"),
                           max_memory_allocated=lambda: 42 * 2**20, max_memory_reserved=lambda: 80 * 2**20)
    moments = iter([10.0, 10.5])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(moments))
    def fail():
        raise RuntimeError("synthetic unit failure")
    with pytest.raises(RuntimeError) as caught:
        worker.measure(SimpleNamespace(cuda=cuda), fail)
    assert caught.value._stylesphere_measurement == {"elapsed_s": .5, "peak_allocated_mb": 42., "peak_reserved_mb": 80.}
    assert calls == ["sync", "reset", "sync"]


def test_fp8_dispatch_observation_preserves_result_and_failure():
    from stylesphere.generation.worker import counted_fp8_matmul
    counters = {"attempts": 0, "successes": 0, "failures": 0}
    sentinel = object()
    assert counted_fp8_matmul(lambda: sentinel, counters)() is sentinel
    failure = RuntimeError("synthetic kernel failure")
    def fail():
        raise failure
    with pytest.raises(RuntimeError) as caught:
        counted_fp8_matmul(fail, counters)()
    assert caught.value is failure
    assert counters == {"attempts": 2, "successes": 1, "failures": 1, "first_failure": "RuntimeError: synthetic kernel failure"}


def test_review_is_blind_and_blank_and_preserves_existing_ratings(tmp_path):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    source = tmp_path / "secret_model.png"
    metadata = PngInfo()
    metadata.add_text("model", "secret_model")
    Image.new("RGB", (4, 4)).save(source, pnginfo=metadata)
    records = [{"run_id": "synthetic_unit_test", "model": "secret_model", "case_id": "p1", "seed": 1,
                "status": "success", "prompt": "<safe brief>", "artifacts": {"image": str(source)}}]
    directory = tmp_path / "review"
    export_review(records, directory)
    assert "secret_model" not in (directory / "review.html").read_text(encoding="utf-8")
    assert "&lt;safe brief&gt;" in (directory / "review.html").read_text(encoding="utf-8")
    with Image.open(next((directory / "images").glob("*.png"))) as im:
        assert "model" not in im.info
    csv = (directory / "ratings.csv").read_text(encoding="utf-8")
    assert csv.splitlines()[1].endswith(",,,,,,,,")
    (directory / "ratings.csv").write_text("existing human work", encoding="utf-8")
    export_review(records, directory)
    assert (directory / "ratings.csv").read_text() == "existing human work"


def test_case_id_cannot_escape_output_directory():
    with pytest.raises(ValueError):
        safe_id("../../outside")


def test_missing_worker_interpreter_reports_blocked(tmp_path):
    result = invoke_worker(Path("worker.py"), {}, tmp_path / "diagnostics", str(tmp_path / "missing_python"), 2)
    assert result["status"] == "blocked"
    assert result["results"] == []


def test_resume_requires_artifact_hash_and_identical_mode_protocol(tmp_path):
    artifact = tmp_path / "image.png"
    artifact.write_bytes(b"synthetic unit fixture only")
    rows = tmp_path / "records.jsonl"
    record = {"status": "success", "model": "A", "case_id": "p", "seed": 1, "mode": "smoke", "config_sha256": "cfg",
              "input_sha256": "input", "protocol_sha256": "protocol", "artifacts": {"image": str(artifact)},
              "artifact_sha256": {"image": file_digest(artifact)}}
    append(rows, record)
    match = dict(model="A", case_id="p", seed=1, mode="smoke", config_sha="cfg", input_sha="input", protocol_sha="protocol")
    assert successful_prior(rows, **match) == record
    assert successful_prior(rows, **(match | {"mode": "evaluation"})) is None
    assert successful_prior(rows, **(match | {"protocol_sha": "changed"})) is None
    artifact.write_bytes(b"changed unit fixture")
    assert successful_prior(rows, **match) is None


def _canonical_fixture(root, protocol="synthetic_protocol", mode="evaluation", suffix=""):
    from PIL import Image
    from stylesphere.common.io import canonical_hash
    prompt = {"prompt_id": "p", "prompt": "Synthetic unit brief", "mandatory_criteria": []}
    frozen = root / "fixture_prompts.jsonl"
    frozen.write_text(json.dumps(prompt) + "\n")
    rows = []
    for index, model in enumerate(("sdxl_lightning", "flux2_klein")):
        image = root / "results" / "generation" / (model + suffix + ".png")
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (3, 3), (index * 80, 0, 0)).save(image)
        row = {"status": "success", "mode": mode, "provenance": "model_inference", "run_id": "synthetic" + suffix,
               "model": model, "case_id": "p", "seed": 1, "input_sha256": canonical_hash(prompt), "config_sha256": model,
               "protocol_sha256": protocol, "prompt": "Synthetic unit brief", "artifacts": {"image": image.relative_to(root).as_posix(), "frozen_prompts": frozen.relative_to(root).as_posix()},
               "artifact_sha256": {"image": file_digest(image), "frozen_prompts": file_digest(frozen)}}
        append(root / "results" / "generation" / "records.jsonl", row)
        rows.append(row)
    return rows


def test_canonical_review_binds_sources_and_refuses_remap_after_human_work(tmp_path):
    from stylesphere.generation.canonical_review import export_canonical_review, validate_canonical_review
    rows = _canonical_fixture(tmp_path)
    result = export_canonical_review(tmp_path, tmp_path / "results")
    assert result["images"] == 2
    destination = tmp_path / result["directory"]
    blank_before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    assert validate_canonical_review(tmp_path, tmp_path / "results")["status"] == "existing_validated_review"
    assert all(path.read_bytes() == value for path, value in blank_before.items())
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["n_matched_inputs"] == 1
    assert {b["source_image_sha256"] for b in manifest["source_bindings"]} == {r["artifact_sha256"]["image"] for r in rows}
    with (destination / "ratings.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, ratings = reader.fieldnames, list(reader)
    assert all(not any(value for key, value in row.items() if key != "review_id") for row in ratings)
    ratings[0].update(rater_id="synthetic_rater", usable="1")
    with (destination / "ratings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ratings)
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    assert export_canonical_review(tmp_path, tmp_path / "results")["status"] == "existing_frozen_review"
    assert all(path.read_bytes() == value for path, value in before.items())
    _canonical_fixture(tmp_path, suffix="new_run")
    with pytest.raises(ValueError, match="source selection changed"):
        export_canonical_review(tmp_path, tmp_path / "results")
    assert all(path.read_bytes() == value for path, value in before.items())


def test_canonical_review_requires_matching_benchmarks_and_verified_sources(tmp_path):
    from stylesphere.generation.canonical_review import export_canonical_review
    rows = _canonical_fixture(tmp_path, mode="smoke")
    with pytest.raises(ValueError, match="one matching"):
        export_canonical_review(tmp_path, tmp_path / "results")
    rows = _canonical_fixture(tmp_path)
    (tmp_path / rows[0]["artifacts"]["image"]).write_bytes(b"altered synthetic source")
    with pytest.raises(ValueError, match="SHA256"):
        export_canonical_review(tmp_path, tmp_path / "results")


def test_canonical_review_explicit_protocol_and_changed_frozen_mapping(tmp_path):
    from stylesphere.generation.canonical_review import export_canonical_review
    _canonical_fixture(tmp_path, protocol="one")
    _canonical_fixture(tmp_path, protocol="two", suffix="two")
    with pytest.raises(ValueError, match="one matching"):
        export_canonical_review(tmp_path, tmp_path / "results")
    result = export_canonical_review(tmp_path, tmp_path / "results", protocol_sha256="two")
    destination = tmp_path / result["directory"]
    ratings = (destination / "ratings.csv").read_text()
    lines = ratings.splitlines()
    lines[1] = lines[1].replace(",,,,,,,,", ",synthetic_rater,,,,,,,a note")
    (destination / "ratings.csv").write_text("\n".join(lines) + "\n")
    (destination / "private_mapping.json").write_text("[]")
    with pytest.raises(ValueError, match="mapping changed"):
        export_canonical_review(tmp_path, tmp_path / "results", protocol_sha256="two")


def test_canonical_review_rejects_a_brief_changed_after_inference(tmp_path):
    from stylesphere.generation.canonical_review import export_canonical_review
    rows = _canonical_fixture(tmp_path)
    append(tmp_path / "results/generation/records.jsonl", {**rows[0], "prompt": "Altered unit brief"})
    with pytest.raises(ValueError, match="prompt differs"):
        export_canonical_review(tmp_path, tmp_path / "results")
    assert not (tmp_path / "results/generation/human_review/ratings.csv").exists()


def test_frozen_prompt_direct_and_historical_sidecar_verify_exact_full_row(tmp_path):
    from stylesphere.common.io import canonical_hash
    from stylesphere.generation.provenance import freeze_prompts, bind_existing_run, verify_prompt_binding
    prompt = {"prompt_id": "p", "prompt": "Synthetic unit brief", "mandatory_criteria": ["one object"], "category": "unit"}
    source = tmp_path / "dataset.jsonl"
    source.write_text(json.dumps(prompt) + "\n")
    pair = tmp_path / "results" / "generation"
    row = {"run_id": "synthetic", "model": "sdxl_lightning", "case_id": "p", "seed": 1, "protocol_sha256": "unit",
           "input_sha256": canonical_hash(prompt), "prompt": prompt["prompt"], "mandatory_criteria": prompt["mandatory_criteria"]}
    append(pair / "records.jsonl", row)
    raw_before = (pair / "records.jsonl").read_bytes()
    frozen = freeze_prompts(tmp_path, source, pair / "synthetic")
    direct = {**row, "artifacts": {"frozen_prompts": frozen.relative_to(tmp_path).as_posix()},
              "artifact_sha256": {"frozen_prompts": file_digest(frozen)}}
    assert verify_prompt_binding(direct, tmp_path) == []
    assert "input_sha256" in verify_prompt_binding({**direct, "input_sha256": "changed"}, tmp_path)[0]
    assert "criteria" in verify_prompt_binding({**direct, "mandatory_criteria": []}, tmp_path)[0]
    bind_existing_run(tmp_path, pair, "synthetic", source, expected_sha256=file_digest(source))
    assert verify_prompt_binding(row, tmp_path) == []
    assert (pair / "records.jsonl").read_bytes() == raw_before
    assert "audited" in verify_prompt_binding({**row, "prompt": "changed"}, tmp_path)[0]
    source.write_text("changed source")
    assert verify_prompt_binding(row, tmp_path) == []  # original frozen exact bytes survive source changes
    with pytest.raises(ValueError, match="previously frozen"):
        freeze_prompts(tmp_path, source, pair / "synthetic")
    frozen.write_text("tampered evidence")
    assert "SHA256" in verify_prompt_binding(row, tmp_path)[0]


def test_runner_retains_hash_of_invalid_export_without_claiming_valid_output(tmp_path, monkeypatch):
    from stylesphere.generation import runner
    from stylesphere.generation.provenance import verify_prompt_binding
    prompt = {"prompt_id": "unit", "prompt": "Synthetic unit prompt", "mandatory_criteria": ["unit only"]}
    dataset = tmp_path / "datasets/generation"
    dataset.mkdir(parents=True)
    (dataset / "prompts.jsonl").write_text(json.dumps(prompt) + "\n")
    def invalid_export(worker, request, directory, python, timeout, working_dir=None):
        case = request["cases"][0]
        path = tmp_path / case["image"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic invalid PNG fixture")
        return {"results": [{"case_id": case["case_id"], "seed": case["seed"], "status": "success", "metrics": {},
                              "artifacts": {"image": case["image"]}, "timings": {"inference_s": .1}}]}
    monkeypatch.setattr(runner, "invoke_worker", invalid_export)
    summary = runner.run(tmp_path, {}, tmp_path / "results", 1, smoke=True, model_keys=["sdxl_lightning"])
    assert summary["failed"] == 1 and summary["success"] == 0
    row = json.loads((tmp_path / "results/generation/records.jsonl").read_text())
    assert row["metrics"]["valid_output"] == 0
    assert set(row["artifacts"]) == set(row["artifact_sha256"]) == {"image", "frozen_prompts"}
    assert row["artifact_sha256"]["image"] == file_digest(tmp_path / row["artifacts"]["image"])
    assert verify_prompt_binding(row, tmp_path) == []
