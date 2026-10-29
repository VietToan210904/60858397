"""Sequential local-GPU retrieval with saved rankings and resumable records."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .adapters import MODEL_IDS, Encoder
from .data import load_dataset
from .metrics import score_query


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, value) -> None:
    from stylesphere.common.io import scrub
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _append(path: Path, record: dict) -> None:
    from stylesphere.common.io import scrub
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(scrub(record), ensure_ascii=False, allow_nan=False) + "\n")


def _artifact(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _valid_artifacts(record: dict, root: Path) -> bool:
    artifacts = record.get("artifacts", {})
    hashes = record.get("artifact_sha256", {})
    if not artifacts or set(artifacts) != set(hashes):
        return False
    for key, location in artifacts.items():
        path = Path(location)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or _file_hash(path) != hashes[key]:
            return False
    return True


def run(root: Path, config: dict, output_dir: Path, seed: int, smoke: bool = False, model_keys: list[str] | None = None) -> dict:
    from stylesphere.common.runtime import Measurement, require_cuda

    directory = output_dir / "retrieval"
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "records.jsonl"
    run_id = uuid.uuid4().hex
    mode = "smoke" if smoke else "evaluation"
    options = config.get("evaluation", {}).get("retrieval", {})
    keys = model_keys or list(MODEL_IDS)
    if any(key not in MODEL_IDS for key in keys):
        raise ValueError("Unknown retrieval model key")
    config_hash = _hash({"models": {key: config["models"][key] for key in MODEL_IDS}, "options": options, "seed": seed, "mode": mode})
    dataset_dir = root / options.get("dataset_dir", "datasets/retrieval")
    base = {"schema_version": 1, "run_id": run_id, "pair": "retrieval", "seed": seed, "mode": mode,
            "provenance": "model_inference", "config_sha256": config_hash, "protocol_sha256": config.get("protocol_sha256")}
    counts = {"success": 0, "failed": 0, "blocked": 0, "resumed": 0}
    try:
        catalogue, queries, qrels = load_dataset(dataset_dir, root)
        if smoke:
            # Catalogue stays identical; a smoke query subset is never a final benchmark.
            queries = queries[:int(options.get("smoke_queries", 2))]
        input_descriptor = {"manifests": {name: _file_hash(dataset_dir / name) for name in ("catalogue.csv", "queries.csv", "qrels.csv")},
                            "images": {row["product_id"]: _file_hash(Path(row["resolved_image_path"])) for row in catalogue}}
        input_hash = _hash(input_descriptor)
    except Exception as exc:
        for key in keys:
            record = {**base, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": key, "model_id": MODEL_IDS[key],
                      "checkpoint_revision": config["models"][key].get("revision"), "case_id": "__dataset__", "status": "blocked",
                      "input_sha256": None, "error": {"type": type(exc).__name__, "message": str(exc)}, "artifacts": {}}
            _append(records_path, record)
            counts["blocked"] += 1
        return {"pair": "retrieval", **counts, "records": str(records_path)}
    previous = _read(records_path)
    successful = {(r["model"], r["case_id"]) for r in previous if r.get("status") == "success" and r.get("config_sha256") == config_hash and r.get("input_sha256") == input_hash and r.get("mode") == mode and r.get("protocol_sha256") == config.get("protocol_sha256") and _valid_artifacts(r, root)}
    for key in keys:
        model_config = dict(config["models"][key])
        for field in ("local_path", "tokenizer_local_path"):
            if model_config.get(field) and not Path(model_config[field]).is_absolute():
                model_config[field] = str((root / model_config[field]).resolve())
        pending = [q for q in queries if (key, q["query_id"]) not in successful]
        counts["resumed"] += len(queries) - len(pending)
        if not pending:
            continue
        encoder = None
        model_base = {**base, "model": key, "model_id": MODEL_IDS[key], "input_sha256": input_hash,
                      "checkpoint_revision": model_config.get("revision"), "precision": model_config.get("precision", "fp16")}
        try:
            require_cuda()
            import numpy as np
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with Measurement() as load_measure:
                encoder = Encoder(key, model_config)
            model_base["checkpoint_revision"] = encoder.revision
            model_base["device"] = {"type": "cuda", "index": 0, "name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "compiled_cuda": torch.version.cuda}
            # Excluded warmup uses the same first catalogue image and a fixed neutral query.
            warmup_count = int(options.get("warmup_runs", config.get("evaluation", {}).get("warmup_runs", 1)))
            if warmup_count < 0:
                raise ValueError("warmup_runs cannot be negative")
            for _ in range(warmup_count):
                encoder.encode_prepared(encoder.prepare_images([catalogue[0]["resolved_image_path"]]), "image")
                encoder.encode_prepared(encoder.prepare_text(["fashion product"]), "text")
            fingerprint = _hash({"inputs": input_hash, "model": model_config, "revision": encoder.revision})[:16]
            model_dir = directory / key / fingerprint
            model_dir.mkdir(parents=True, exist_ok=True)
            descriptor_path = model_dir / "input_descriptor.json"
            _write(descriptor_path, input_descriptor)
            frozen_artifacts = {"input_descriptor": _artifact(descriptor_path, root)}
            frozen_hashes = {"input_descriptor": _file_hash(descriptor_path)}
            for manifest_name in ("catalogue.csv", "queries.csv", "qrels.csv"):
                frozen_path = model_dir / manifest_name
                frozen_path.write_bytes((dataset_dir / manifest_name).read_bytes())
                if _file_hash(frozen_path) != input_descriptor["manifests"][manifest_name]:
                    raise ValueError("Dataset manifest changed during execution; freeze the dataset before retrying")
                artifact_key = "frozen_" + frozen_path.stem
                frozen_artifacts[artifact_key] = _artifact(frozen_path, root)
                frozen_hashes[artifact_key] = _file_hash(frozen_path)
            index_path = model_dir / "catalogue_index.npz"
            index_meta_path = model_dir / "index_metadata.json"
            index_reused = index_path.exists() and index_meta_path.exists()
            if index_reused:
                with np.load(index_path, allow_pickle=False) as saved:
                    embeddings = saved["embeddings"]
                    product_ids = saved["product_ids"].tolist()
                index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
                if index_meta.get("index_sha256") != _file_hash(index_path):
                    raise ValueError("Saved catalogue index has changed since it was created")
                if product_ids != [row["product_id"] for row in catalogue] or embeddings.shape[0] != len(catalogue):
                    raise ValueError("Saved retrieval index does not match catalogue identifiers")
            else:
                batches = []
                preprocessing_s = 0.0
                indexing_inference_s = 0.0
                peak_allocated = 0.0
                peak_reserved = 0.0
                indexing_start = time.perf_counter()
                batch_size = int(options.get("batch_size", 16))
                if batch_size < 1:
                    raise ValueError("batch_size must be positive")
                for start in range(0, len(catalogue), batch_size):
                    with Measurement() as preprocessing:
                        batch = encoder.prepare_images([row["resolved_image_path"] for row in catalogue[start:start + batch_size]])
                    with Measurement() as inference:
                        batches.append(encoder.encode_prepared(batch, "image"))
                    preprocessing_s += preprocessing.elapsed_s
                    indexing_inference_s += inference.elapsed_s
                    peak_allocated = max(peak_allocated, inference.peak_allocated_mb)
                    peak_reserved = max(peak_reserved, inference.peak_reserved_mb)
                embeddings = np.concatenate(batches)
                product_ids = [row["product_id"] for row in catalogue]
                np.savez_compressed(index_path, embeddings=embeddings, product_ids=np.asarray(product_ids, dtype=str))
                index_meta = {"model": key, "model_id": MODEL_IDS[key], "checkpoint_revision": encoder.revision,
                              "input_sha256": input_hash, "catalogue_count": len(catalogue), "embedding_dimension": int(embeddings.shape[1]),
                              "indexing_s": time.perf_counter() - indexing_start, "preprocessing_s": preprocessing_s,
                              "inference_s": indexing_inference_s, "peak_allocated_mb": peak_allocated, "peak_reserved_mb": peak_reserved,
                              "index_sha256": _file_hash(index_path),
                              "warmup_runs": warmup_count, "tokenizer_revision": getattr(encoder, "tokenizer_revision", encoder.revision)}
                _write(index_meta_path, index_meta)
            for query in pending:
                record = {**model_base, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "case_id": query["query_id"], "query_text": query["query_text"]}
                try:
                    with Measurement() as preprocessing:
                        inputs = encoder.prepare_text([query["query_text"]])
                    with Measurement() as inference:
                        query_embedding = encoder.encode_prepared(inputs, "text")[0]
                    rank_start = time.perf_counter()
                    scores = embeddings @ query_embedding
                    # Stable identifiers break equal-score ties, independent of file order.
                    order = sorted(range(len(product_ids)), key=lambda i: (-float(scores[i]), product_ids[i]))
                    ranking = [product_ids[i] for i in order]
                    ranking_s = time.perf_counter() - rank_start
                    prediction = {"query_id": query["query_id"], "query_text": query["query_text"], "ranking": ranking,
                                  "scores": [float(scores[i]) for i in order], "qrels": qrels[query["query_id"]], "qrels_complete": query["qrels_complete"], "recall_k": int(options.get("recall_k", 10))}
                    prediction_path = model_dir / f"{query['query_id']}.json"
                    _write(prediction_path, prediction)
                    record.update(status="success", metrics=score_query(ranking, qrels[query["query_id"]], complete=query["qrels_complete"], k=int(options.get("recall_k", 10))),
                                  timings={"model_load_s": load_measure.elapsed_s, "preprocessing_s": preprocessing.elapsed_s,
                                           "query_inference_s": inference.elapsed_s, "ranking_s": ranking_s,
                                           "end_to_end_s": preprocessing.elapsed_s + inference.elapsed_s + ranking_s,
                                           "catalogue_indexing_s": index_meta["indexing_s"],
                                           "peak_allocated_mb": inference.peak_allocated_mb, "peak_reserved_mb": inference.peak_reserved_mb},
                                  artifacts={**frozen_artifacts, "prediction": _artifact(prediction_path, root), "index": _artifact(index_path, root), "index_metadata": _artifact(index_meta_path, root)},
                                  artifact_sha256={**frozen_hashes, "prediction": _file_hash(prediction_path), "index": _file_hash(index_path), "index_metadata": _file_hash(index_meta_path)},
                                  index_reused=index_reused, warmup_runs=warmup_count)
                except Exception as exc:
                    record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, artifacts={})
                counts[record["status"]] += 1
                _append(records_path, record)
        except Exception as exc:
            status = "failed" if "out of memory" in str(exc).lower() else "blocked"
            for query in pending:
                record = {**model_base, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "case_id": query["query_id"], "status": status,
                          "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "artifacts": {}}
                counts[status] += 1
                _append(records_path, record)
        finally:
            if encoder is not None:
                encoder.close()
    latest = {}
    for record in _read(records_path):
        if record.get("config_sha256") == config_hash and record.get("input_sha256") == input_hash:
            latest[(record["model"], record["case_id"])] = record
    aggregate = {}
    for key in keys:
        successful_records = [r for (model, _), r in latest.items() if model == key and r["status"] == "success"]
        metrics = {}
        for name in ("ndcg_at_10", "mrr", "hit_at_5", f"recall_at_{int(options.get('recall_k', 10))}"):
            values = [r["metrics"][name] for r in successful_records if r["metrics"].get(name) is not None]
            metrics[name] = {"mean": sum(values) / len(values) if values else None, "eligible_queries": len(values)}
        model_records = [r for (model, _), r in latest.items() if model == key]
        attempted = [r for r in model_records if r["status"] != "blocked"]
        aggregate[key] = {"metrics": metrics, "query_count": len(model_records), "failed_query_rate": sum(r["status"] == "failed" for r in attempted) / len(attempted) if attempted else None,
                          "blocked_count": sum(r["status"] == "blocked" for r in model_records)}
    _write(directory / f"summary_{mode}.json", {"mode": mode, "input_sha256": input_hash, "config_sha256": config_hash, "models": aggregate})
    return {"pair": "retrieval", **counts, "records": str(records_path)}
