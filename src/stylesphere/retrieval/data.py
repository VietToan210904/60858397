"""Strict catalogue/query/qrels loading without heavyweight optional dependencies."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
        return list(reader)


def _unique(rows: list[dict], key: str) -> None:
    ids = [row[key].strip() for row in rows]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{key} must be a unique safe identifier containing letters, digits, underscores or hyphens")


def load_dataset(directory: Path, root: Path, *, require_images: bool = True) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    catalogue = read_csv(directory / "catalogue.csv", {"product_id", "image_path", "category", "source"})
    queries = read_csv(directory / "queries.csv", {"query_id", "query_text"})
    labels = read_csv(directory / "qrels.csv", {"query_id", "product_id", "relevance"})
    if not catalogue or not queries:
        raise ValueError("Retrieval requires a nonempty real catalogue and query set")
    _unique(catalogue, "product_id")
    _unique(queries, "query_id")
    pids = {row["product_id"] for row in catalogue}
    qids = {row["query_id"] for row in queries}
    for row in catalogue:
        image = Path(row["image_path"])
        image = image if image.is_absolute() else root / image
        if require_images and not image.is_file():
            raise FileNotFoundError(f"Missing catalogue image for {row['product_id']}: {image}")
        row["resolved_image_path"] = str(image)
    for row in queries:
        if not row["query_text"].strip():
            raise ValueError("query_text cannot be empty")
        flag = str(row.get("qrels_complete", "false")).lower()
        if flag not in {"true", "false", "1", "0", ""}:
            raise ValueError("qrels_complete must be true or false")
        row["qrels_complete"] = flag in {"true", "1"}
    qrels: dict[str, dict[str, int]] = {qid: {} for qid in qids}
    for row in labels:
        qid, pid = row["query_id"], row["product_id"]
        if qid not in qids or pid not in pids:
            raise ValueError(f"qrels references unknown query/product: {qid}, {pid}")
        try:
            grade = int(row["relevance"])
        except (TypeError, ValueError) as exc:
            raise ValueError("qrels relevance must be an integer") from exc
        if grade < 0 or pid in qrels[qid]:
            raise ValueError("qrels grades must be nonnegative and query/product pairs unique")
        qrels[qid][pid] = grade
    for row in queries:
        if row["qrels_complete"] and set(qrels[row["query_id"]]) != pids:
            raise ValueError("A complete qrels declaration requires an explicit judgment for every catalogue product")
    return catalogue, queries, qrels
