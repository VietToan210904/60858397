"""Ranking metrics. Unjudged documents have zero gain; missing relevance is explicit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _validate(ranking: Sequence[str], qrels: Mapping[str, int], k: int | None = None) -> None:
    if len(set(ranking)) != len(ranking):
        raise ValueError("A ranking must contain unique product identifiers")
    if k is not None and (type(k) is not int or k < 1):
        raise ValueError("k must be positive")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in qrels.values()):
        raise ValueError("Relevance grades must be nonnegative integers")


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float | None:
    """Exponential gain (2**grade-1), log2 discount; None without positive judgments."""
    _validate(ranking, qrels, k)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    if not idcg:
        return None
    dcg = sum((2 ** qrels.get(pid, 0) - 1) / math.log2(i + 2) for i, pid in enumerate(ranking[:k]))
    return dcg / idcg


def mean_reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, int]) -> float | None:
    """Single-query reciprocal rank; average only the eligible queries for MRR."""
    _validate(ranking, qrels)
    if not any(v > 0 for v in qrels.values()):
        return None
    return next((1 / i for i, pid in enumerate(ranking, 1) if qrels.get(pid, 0) > 0), 0.0)


def hit_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 5) -> float | None:
    _validate(ranking, qrels, k)
    if not any(v > 0 for v in qrels.values()):
        return None
    return float(any(qrels.get(pid, 0) > 0 for pid in ranking[:k]))


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10, *, complete: bool = False) -> float | None:
    _validate(ranking, qrels, k)
    if not complete:
        return None
    relevant = {pid for pid, grade in qrels.items() if grade > 0}
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else None


def score_query(ranking: Sequence[str], qrels: Mapping[str, int], *, complete: bool = False, k: int = 10) -> dict:
    return {
        "ndcg_at_10": ndcg_at_k(ranking, qrels, 10),
        "mrr": mean_reciprocal_rank(ranking, qrels),
        "hit_at_5": hit_at_k(ranking, qrels, 5),
        f"recall_at_{k}": recall_at_k(ranking, qrels, k, complete=complete),
        "judged_relevant_count": sum(v > 0 for v in qrels.values()),
        "ranking_metrics_eligible": any(v > 0 for v in qrels.values()),
        "qrels_complete": complete,
        "unjudged_retrieved_count": sum(pid not in qrels for pid in ranking[:10]),
    }
