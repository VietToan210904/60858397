import numpy as np


def summary(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "std": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observations are not allowed")
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std(ddof=1)) if len(values) > 1 else None}


def paired_bootstrap(a, b, *, seed=8410, iterations=2000, confidence=0.95, clusters=None):
    """CI for mean(A-B); resample matched independent clusters with equal weight."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonempty finite matched one-dimensional observations required")
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    if clusters is not None:
        if len(clusters) != len(a):
            raise ValueError("Each pair needs a cluster")
        keys = list(dict.fromkeys(clusters))
        masks = [np.array([x == k for x in clusters]) for k in keys]
        a = np.array([a[mask].mean() for mask in masks])
        b = np.array([b[mask].mean() for mask in masks])
    d = a - b
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(iterations)])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return {"n_clusters": len(d), "mean_difference_a_minus_b": float(d.mean()), "ci_low": float(low) if len(d) > 1 else None, "ci_high": float(high) if len(d) > 1 else None, "confidence": confidence, "iterations": iterations, "seed": seed, "relative_difference": float((a.mean() - b.mean()) / b.mean()) if b.mean() != 0 else None}
