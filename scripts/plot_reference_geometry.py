"""Export figures from genuine, separately labelled GSO test measurements."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stylesphere.common.io import sha256_file


def main():
    root = Path.cwd()
    source = root / "results/reference_test/reference_metrics.jsonl"
    rows = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    data = {(r["model"], r["case_id"]): r for r in rows if r["status"] == "success"}
    cases = sorted({c for m, c in data if m == "sf3d"} & {c for m, c in data if m == "spar3d"})
    if len(cases) != 20:
        raise ValueError("Figure requires all 20 paired test objects, excluding development")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.3), constrained_layout=True)
    specs = [("chamfer_squared", "Squared Chamfer (lower is better)"),
             ("fscore_at_0.02", "F-score at 2% diagonal (higher is better)"),
             ("supplementary_silhouette_iou", "Supplementary silhouette IoU")]
    for row_index, condition in enumerate(("unaligned", "rigid_aligned")):
        for column, (metric, label) in enumerate(specs):
            ax = axes[row_index, column]
            values = [[data[(model, c)]["metrics"][condition + "_" + metric] for c in cases] for model in ("sf3d", "spar3d")]
            boxes = ax.boxplot(values, positions=[1, 2], widths=0.45, patch_artist=True,
                               medianprops={"color": "black", "linewidth": 1.5}, showfliers=False)
            for box, color in zip(boxes["boxes"], ("#8DB6D8", "#EDBD86")):
                box.set_facecolor(color)
                box.set_alpha(0.65)
            for i, case in enumerate(cases):
                backpack = data[("sf3d", case)]["category"] == "backpack"
                ax.plot([1, 2], [values[0][i], values[1][i]], color="#355570" if backpack else "#777777",
                        alpha=0.65 if backpack else 0.25, linewidth=0.8,
                        marker="^" if backpack else ".", markersize=5 if backpack else 4)
            ax.set_xticks([1, 2], ["SF3D", "SPAR3D"])
            ax.set_xlim(0.6, 2.4)
            ax.set_title(label, fontsize=10)
            ax.grid(axis="y", alpha=0.2)
            if column:
                ax.set_ylim(0, 1)
            if column == 0:
                ax.set_ylabel("Normalized shape\n" + ("Exported orientation" if row_index == 0 else "After reference-assisted rigid ICP"), fontsize=10)
    fig.suptitle("Controlled GSO reference benchmark: 20 paired real scanned objects\n"
                 "Rendered inputs; independent unit-diagonal normalization; 17 shoes + 3 backpacks (triangles)", fontsize=13)
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    outputs = [directory / "reference_geometry_comparison.png", directory / "reference_geometry_comparison.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    receipt = {"source": source.relative_to(root).as_posix(), "source_sha256": sha256_file(source),
               "paired_test_objects": len(cases), "development_included": False, "matplotlib_version": matplotlib.__version__,
               "outputs": {p.relative_to(root).as_posix(): sha256_file(p) for p in outputs}}
    (root / "environment/reference_figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
