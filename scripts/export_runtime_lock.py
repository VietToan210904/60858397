"""Export observed retrieval/assistance runtime dependencies using metadata only.

Run in the corresponding measured interpreter. No model, package or GPU is loaded;
no packages are installed and no network request is made. CUDA Torch/torchvision
binaries remain a separate explicit installation; their pure dependencies are kept.
"""
import argparse
from collections import deque
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["retrieval", "assistance"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Core CLI/reporting dependencies plus the selected native inference adapters.
    roots = ["numpy", "scipy", "PyYAML", "Pillow", "matplotlib", "torch", "torchvision",
             "transformers", "huggingface-hub", "safetensors"]
    roots += ["open-clip-torch", "sentencepiece", "requests"] if args.pair == "retrieval" else ["accelerate"]
    queue = deque((name, frozenset()) for name in roots)
    seen, versions, missing, violations = set(), {}, set(), []
    while queue:
        name, extras = queue.popleft()
        name = canonicalize_name(name)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.add(name)
            continue
        versions[name] = distribution.version
        for declared in distribution.requires or []:
            dependency = Requirement(declared)
            if dependency.marker and not any(dependency.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                continue
            try:
                version = metadata.version(dependency.name)
                if dependency.specifier and version not in dependency.specifier:
                    violations.append({"package": name, "declared_dependency": declared, "observed_version": version})
            except metadata.PackageNotFoundError:
                pass
            queue.append((dependency.name, frozenset(dependency.extras)))
    excluded = {"torch", "torchvision"}
    target = root / "environment" / f"requirements-{args.pair}-runtime.lock.txt"
    target.write_text(
        "# Observed Windows/Python runtime closure, not a cold-install verification.\n"
        "# Install CUDA Torch/torchvision separately, then this file with --no-deps.\n"
        "# Includes their pure dependencies; excludes unrelated inherited packages and test/build tools.\n" +
        "\n".join(f"{name}=={version}" for name, version in sorted(versions.items()) if name not in excluded) + "\n",
        encoding="utf-8")
    report = {"pair": args.pair, "python_version": platform.python_version(), "platform": sys.platform,
              "runtime_roots": roots, "root_versions": {canonicalize_name(name): versions.get(canonicalize_name(name)) for name in roots},
              "omitted_binary_packages": {name: versions.get(name) for name in sorted(excluded)},
              "missing_declared_dependencies": sorted(missing), "observed_constraint_mismatches": violations,
              "pinned_packages": len(versions.keys() - excluded), "metadata_only": True,
              "packages_installed": False, "model_weights_loaded": False, "gpu_inference_performed": False,
              "network_access_performed": False, "cold_install_verified": False,
              "note": "Observed metadata closure from the working interpreter; no fresh installation or dependency changes were performed. Optional quantization, test/build tools and unrelated globally inherited packages are excluded."}
    (root / "environment" / f"dependency-closure-{args.pair}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
