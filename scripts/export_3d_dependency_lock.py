"""Export the observed 3D runtime dependency closure, excluding shared Torch binaries.

Run in each corresponding model environment. The resulting lock installs all
required pure/runtime packages locally and never relies on global site-packages.
CLIP/AlphaCLIP are installed separately from their reviewed source commits.
"""
import argparse
from collections import deque
from importlib import metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sf3d", "spar3d"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = ["torch", "torchvision", "setuptools", "wheel", "ninja", "numpy", "scipy", "pillow", "packaging",
             "einops", "jaxtyping", "omegaconf", "transformers", "trimesh", "huggingface-hub", "safetensors"]
    roots += ["open-clip-torch", "rembg", "gpytoolbox", "pynanoinstantmeshes"] if args.model == "sf3d" else ["clip", "alpha-clip", "loralib", "transparent-background"]
    queue, seen, versions, missing, violations = deque((name, frozenset()) for name in roots), set(), {}, set(), []
    unused_gui_cli = {"cookiecutter", "fastapi", "qrcode", "uvicorn", "watchdog"}
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
            if name == "flet" and canonicalize_name(dependency.name) in unused_gui_cli:
                continue  # The reviewed inference worker never launches Flet GUI/CLI/server.
            if dependency.marker and not any(dependency.marker.evaluate({"extra": extra}) for extra in {"", *extras}):
                continue
            try:
                version = metadata.version(dependency.name)
                if dependency.specifier and version not in dependency.specifier:
                    violations.append({"package": name, "declared_dependency": declared, "observed_version": version})
            except metadata.PackageNotFoundError:
                pass
            queue.append((dependency.name, frozenset(dependency.extras)))
    excluded = {"torch", "torchvision", "clip", "alpha-clip"}
    lock = root / "environment" / f"requirements-{args.model}-runtime.lock.txt"
    lock.write_text("# Observed runtime dependency closure; install with --no-deps into a clean model venv.\n# Torch/torchvision come only from the local shared CUDA environment; CLIP/AlphaCLIP from reviewed source.\n" +
                    "\n".join(f"{name}=={version}" for name, version in sorted(versions.items()) if name not in excluded) + "\n", encoding="utf-8")
    report = {"model": args.model, "roots": roots, "omitted_shared_or_reviewed_sources": sorted(excluded), "missing_declared_dependencies": sorted(missing),
              "unused_flet_gui_cli_dependencies_not_in_runtime": sorted(unused_gui_cli) if args.model == "spar3d" else [],
              "observed_constraint_mismatches": violations, "pinned_packages": len(versions.keys() - excluded),
              "cold_install_verified": False, "note": "Observed imports/builds/kernel checks are verified separately; a fresh-extraction installation has not been executed in this run."}
    (root / "environment" / f"dependency-closure-{args.model}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
