"""Build reviewed official 3D extensions using existing Windows tools only.

No driver, SDK, compiler or CUDA toolkit is installed. Dependencies and the local
CUDA PyTorch overlay must already be prepared in the selected isolated environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["spar3d", "sf3d"], required=True)
    parser.add_argument("--cuda-path", type=Path, required=True, help="Existing CUDA toolkit directory; this run selected v12.6.")
    parser.add_argument("--vcvars", type=Path, required=True, help="Existing vcvars64.bat; no Visual Studio installation is performed.")
    parser.add_argument("--vcvars-version", default="14.44")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--extension", choices=["all", "uv_unwrapper", "texture_baker"], default="all")
    parser.add_argument("--cuda-header-compatibility", action="store_true", help="Build an explicitly recorded overlay replacing the CUDA-only Python-binding header with torch/types.h; original checkout remains unchanged.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkout = root / "vendor" / ("SPAR3D" if args.model == "spar3d" else "SF3D")
    python = root / f".venv-{args.model}" / "Scripts" / "python.exe"
    if not all(path.exists() for path in (checkout, python, args.cuda_path / "bin/nvcc.exe", args.vcvars)):
        raise SystemExit("Existing checkout, interpreter, toolkit or compiler activation script is missing")
    source_revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("Tracked official source modifications require separate review before building")
    env = os.environ.copy()
    env.update({"CUDA_HOME": str(args.cuda_path.resolve()), "CUDA_PATH": str(args.cuda_path.resolve()),
                "TORCH_CUDA_ARCH_LIST": "8.9", "USE_NATIVE_ARCH": "0", "USE_CUDA": "1", "MAX_JOBS": str(args.max_jobs),
                "DISTUTILS_USE_SDK": "1", "MSSdk": "1"})
    env["PATH"] = str(args.cuda_path / "bin") + os.pathsep + str(python.parent) + os.pathsep + env["PATH"]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "model": args.model, "source_revision": source_revision,
              "python": python.relative_to(root).as_posix(), "cuda_toolkit": str(args.cuda_path), "vcvars_version": args.vcvars_version,
              "torch_cuda_arch_list": "8.9", "use_native_arch": False, "max_jobs": args.max_jobs,
              "system_components_installed": False, "gpu_inference_performed": False, "extensions": []}
    report_path = root / "environment" / f"native-build-{args.model}.json"
    archive_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if report_path.exists():
        shutil.copy2(report_path, report_path.with_name(report_path.stem + "-" + archive_suffix + ".json"))
    extensions = ("uv_unwrapper", "texture_baker") if args.extension == "all" else (args.extension,)
    for extension in extensions:
        log = root / "environment" / f"build-{args.model}-{extension}.log"
        if log.exists():
            shutil.copy2(log, log.with_name(log.stem + "-" + archive_suffix + ".log"))
        # A batch file keeps Windows' nested quote parsing out of subprocess argv.
        batch = root / "environment" / f"build-{args.model}-{extension}.cmd"
        build_source = checkout / extension
        patch_record = None
        if args.cuda_header_compatibility and extension == "texture_baker":
            build_source = root / ".cache" / "build-overlays" / args.model / extension
            shutil.copytree(checkout / extension, build_source, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "*.pyd", "__pycache__"))
            kernel = build_source / "texture_baker/csrc/baker_kernel.cu"
            original = kernel.read_bytes()
            if original.count(b"#include <torch/extension.h>") != 1:
                raise SystemExit("Expected CUDA include was not found exactly once; review changed source")
            modified = original.replace(b"#include <torch/extension.h>", b"#include <torch/types.h>")
            kernel.write_bytes(modified)
            patch_path = root / "environment" / f"{args.model}-cuda-header-compatibility.patch"
            patch_path.write_text("--- a/texture_baker/texture_baker/csrc/baker_kernel.cu\n+++ b/texture_baker/texture_baker/csrc/baker_kernel.cu\n@@ -4,1 +4,1 @@\n-#include <torch/extension.h>\n+#include <torch/types.h>\n", encoding="utf-8")
            patch_record = {"source_file": "texture_baker/texture_baker/csrc/baker_kernel.cu", "original_sha256": hashlib.sha256(original).hexdigest(),
                            "overlay_sha256": hashlib.sha256(modified).hexdigest(), "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                            "patch_path": patch_path.relative_to(root).as_posix(), "changes_kernels_or_model": False}
        batch.write_text(f'@echo off\ncall "{args.vcvars}" -vcvars_ver={args.vcvars_version}\nif errorlevel 1 exit /b %errorlevel%\n"{python}" -m pip install --ignore-installed --no-deps --no-build-isolation "{build_source}"\n', encoding="utf-8")
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        report["extensions"].append({"extension": extension, "returncode": process.returncode, "elapsed_seconds": time.perf_counter() - start,
                                     "log": log.relative_to(root).as_posix(), "build_source": build_source.relative_to(root).as_posix(), "compatibility_patch": patch_record})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["extensions"][-1]), flush=True)
    raise SystemExit(0 if all(record["returncode"] == 0 for record in report["extensions"]) else 1)


if __name__ == "__main__":
    main()
