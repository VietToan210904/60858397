"""Offline GPU worker around reviewed official SPAR3D/SF3D Python interfaces."""
from __future__ import annotations

import json
import hashlib
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import random
import sys
import subprocess
import time
import traceback


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_reference(root: Path, path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError("Retained 3D outputs must remain within the project root for portable packaging") from None


def checked_asset(weights: Path, value: str, expected: str, label: str) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (weights / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing explicitly acquired local {label}; automatic auxiliary downloads are disabled")
    if len(expected) != 64 or file_sha256(path) != expected:
        raise ValueError(f"Pinned local {label} SHA256 mismatch")
    return path


def verify_checkpoint(config: dict, weights: Path) -> dict:
    """Bind the local final checkpoint/config to the inspected immutable snapshot."""
    planned = config.get("checkpoint_files", {})
    if set(planned) != {"config.yaml", "model.safetensors"}:
        raise ValueError("Configure both official checkpoint file identities from the pinned acquisition plan")
    evidence = {"revision": config.get("revision"), "files": {}}
    for name, item in planned.items():
        path = weights / name
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Pinned final checkpoint file size differs: {name}")
        digest = file_sha256(path)
        if item.get("sha256"):
            if digest != item["sha256"]:
                raise ValueError(f"Pinned final checkpoint SHA256 differs: {name}")
        elif item.get("git_blob_sha1"):
            data = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != item["git_blob_sha1"]:
                raise ValueError(f"Pinned final checkpoint Git blob differs: {name}")
        else:
            raise ValueError(f"Missing official final checkpoint content identity: {name}")
        evidence["files"][name] = {"size_bytes": path.stat().st_size, "sha256": digest}
    return evidence


def verify_native_build(config: dict, weights: Path) -> dict:
    build = config.get("extension_build")
    if not build:
        raise ValueError("Configure the reviewed native extension build receipt and binary hashes")
    receipt_path = checked_asset(weights, build["receipt"], build["receipt_sha256"], "native build receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("source_revision") != config.get("source_revision"):
        raise ValueError("Native build does not match the configured official source commit")
    if not receipt.get("extensions") or any(row.get("returncode") != 0 for row in receipt["extensions"]):
        raise ValueError("Native build receipt does not record successful extension compilation")
    patches = [row.get("compatibility_patch") for row in receipt["extensions"] if row.get("compatibility_patch")]
    if not patches or any(patch["patch_sha256"] != build["patch_sha256"] for patch in patches):
        raise ValueError("Native build does not record the expected reviewed CUDA include patch")
    # Receipt/patch paths are relative to the repository, which is two levels
    # above the configured models/<model> weight directory in this deployment.
    project_root = receipt_path.parent.parent
    for patch in patches:
        path = project_root / patch["patch_path"]
        if file_sha256(path) != build["patch_sha256"]:
            raise ValueError("Reviewed compatibility patch bytes changed")
    artifacts = {}
    for module in ("texture_baker._C", "uv_unwrapper._C"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise ImportError(f"Missing compiled native extension: {module}")
        path = Path(spec.origin)
        digest = file_sha256(path)
        if digest != build["binary_sha256"].get(module):
            raise ValueError(f"Native extension differs from the recorded build: {module}")
        artifacts[module] = {"path": str(path), "sha256": digest}
    return {"receipt_sha256": file_sha256(receipt_path), "patch_sha256": build["patch_sha256"], "artifacts": artifacts}


def redirect_dinov2_initializers(native, key: str, dino_dir: Path) -> list[str]:
    """Redirect only the pinned DINO initializers, including SPAR point diffusion."""
    from omegaconf import OmegaConf
    tokenizers = ["image_tokenizer"]
    if key == "spar3d":
        expected = "spar3d.models.tokenizers.image.DINOV2SingleImageTokenizer"
        if OmegaConf.select(native, "pdiff_image_tokenizer_cls") != expected:
            raise ValueError("Review unexpected SPAR point-diffusion image tokenizer class without substituting a backbone")
        tokenizers.append("pdiff_image_tokenizer")
    paths = [name + ".pretrained_model_name_or_path" for name in tokenizers]
    for path in paths:
        if OmegaConf.select(native, path, default="facebook/dinov2-large") != "facebook/dinov2-large":
            raise ValueError("The official config requests an unexpected DINOv2 initializer; review without substituting a backbone")
    for path in paths:
        OmegaConf.update(native, path, str(dino_dir), merge=False)
    return paths


def offline_initialisation_config(config: dict, weights: Path, key: str, destination: Path) -> tuple[Path, dict]:
    """Use exactly identified local initialisers without altering model architecture."""
    from omegaconf import OmegaConf
    native = OmegaConf.load(weights / "config.yaml")
    auxiliary = config.get("auxiliary", {})
    dino = auxiliary.get("dinov2")
    if not dino:
        raise ValueError("Configure the exact locally acquired DINOv2 initialisation snapshot")
    dino_dir = (weights / dino["path"]).resolve()
    dino_model = checked_asset(weights, str(dino_dir / "model.safetensors"), dino["model_sha256"], "DINOv2 weights")
    dino_config = checked_asset(weights, str(dino_dir / "config.json"), dino["config_sha256"], "DINOv2 config")
    redirected = redirect_dinov2_initializers(native, key, dino_dir)
    evidence = {"dinov2": {"repo_id": "facebook/dinov2-large", "revision": dino["revision"],
                           "weights_sha256": file_sha256(dino_model), "config_sha256": file_sha256(dino_config),
                           "configuration_paths": redirected}}
    if key == "sf3d":
        clip = auxiliary.get("open_clip")
        if not clip:
            raise ValueError("Configure the exact local OpenCLIP initialization checkpoint")
        if OmegaConf.select(native, "image_estimator.model", default="ViT-B-32") != "ViT-B-32" or OmegaConf.select(native, "image_estimator.pretrain", default="laion2b_s34b_b79k") != "laion2b_s34b_b79k":
            raise ValueError("Official SF3D config requests an unexpected OpenCLIP initializer; review without substitution")
        path = checked_asset(weights, clip["path"], clip["sha256"], "OpenCLIP initialization weights")
        # Official open_clip2.24 uses torch.load for local checkpoints, so retain
        # its exact official .bin format rather than silently loading safetensors.
        if path.suffix != ".bin":
            raise ValueError("The pinned official OpenCLIP2.24 initializer requires its .bin checkpoint")
        OmegaConf.update(native, "image_estimator.pretrain", str(path), merge=False)
        evidence["open_clip"] = {"repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "revision": clip["revision"], "sha256": file_sha256(path)}
    else:
        clip = auxiliary.get("alpha_clip")
        if not clip:
            raise ValueError("Configure the exact local AlphaCLIP/OpenAI initialization checkpoint")
        if OmegaConf.select(native, "image_estimator.model", default="ViT-L/14@336px") != "ViT-L/14@336px":
            raise ValueError("Official SPAR3D config requests an unexpected AlphaCLIP initializer; review without substitution")
        official_sha = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
        if clip["sha256"] != official_sha:
            raise ValueError("AlphaCLIP initialization must use the official OpenAI ViT-L/14@336px SHA256")
        path = checked_asset(weights, clip["path"], official_sha, "AlphaCLIP/OpenAI initialization weights")
        # An explicit existing path bypasses AlphaCLIP's independent URL loader,
        # which otherwise ignores HF_HUB_OFFLINE.
        OmegaConf.update(native, "image_estimator.model", str(path), merge=False)
        os.environ["ALPHA_CLIP_PATH"] = str(path.parent)
        evidence["alpha_clip"] = {"model": "OpenAI ViT-L/14@336px", "sha256": file_sha256(path)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(native, destination)
    evidence["operational_config_sha256"] = file_sha256(destination)
    evidence["changes"] = "Only local initialization asset paths; architecture, kernels and final model weights unchanged"
    return destination, evidence


def deny_network(event, args):
    if event == "socket.connect":
        raise RuntimeError("Network connection attempted during offline 3D inference; pre-acquire all reviewed assets")


def timed(torch, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    def snapshot():
        measured = {"elapsed_s": time.perf_counter() - start}
        errors = {}
        for name, getter in (("peak_allocated_mb", torch.cuda.max_memory_allocated),
                             ("peak_reserved_mb", torch.cuda.max_memory_reserved)):
            try:
                measured[name] = getter() / 2**20
            except Exception as exc:
                measured[name] = None
                errors[name] = f"{type(exc).__name__}: {exc}"
        if errors:
            measured["measurement_errors"] = errors
        return measured
    try:
        result = operation()
        torch.cuda.synchronize()
    except Exception as exc:
        synchronization_error = None
        try:
            torch.cuda.synchronize()
        except Exception as sync_exc:
            synchronization_error = f"{type(sync_exc).__name__}: {sync_exc}"
        measured = snapshot()
        if synchronization_error:
            measured["synchronization_error"] = synchronization_error
        exc._stylesphere_measurement = measured
        raise
    return result, snapshot()


def case_wall_timings(start: float, preprocessing_end: float | None, warmup_s: float | None, end: float) -> dict:
    """Observed worker case wall window, subtracting only measured warmup."""
    measured = {"preprocessing_s": (preprocessing_end if preprocessing_end is not None else end) - start,
                "end_to_end_s": end - start - (warmup_s if warmup_s is not None else 0)}
    if warmup_s is not None:
        measured["warmup_excluded_s"] = warmup_s
    return measured


def failed_case(case: dict, exc: Exception, stage: str, timing: dict | None, *, is_oom: bool,
                artifacts: dict | None = None, wall_timings: dict | None = None) -> dict:
    """Keep actual measured inference windows separate from failed warmup/load."""
    measured = dict(getattr(exc, "_stylesphere_measurement", None) or timing or {})
    elapsed = measured.pop("elapsed_s", None)
    diagnostics = {"stage": stage, "cuda_out_of_memory": is_oom}
    for name in ("measurement_errors", "synchronization_error"):
        if name in measured:
            diagnostics[name] = measured.pop(name)
    timings, metrics = {}, {}
    if stage in {"inference", "export"}:
        if elapsed is not None:
            timings["inference_s"] = elapsed
        metrics = measured
    else:
        diagnostics["failed_window"] = {"elapsed_s": elapsed, **measured} if elapsed is not None else None
    timings.update(wall_timings or {})
    return {"case_id": case["case_id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "timings": timings, "metrics": metrics, "diagnostics": diagnostics, "artifacts": artifacts or {}}


def seed_case(torch, numpy, seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(request_path: str, response_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    environment = {"python": sys.executable, "adapter_source_sha256": file_sha256(Path(__file__)),
                   "seed_policy": "Reset Python/NumPy/Torch/CUDA seed immediately before each measured inference, after excluded warmup"}
    results, stage = [], "preflight"
    try:
        config, key = request["model_config"], request["model"]
        project_root = Path(request.get("project_root", Path(__file__).resolve().parents[3])).resolve()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        if key == "spar3d" and config.get("low_vram", True):
            os.environ["SPAR3D_LOW_VRAM"] = "1"
        import torch
        environment.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
        libraries = {}
        for package in ("torch", "torchvision", "numpy", "scipy", "Pillow", "transformers", "omegaconf", "trimesh", "safetensors", "huggingface-hub", "clip", "alpha-clip", "open-clip-torch"):
            try:
                libraries[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                libraries[package] = None
        environment["libraries"] = libraries
        import numpy as np
        from PIL import Image
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}, compiled_cuda={torch.version.cuda}; CPU fallback is disabled")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not config.get("checkout"):
            raise ValueError("Set checkout to a reviewed local official SPAR3D/SF3D repository")
        checkout = Path(config["checkout"]).resolve()
        if not (checkout / "run.py").is_file():
            raise FileNotFoundError("Set checkout to a reviewed local official SPAR3D/SF3D repository")
        source = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        source_revision = source.stdout.strip() if source.returncode == 0 else None
        changes = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
        source_dirty = bool(changes.stdout.strip())
        if config.get("source_revision") and (source_revision != config["source_revision"] or source_dirty):
            raise ValueError("3D checkout must be clean and match configured source_revision commit")
        sys.path.insert(0, str(checkout))
        os.chdir(checkout)
        # Resolve the exact gated snapshot locally; never trigger from_pretrained downloads.
        if config.get("weights_path"):
            weights = Path(config["weights_path"])
        else:
            from huggingface_hub import snapshot_download
            weights = Path(snapshot_download(repo_id=request["model_id"], revision=config.get("revision", "main"), local_files_only=True))
        if not all((weights / name).is_file() for name in ("config.yaml", "model.safetensors")):
            raise FileNotFoundError("Missing locally approved model config.yaml/model.safetensors snapshot")
        environment["checkpoint_provenance"] = verify_checkpoint(config, weights)
        environment["native_extension_provenance"] = verify_native_build(config, weights)
        operational_config, auxiliary_evidence = offline_initialisation_config(config, weights, key, Path(response_path).resolve().parent / "operational_model_config.yaml")
        environment["auxiliary_provenance"] = auxiliary_evidence
        sys.addaudithook(deny_network)
        if key == "spar3d":
            from spar3d.system import SPAR3D
            from spar3d.utils import foreground_crop
            model_cls = SPAR3D
            preprocessing = lambda image: foreground_crop(image, float(config.get("foreground_ratio", 1.3)))
            extra = {"low_vram_mode": bool(config.get("low_vram", True))}
        else:
            from sf3d.system import SF3D
            from sf3d.utils import resize_foreground
            model_cls = SF3D
            preprocessing = lambda image: resize_foreground(image, float(config.get("foreground_ratio", 0.85)))
            extra = {}
        def load():
            model = model_cls.from_pretrained(str(weights), config_name=str(operational_config), weight_name="model.safetensors", **extra)
            model.to("cuda")
            model.eval()
            return model
        stage = "model_load"
        model, loading = timed(torch, load)
        environment.update(device=torch.cuda.get_device_name(0), source_revision=source_revision, source_dirty=source_dirty,
                           resolved_checkpoint_revision=weights.name if weights.parent.name == "snapshots" else config.get("revision", "main"))
        for index, case in enumerate(request["cases"]):
            timing, stage, prepared_path = None, "preprocessing", None
            case_start = time.perf_counter()
            preprocessing_end, warmup_s = None, None
            try:
                seed_case(torch, np, case["seed"])
                with Image.open(case["source_image"]) as source:
                    if source.mode != "RGBA" or source.getextrema()[3][0] == 255:
                        raise ValueError("Supply an identically prepared foreground RGBA input with transparency; automatic background-model downloads are disabled")
                    image = preprocessing(source.copy())
                directory = Path(case["mesh"]).parent
                directory.mkdir(parents=True, exist_ok=True)
                prepared_path = directory / "prepared_input.png"
                image.save(prepared_path)
                preprocessing_end = time.perf_counter()
                def infer():
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        mesh, _ = model.run_image(image, bake_resolution=request["texture_resolution"], remesh="none")
                    return mesh
                if index == 0 and request.get("warmup_runs", 1) > 0:
                    stage = "warmup"
                    warmup_start = time.perf_counter()
                    try:
                        for _ in range(request.get("warmup_runs", 1)):
                            timed(torch, infer)
                        torch.cuda.synchronize()
                    finally:
                        warmup_s = time.perf_counter() - warmup_start
                stage = "inference"
                # Warmup may consume SPAR diffusion RNG. Restore the declared
                # per-case seed so outputs do not depend on warmup/resume order.
                seed_case(torch, np, case["seed"])
                mesh, timing = timed(torch, infer)
                stage = "export"
                export_start = time.perf_counter()
                mesh.export(case["mesh"], include_normals=True)
                case_end = time.perf_counter()
                results.append({"case_id": case["case_id"], "status": "success", "timings": {"inference_s": timing.pop("elapsed_s"),
                                "model_load_s": loading["elapsed_s"], "export_s": case_end - export_start,
                                **case_wall_timings(case_start, preprocessing_end, warmup_s, case_end)},
                                "metrics": timing, "artifacts": {"mesh": artifact_reference(project_root, case["mesh"]),
                                "prepared_input": artifact_reference(project_root, directory / "prepared_input.png")}, "error": None})
            except Exception as exc:
                case_end = time.perf_counter()
                traceback.print_exc()
                artifacts = {}
                for name, path in (("prepared_input", prepared_path), ("mesh", Path(case["mesh"]))):
                    if path is not None and path.is_file():
                        artifacts[name] = artifact_reference(project_root, path)
                failure = failed_case(case, exc, stage, timing, is_oom=isinstance(exc, torch.cuda.OutOfMemoryError), artifacts=artifacts,
                                      wall_timings=case_wall_timings(case_start, preprocessing_end, warmup_s, case_end))
                if stage == "export":
                    failure["timings"]["export_s"] = case_end - export_start
                results.append(failure)
                if failure["diagnostics"]["cuda_out_of_memory"]:
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cleanup_exc:
                        failure["diagnostics"]["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            temporary = Path(response_path).with_suffix(".tmp")
            temporary.write_text(json.dumps({"results": results, **environment}, indent=2), encoding="utf-8")
            temporary.replace(response_path)
        response = {"results": results, **environment, "python": sys.executable, "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "source_revision": source_revision, "source_dirty": source_dirty,
                    "resolved_checkpoint_revision": weights.name if weights.parent.name == "snapshots" else config.get("revision", "main")}
    except Exception as exc:
        traceback.print_exc()
        response = {"results": results, "status": "blocked", **environment, "error": f"{type(exc).__name__}: {exc}",
                    "diagnostics": {"stage": stage, "cuda_out_of_memory": type(exc).__name__ == "OutOfMemoryError",
                                    "failed_window": getattr(exc, "_stylesphere_measurement", None)}}
    Path(response_path).write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 2 if response.get("status") == "blocked" else 0


if __name__ == "__main__":
    request_file, response_file = sys.argv[1:3]
    sys.argv = [sys.argv[0]]
    raise SystemExit(main(request_file, response_file))
