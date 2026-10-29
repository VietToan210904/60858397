"""Isolated real inference worker. Invoked by runner, never downloads weights.

ComfyUI orchestration uses reviewed core nodes, without loading custom nodes or
starting a server. Source/API references are in docs/PAIR2_PAIR4_SOURCES.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback


def measure(torch, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    def snapshot():
        result = {"elapsed_s": time.perf_counter() - start}
        for name, getter in (("peak_allocated_mb", torch.cuda.max_memory_allocated), ("peak_reserved_mb", torch.cuda.max_memory_reserved)):
            try:
                result[name] = getter() / 2**20
            except Exception as exc:
                result[name] = None
                result["measurement_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        result = fn()
        torch.cuda.synchronize()
    except Exception as exc:
        synchronization_error = None
        try:
            torch.cuda.synchronize()
        except Exception as sync_exc:
            synchronization_error = f"{type(sync_exc).__name__}: {sync_exc}"
        failed = snapshot()
        if synchronization_error:
            failed["synchronization_error"] = synchronization_error
        exc._stylesphere_measurement = failed
        raise
    return result, snapshot()


class SDXLLightning:
    def __init__(self, config: dict):
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        base = config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        rev = config.get("revision", "main")
        base_rev = config.get("base_revision", "main")
        options = {"local_files_only": True, "revision": base_rev, "cache_dir": config.get("cache_dir")}
        unet_config = UNet2DConditionModel.load_config(base, subfolder="unet", **options)
        # Avoid allocating an unneeded FP32 copy before loading the selected UNet.
        with torch.device("meta"):
            unet = UNet2DConditionModel.from_config(unet_config)
        checkpoint = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors", revision=rev, local_files_only=True, cache_dir=config.get("cache_dir"))
        self.resolved_revision = Path(checkpoint).parent.name
        unet.load_state_dict(load_file(checkpoint, device="cpu"), assign=True)
        unet.to(dtype=torch.float16)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", **options)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        if config.get("cpu_offload", True):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.enable_vae_slicing()
        self.runtime_precision = {"unet_parameter_dtypes": sorted({str(p.dtype) for p in unet.parameters()})}

    def generate(self, prompt: str, seed: int, width: int, height: int):
        import torch
        return self.pipe(prompt, num_inference_steps=4, guidance_scale=0.0, width=width, height=height,
                         generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]


class FluxKleinFP8:
    def __init__(self, config: dict):
        if not config.get("checkout"):
            raise ValueError("Set flux2_klein.checkout to a reviewed local ComfyUI repository")
        checkout = Path(config["checkout"]).resolve()
        if not (checkout / "nodes.py").is_file():
            raise FileNotFoundError("FLUX FP8 needs a reviewed local ComfyUI checkout containing nodes.py")
        source = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        self.source_revision = source.stdout.strip() if source.returncode == 0 else None
        changes = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
        self.source_dirty = bool(changes.stdout.strip())
        if config.get("source_revision") and (self.source_revision != config["source_revision"] or self.source_dirty):
            raise ValueError("ComfyUI checkout must be clean and match the configured source_revision commit")
        # The worker runs in this isolated interpreter; do not initialize custom nodes.
        sys.path.insert(0, str(checkout))
        import nodes
        import folder_paths
        from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage, Flux2Scheduler
        import comfy.samplers

        for family, key, expected in (("diffusion_models", "checkpoint", "flux-2-klein-4b-fp8.safetensors"),
                                      ("text_encoders", "text_encoder", "qwen_3_4b.safetensors"),
                                      ("vae", "vae", "flux2-vae.safetensors")):
            path = Path(config.get(key, checkout / "models" / family / expected)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing local {family} asset: {path}")
            if key == "checkpoint" and path.name != expected:
                raise ValueError("Exact BFL distilled 4B FP8 checkpoint filename required; no Base/BF16 substitution")
            folder_paths.add_model_folder_path(family, str(path.parent), is_default=True)
            config[key] = path.name
        self.model = nodes.UNETLoader().load_unet(config["checkpoint"], "default")[0]
        self.clip = nodes.CLIPLoader().load_clip(config["text_encoder"], type="flux2", device="default")[0]
        self.vae = nodes.VAELoader().load_vae(config["vae"])[0]
        self.nodes, self.guider, self.noise, self.sampler = nodes, BasicGuider, RandomNoise, SamplerCustomAdvanced
        self.latent, self.scheduler = EmptyFlux2LatentImage, Flux2Scheduler
        self.euler = comfy.samplers.sampler_object("euler")
        self.resolved_revision = config.get("revision", "main")
        import comfy.quant_ops
        dtype_counts = {}
        for parameter in self.model.model.diffusion_model.parameters():
            name = str(parameter.dtype)
            dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
        backends = comfy.quant_ops.ck.list_backends() if getattr(comfy.quant_ops, "_CK_AVAILABLE", False) else {}
        self.runtime_precision = {"checkpoint_format": "official_scaled_fp8", "diffusion_parameter_dtype_counts": dtype_counts,
                                  "comfy_kitchen_backends": {name: {field: details.get(field) for field in ("available", "disabled", "unavailable_reason")}
                                                             for name, details in backends.items()}}

    def generate(self, prompt: str, seed: int, width: int, height: int):
        import numpy as np
        from PIL import Image

        condition = self.nodes.CLIPTextEncode().encode(self.clip, prompt)[0]
        # BasicGuider is the guidance-distilled CFG=1 path in the official workflow.
        guider = self.guider.execute(self.model, condition)[0]
        noise = self.noise.execute(seed)[0]
        sigmas = self.scheduler.execute(4, width, height)[0]
        latent = self.latent.execute(width, height, 1)[0]
        sampled = self.sampler.execute(noise, guider, self.euler, sigmas, latent)[0]
        pixels = self.nodes.VAEDecode().decode(self.vae, sampled)[0][0]
        return Image.fromarray(np.clip(pixels.cpu().numpy() * 255, 0, 255).astype(np.uint8))


def main(request_path: str, response_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    results = []
    environment = {"python": sys.executable}
    stage = "preflight"
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        environment.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}, compiled_cuda={torch.version.cuda}; local GPU execution required")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        config = request["model_config"]
        if request["model"] == "flux2_klein" and torch.cuda.get_device_capability(0) < (8, 9):
            raise RuntimeError("Exact FP8 evaluation requires CUDA compute capability >= 8.9; no BF16 substitution")
        adapter_cls = SDXLLightning if request["model"] == "sdxl_lightning" else FluxKleinFP8
        stage = "model_load"
        adapter, loading = measure(torch, lambda: adapter_cls(config))
        environment.update(device=torch.cuda.get_device_name(0), source_revision=getattr(adapter, "source_revision", None),
                           source_dirty=getattr(adapter, "source_dirty", None), resolved_checkpoint_revision=getattr(adapter, "resolved_revision", None),
                           runtime_precision=getattr(adapter, "runtime_precision", None))
        def infer(case):
            with torch.inference_mode():
                return adapter.generate(case["prompt"], case["seed"], request["width"], request["height"])
        if request["cases"]:
            first = request["cases"][0]
            stage = "warmup"
            for _ in range(request.get("warmup_runs", 1)):
                measure(torch, lambda: infer(first))
            torch.cuda.synchronize()
        for case in request["cases"]:
            timing, stage = None, "inference"
            try:
                image, timing = measure(torch, lambda: infer(case))
                stage = "export"
                path = Path(case["image"])
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                results.append({"case_id": case["case_id"], "seed": case["seed"], "status": "success",
                                "timings": {"inference_s": timing.pop("elapsed_s"), "model_load_s": loading["elapsed_s"]},
                                "metrics": timing, "artifacts": {"image": str(path)}, "error": None})
            except Exception as exc:
                traceback.print_exc()
                failed = dict(getattr(exc, "_stylesphere_measurement", None) or timing or {})
                elapsed = failed.pop("elapsed_s", None)
                diagnostics = {"stage": stage, "cuda_out_of_memory": isinstance(exc, torch.cuda.OutOfMemoryError),
                               **{k: failed.pop(k) for k in ("measurement_error", "synchronization_error") if k in failed}}
                results.append({"case_id": case["case_id"], "seed": case["seed"], "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                "timings": {"inference_s": elapsed} if elapsed is not None else {}, "metrics": failed, "diagnostics": diagnostics})
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            temporary = Path(response_path).with_suffix(".tmp")
            temporary.write_text(json.dumps({"results": results, **environment}, indent=2), encoding="utf-8")
            temporary.replace(response_path)
        response = {"results": results, "python": sys.executable, "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "source_revision": getattr(adapter, "source_revision", None), "source_dirty": getattr(adapter, "source_dirty", None),
                    "runtime_precision": getattr(adapter, "runtime_precision", None),
                    "resolved_checkpoint_revision": getattr(adapter, "resolved_revision", None)}
    except Exception as exc:
        measured = getattr(exc, "_stylesphere_measurement", None)
        response = {"results": [], "status": "blocked", **environment, "error": f"{type(exc).__name__}: {exc}",
                    "diagnostics": {"stage": stage, "cuda_out_of_memory": type(exc).__name__ == "OutOfMemoryError", "failed_window": measured}}
        traceback.print_exc()
    Path(response_path).write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0 if response.get("status") != "blocked" else 2


if __name__ == "__main__":
    # Remove our CLI args before ComfyUI parses its own command line on import.
    request_file, response_file = sys.argv[1:3]
    sys.argv = [sys.argv[0]]
    raise SystemExit(main(request_file, response_file))
