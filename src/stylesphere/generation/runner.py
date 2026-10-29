"""Offline-by-default generation experiment orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .review import export_review
from .provenance import freeze_prompts
from stylesphere.common.io import scrub, portable_path

MODEL_IDS = {"sdxl_lightning": "ByteDance/SDXL-Lightning", "flux2_klein": "black-forest-labs/FLUX.2-klein-4b-fp8"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Case IDs must contain only letters, digits, underscores and hyphens")
    return value


def successful_prior(path: Path, *, model: str, case_id: str, seed: int, mode: str, config_sha: str, input_sha: str, protocol_sha, root: Path | None = None) -> dict | None:
    """Resume only a matched successful record whose actual artifacts still hash."""
    if not path.is_file():
        return None
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if (row.get("status"), row.get("model"), row.get("case_id"), row.get("seed"), row.get("mode"), row.get("config_sha256"), row.get("input_sha256"), row.get("protocol_sha256")) == ("success", model, case_id, seed, mode, config_sha, input_sha, protocol_sha):
                matches.append(row)
    for row in reversed(matches):
        artifacts, hashes = row.get("artifacts", {}), row.get("artifact_sha256", {})
        if artifacts and set(artifacts) == set(hashes):
            try:
                if all(resolve(root or Path.cwd(), value).is_file() and file_digest(resolve(root or Path.cwd(), value)) == hashes[name] for name, value in artifacts.items()):
                    return row
            except OSError:
                pass
    return None


def append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(scrub(record), ensure_ascii=False, allow_nan=False) + "\n")


def resolve(root: Path, value: str | Path) -> Path:
    return portable_path(root, value).resolve()


def base_record(pair, model, model_id, case_id, config, seed, smoke, run_id, input_value):
    return {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
            "pair": pair, "model": model, "model_id": model_id, "case_id": case_id, "status": "blocked",
            "seed": seed, "mode": "smoke" if smoke else "evaluation", "provenance": "model_inference",
            "checkpoint_revision": config.get("revision", "main"), "input_sha256": digest(input_value),
            "model_licence": config.get("licence"),
            "config_sha256": digest(config), "metrics": {}, "timings": {}, "artifacts": {}, "error": None}


def invoke_worker(worker: Path, request: dict, directory: Path, python: str, timeout: float, working_dir: Path | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    req, resp = directory / "request.json", directory / "response.json"
    req.write_text(json.dumps(scrub(request), indent=2), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    # Captured diagnostics never include an environment dump or token values.
    try:
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([python, str(worker), str(req), str(resp)], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout, check=False, cwd=working_dir)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_path = directory / "worker.log"
        if log_path.is_file():
            log_path.write_text(scrub(log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        partial = json.loads(resp.read_text(encoding="utf-8")) if resp.is_file() else {"results": []}
        partial.update(status="blocked", error=scrub(f"{type(exc).__name__}: {exc}"))
        partial = scrub(partial)
        if resp.is_file():
            resp.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        return partial
    log_path = directory / "worker.log"
    log_path.write_text(scrub(log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    if not resp.is_file():
        return {"results": [], "status": "blocked", "error": f"Worker exited {result.returncode} without response; see {directory / 'worker.log'}"}
    response = scrub(json.loads(resp.read_text(encoding="utf-8")))
    resp.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return response


def run(root: Path, config: dict, output_dir: Path, seed: int, smoke: bool = False, model_keys: list[str] | None = None) -> dict:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    target = output_dir / "generation"
    run_id = uuid.uuid4().hex
    eval_config = config.get("evaluation", {}).get("generation", {})
    path = root / eval_config.get("dataset_dir", "datasets/generation") / "prompts.jsonl"
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({p.get("prompt_id") for p in prompts}) != len(prompts):
        raise ValueError("Duplicate generation prompt IDs")
    if smoke:
        prompts = prompts[:1]
    frozen_prompts = freeze_prompts(root, path, target / run_id)
    selected = model_keys or list(MODEL_IDS)
    records, resumed = [], 0
    for key in selected:
        if key not in MODEL_IDS:
            raise ValueError(f"Unknown generation model {key}")
        options = dict(config.get("models", {}).get(key, {}))
        for option in ("checkout", "text_encoder", "vae", "cache_dir"):
            if options.get(option):
                options[option] = str(resolve(root, options[option]))
        if key == "flux2_klein" and options.get("checkpoint"):
            options["checkpoint"] = str(resolve(root, options["checkpoint"]))
        if options.get("python") and any(s in options["python"] for s in ("/", "\\")):
            options["python"] = str(resolve(root, options["python"]))
        cases = []
        for prompt in prompts:
            if not prompt.get("prompt_id") or not prompt.get("prompt"):
                raise ValueError("Generation prompt requires prompt_id and prompt")
            safe_id(prompt["prompt_id"])
            seeds = [seed] if smoke else [seed + i for i in range(int(eval_config.get("repeats", eval_config.get("images_per_prompt", 3))))]
            for case_seed in seeds:
                cases.append({"case_id": prompt["prompt_id"], "prompt": prompt["prompt"], "seed": case_seed,
                              "image": (target / run_id / key / f"{prompt['prompt_id']}_{case_seed}.png").relative_to(root).as_posix()})
        request = {"model": key, "model_config": options, "cases": cases,
                   "warmup_runs": int(config.get("evaluation", {}).get("warmup_runs", 1)),
                   "width": int(eval_config.get("width", 1024)), "height": int(eval_config.get("height", 1024))}
        measurement_config = {**options, "width": request["width"], "height": request["height"], "steps": 4,
                              "warmup_runs": request["warmup_runs"], "guidance_scale": 0 if key == "sdxl_lightning" else 1}
        pending, templates = [], {}
        for case in cases:
            prompt = next(p for p in prompts if p["prompt_id"] == case["case_id"])
            template = base_record("generation", key, MODEL_IDS[key], case["case_id"], measurement_config, case["seed"], smoke, run_id, prompt)
            template["protocol_sha256"] = config.get("protocol_sha256")
            previous = successful_prior(target / "records.jsonl", model=key, case_id=case["case_id"], seed=case["seed"], mode=template["mode"],
                                        config_sha=template["config_sha256"], input_sha=template["input_sha256"], protocol_sha=template["protocol_sha256"], root=root)
            if previous:
                records.append(previous)
                resumed += 1
            else:
                pending.append(case)
                templates[(case["case_id"], case["seed"])] = template
        request["cases"] = pending
        if not pending:
            continue
        response = invoke_worker(Path(__file__).with_name("worker.py"), request, target / run_id / key / "diagnostics",
                                 options.get("python", sys.executable), float(eval_config.get("timeout_s", 7200)), working_dir=root)
        lookup = {(r["case_id"], r["seed"]): r for r in response.get("results", [])}
        for case in pending:
            prompt = next(p for p in prompts if p["prompt_id"] == case["case_id"])
            record = templates[(case["case_id"], case["seed"])]
            result = lookup.get((case["case_id"], case["seed"]))
            record.update(result or {"status": "blocked", "error": response.get("error", "Worker produced no case result")})
            if not result and response.get("diagnostics"):
                record["diagnostics"] = response["diagnostics"]
            record["prompt"] = prompt["prompt"]
            record["execution"] = {k: response.get(k) for k in ("python", "device", "compute_capability", "torch_version", "cuda_version", "libraries", "source_revision", "source_dirty", "resolved_checkpoint_revision", "runtime_precision")}
            if response.get("resolved_checkpoint_revision"):
                record["checkpoint_revision"] = response["resolved_checkpoint_revision"]
            record["mandatory_criteria"] = prompt.get("mandatory_criteria", [])
            record["hypothetical_business_input"] = bool(prompt.get("hypothetical_business_input", True))
            record["generation_config"] = {"width": request["width"], "height": request["height"], "num_inference_steps": 4,
                                           "warmup_runs": request["warmup_runs"],
                                           "guidance_scale": 0 if key == "sdxl_lightning" else 1, "precision": "fp16" if key == "sdxl_lightning" else "official_scaled_fp8"}
            record["artifacts"]["frozen_prompts"] = frozen_prompts.relative_to(root).as_posix()
            record["artifacts"] = {name: resolve(root, value).relative_to(root).as_posix() for name, value in record["artifacts"].items()}
            if record["status"] == "success":
                try:
                    from PIL import Image
                    with Image.open(resolve(root, record["artifacts"]["image"])) as image:
                        image.verify()
                    record["metrics"]["valid_output"] = 1
                except Exception as exc:
                    record.update(status="failed", error=f"Invalid output image: {exc}")
                    record["metrics"]["valid_output"] = 0
            # A failed export may still leave an invalid file worth diagnosing.
            # Retain its actual hash without treating it as a valid image.
            record["artifacts"] = {name: value for name, value in record["artifacts"].items() if resolve(root, value).is_file()}
            record["artifact_sha256"] = {name: file_digest(resolve(root, value)) for name, value in record["artifacts"].items()}
            append(target / "records.jsonl", record)
            records.append(record)
    review = export_review(records, target / run_id / "review", seed, root=root)
    return {"pair": "generation", "run_id": run_id, "records": len(records), "resumed": resumed, "success": sum(r["status"] == "success" for r in records),
            "blocked": sum(r["status"] == "blocked" for r in records), "failed": sum(r["status"] == "failed" for r in records), "review": review}
