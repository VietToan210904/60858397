from pathlib import Path
import yaml

PAIRS = ("retrieval", "generation", "assistance", "reconstruction")
MODEL_IDS = {"fashion_siglip": "Marqo/marqo-fashionSigLIP", "siglip2": "google/siglip2-base-patch16-224", "sdxl_lightning": "ByteDance/SDXL-Lightning", "flux2_klein": "black-forest-labs/FLUX.2-klein-4b-fp8", "phi4_mini": "microsoft/Phi-4-mini-instruct", "qwen35_4b": "Qwen/Qwen3.5-4B", "spar3d": "stabilityai/stable-point-aware-3d", "sf3d": "stabilityai/stable-fast-3d"}


def dataset_fingerprints(root, config):
    from .io import sha256_file
    result = {}
    for pair in PAIRS:
        dataset = Path(root) / config.get("evaluation", {}).get(pair, {}).get("dataset_dir", f"datasets/{pair}")
        result[pair] = {p.name: sha256_file(p) for p in sorted(dataset.glob("*")) if p.is_file() and p.suffix in {".csv", ".json", ".jsonl"} and p.name != "manifest.json"}
    return result


def pair_protocol_sha256(config, pair, fingerprints):
    from .io import canonical_hash
    return canonical_hash({"models": {k: v for k, v in config["models"].items() if v["pair"] == pair},
                           "evaluation": config.get("evaluation", {}).get(pair, {}),
                           "warmup_runs": config.get("evaluation", {}).get("warmup_runs", 1),
                           "dataset": fingerprints[pair], "seed": config["seed"]})


def merge(base, extra):
    result = dict(base)
    for key, value in extra.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(root, path=None):
    root = Path(root).resolve()
    result = {}
    for filename in ("models.yaml", "evaluation.yaml", "hardware.yaml", "dataset_sources.yaml", "local.yaml"):
        file = root / "config" / filename
        if file.exists():
            result = merge(result, yaml.safe_load(file.read_text(encoding="utf-8")) or {})
    if path:
        file = Path(path)
        if not file.is_absolute():
            file = root / file
        result = merge(result, yaml.safe_load(file.read_text(encoding="utf-8")) or {})
    validate_config(result)
    return result


def validate_config(config):
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    models = config.get("models")
    if not isinstance(models, dict) or len(models) != 8:
        raise ValueError("Exactly eight identifiable models must be configured")
    if len({x.get("model_id") for x in models.values()}) != 8:
        raise ValueError("Model identities must be distinct")
    for key, model in models.items():
        if MODEL_IDS.get(key) != model.get("model_id"):
            raise ValueError(f"Model substitution is prohibited: {key}")
        if model.get("pair") not in PAIRS or not model.get("revision") or not model.get("licence"):
            raise ValueError(f"Invalid model configuration: {key}")
        if model.get("trust_remote_code", False):
            raise ValueError("Remote code requires a reviewed, explicit integration; disabled here")
    for pair in PAIRS:
        if sum(m["pair"] == pair for m in models.values()) != 2:
            raise ValueError(f"Expected two models for {pair}")
