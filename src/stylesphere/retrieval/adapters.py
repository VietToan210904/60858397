"""Model-native encoders. Imports and checkpoint access happen only at construction."""

from __future__ import annotations

import json
from pathlib import Path


MODEL_IDS = {"fashion_siglip": "Marqo/marqo-fashionSigLIP", "siglip2": "google/siglip2-base-patch16-224"}


def local_snapshot(config: dict) -> Path:
    """Never downloads weights. Acquisition is a separate, explicitly approved operation."""
    if config.get("local_path"):
        path = Path(config["local_path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Configured model directory is missing: {path}")
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=config["model_id"], revision=config.get("revision", "main"), local_files_only=True))


class Encoder:
    def __init__(self, key: str, config: dict, device: str = "cuda:0"):
        import torch
        self.torch = torch
        self.device = device
        self.key = key
        config = {**config, "model_id": MODEL_IDS[key]}
        self.snapshot = local_snapshot(config)
        self.revision = self.snapshot.name if len(self.snapshot.name) == 40 else config.get("revision")
        precision = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(config.get("precision"), config.get("precision", "fp16"))
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Retrieval precision must be fp32, fp16, or bf16")
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        if key == "fashion_siglip":
            import open_clip
            # OpenCLIP loads the official architecture, native transforms and weights.
            # Its local-dir schema avoids executing the repository's Python wrapper.
            factory = __import__("open_clip.factory", fromlist=["parse_model_name"])
            if not hasattr(factory, "parse_model_name"):
                raise RuntimeError("Upgrade open_clip_torch: this adapter requires the local-dir model schema")
            name = f"local-dir:{self.snapshot.resolve()}"
            self.model, _, self.processor = open_clip.create_model_and_transforms(name, precision=precision, device=device, require_pretrained=True)
            oc = json.loads((self.snapshot / "open_clip_config.json").read_text(encoding="utf-8"))
            token_repo = oc["model_cfg"]["text_cfg"]["hf_tokenizer_name"]
            token_path = local_snapshot({"model_id": token_repo, "revision": config.get("tokenizer_revision", "main"), "local_path": config.get("tokenizer_local_path")})
            from open_clip.tokenizer import HFTokenizer
            self.tokenizer = HFTokenizer(str(token_path), context_length=64, clean="canonicalize")
            self.tokenizer_revision = token_path.name if len(token_path.name) == 40 else config.get("tokenizer_revision")
        else:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.snapshot, local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(self.snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=self.dtype).to(device)
        self.model.eval()

    def prepare_images(self, paths: list[str]):
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if self.key == "fashion_siglip":
            return self.torch.stack([self.processor(image) for image in images]).to(self.device, dtype=self.dtype)
        batch = self.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in batch.items()}

    def prepare_text(self, texts: list[str]):
        if self.key == "fashion_siglip":
            return self.tokenizer(texts).to(self.device)
        batch = self.processor(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def encode_prepared(self, inputs, modality: str):
        with self.torch.inference_mode():
            if self.key == "fashion_siglip":
                features = self.model.encode_image(inputs) if modality == "image" else self.model.encode_text(inputs)
            else:
                features = self.model.get_image_features(**inputs) if modality == "image" else self.model.get_text_features(**inputs)
            # Recent Transformers may return a pooled model output instead of a tensor.
            if not self.torch.is_tensor(features):
                features = features.pooler_output
            features = self.torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        return features.detach().cpu().numpy()

    def close(self):
        self.model = None
        self.torch.cuda.empty_cache()
