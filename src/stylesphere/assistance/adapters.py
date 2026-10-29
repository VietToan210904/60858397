"""Native local Transformers generation; no hosted API and no automatic downloads."""

from __future__ import annotations

from pathlib import Path

MODEL_IDS = {"phi4_mini": "microsoft/Phi-4-mini-instruct", "qwen35_4b": "Qwen/Qwen3.5-4B"}


class ChatModel:
    def __init__(self, key: str, config: dict):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.key = key
        self.config = config
        if config.get("local_path"):
            snapshot = Path(config["local_path"])
            if not snapshot.is_dir():
                raise FileNotFoundError(f"Configured model directory is missing: {snapshot}")
        else:
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(MODEL_IDS[key], revision=config.get("revision", "main"), local_files_only=True))
        self.revision = snapshot.name if len(snapshot.name) == 40 else config.get("revision")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        precision = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(config.get("precision"), config.get("precision", "fp16"))
        quantization = config.get("quantization", "none")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Assistance precision must be fp32, fp16, or bf16")
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        kwargs = {"local_files_only": True, "trust_remote_code": False, "torch_dtype": dtype,
                  "device_map": {"": "cuda:0"}, "attn_implementation": config.get("attn_implementation", "sdpa")}
        if quantization in {"nf4", "int4", "4bit"}:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        elif quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization not in {"none", None}:
            raise ValueError("Quantization must be none, nf4 or int8; no checkpoint substitution is permitted")
        if key == "qwen35_4b":
            if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
                raise RuntimeError("Installed Transformers lacks Qwen3.5; install a release with Qwen3_5ForConditionalGeneration (see official model card)")
            model_class = transformers.Qwen3_5ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
        self.model = model_class.from_pretrained(snapshot, **kwargs).eval()
        dtype_counts, tensor_counts, device_counts = {}, {}, {}
        for parameter in self.model.parameters():
            name = str(parameter.dtype)
            dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
            tensor_counts[name] = tensor_counts.get(name, 0) + 1
            device = str(parameter.device)
            device_counts[device] = device_counts.get(device, 0) + parameter.numel()
        self.metadata = {"model_class": type(self.model).__name__, "text_only_inputs": True,
                         "vision_tower_loaded": key == "qwen35_4b", "precision": precision,
                         "quantization": quantization, "checkpoint_revision": self.revision,
                         "transformers_version": transformers.__version__,
                         "enable_thinking": bool(config.get("enable_thinking", False)) if key == "qwen35_4b" else None,
                         "decoding": "greedy", "local_files_only": True}
        self.metadata.update(parameter_count=sum(dtype_counts.values()), parameter_dtype_counts=dtype_counts,
                             parameter_tensor_counts=tensor_counts,
                             parameter_device_counts=device_counts,
                             hf_device_map={str(name): str(device) for name, device in getattr(self.model, "hf_device_map", {}).items()})
        self.metadata.update(max_new_tokens=int(config.get("max_new_tokens", 512)), max_input_tokens=int(config.get("max_input_tokens", 8192)))
        self.last_timings = {}

    def generate(self, messages: list[dict]) -> str:
        from stylesphere.common.runtime import Measurement
        with Measurement() as preprocessing:
            template_kwargs = {"enable_thinking": bool(self.config.get("enable_thinking", False))} if self.key == "qwen35_4b" else {}
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[-1] > int(self.config.get("max_input_tokens", 8192)):
                raise ValueError("Conversation exceeds max_input_tokens; refusing silent context truncation")
            inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
        with Measurement() as inference:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=int(self.config.get("max_new_tokens", 512)),
                                             do_sample=False, use_cache=True,
                                             pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        generated = output[0, inputs["input_ids"].shape[-1]:]
        self.last_timings = {"preprocessing_s": preprocessing.elapsed_s, "inference_s": inference.elapsed_s,
                             "peak_allocated_mb": inference.peak_allocated_mb, "peak_reserved_mb": inference.peak_reserved_mb,
                             "input_tokens": int(inputs["input_ids"].shape[-1]), "generated_tokens": int(generated.shape[-1])}
        # Preserve <think> delimiters even if the tokenizer marks them special.
        # Remove only known generation terminators/padding, not all special tokens.
        terminators = set()
        for value in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.model.generation_config.eos_token_id):
            if isinstance(value, (list, tuple)):
                terminators.update(value)
            elif value is not None:
                terminators.add(value)
        visible_ids = [value for value in generated.tolist() if value not in terminators]
        return self.tokenizer.decode(visible_ids, skip_special_tokens=False)

    def close(self):
        self.model = None
        self.torch.cuda.empty_cache()
