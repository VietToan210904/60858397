import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def scrub(value):
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if re.fullmatch(r"(?i)(hf_token|access_token|api_key|password|secret|authorization|token)", str(k)) else scrub(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"hf_[A-Za-z0-9]{10,}|Bearer\s+\S+", "[REDACTED]", value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(scrub(value), indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(scrub(value), ensure_ascii=False, allow_nan=False, default=str) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{line_number}") from exc
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def portable_path(root, value):
    """Resolve retained paths, including audited original-workspace aliases.

    Historical raw records remain immutable. A packaged origin map allows their
    original absolute Windows paths to refer to the extracted artifact copies.
    """
    root = Path(root)
    normalized = str(value).replace("\\", "/")
    origins = root / "environment" / "path_origins.json"
    if origins.is_file():
        for origin in json.loads(origins.read_text(encoding="utf-8"))["original_project_roots"]:
            origin = str(origin).replace("\\", "/").rstrip("/")
            if normalized.casefold().startswith(origin.casefold() + "/"):
                return root / normalized[len(origin) + 1:]
    path = Path(normalized)
    return path if path.is_absolute() else Path(root) / path
