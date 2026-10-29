"""Exact generation inputs, retained separately from immutable inference records."""
from __future__ import annotations

import json
from pathlib import Path

from stylesphere.common.io import canonical_hash, portable_path, read_jsonl, sha256_file, utc_now, write_json


def freeze_prompts(root: Path, source: Path, run_directory: Path, expected_sha256: str | None = None) -> Path:
    """Copy exact source bytes; never silently replace an existing frozen input."""
    root, source, run_directory = Path(root).resolve(), Path(source), Path(run_directory)
    actual = sha256_file(source)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError("Prompt source differs from the pre-run frozen dataset SHA256")
    target = run_directory / "evidence" / "prompts.jsonl"
    if not target.resolve().is_relative_to(root):
        raise ValueError("Frozen prompts must remain inside the project root")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) != actual:
        raise ValueError("Refuse to replace previously frozen generation inputs")
    if not target.exists():
        target.write_bytes(source.read_bytes())
    return target


def _compare_prompt(record: dict, frozen: Path) -> list[str]:
    rows = read_jsonl(frozen)
    matches = [row for row in rows if row.get("prompt_id") == record.get("case_id")]
    if len(matches) != 1:
        return ["Frozen prompts require exactly one matching case ID"]
    prompt = matches[0]
    errors = []
    if canonical_hash(prompt) != record.get("input_sha256"):
        errors.append("Frozen full prompt row does not match record input_sha256")
    if prompt.get("prompt") != record.get("prompt"):
        errors.append("Recorded prompt differs from frozen prompt")
    if prompt.get("mandatory_criteria", []) != record.get("mandatory_criteria", []):
        errors.append("Recorded mandatory criteria differ from frozen prompt")
    if bool(prompt.get("hypothetical_business_input", True)) != record.get("hypothetical_business_input", True):
        errors.append("Recorded hypothetical-input declaration differs from frozen prompt")
    return errors


def bind_existing_run(root: Path, pair_dir: Path, run_id: str, source: Path, *, expected_sha256: str,
                      deployment_freeze: Path | None = None) -> Path:
    """Audit an earlier run without rewriting any raw inference records."""
    root, pair_dir = Path(root).resolve(), Path(pair_dir)
    records = [row for row in read_jsonl(pair_dir / "records.jsonl") if row.get("run_id") == run_id]
    if not records:
        raise ValueError("No original inference records exist for the requested run")
    frozen = freeze_prompts(root, source, pair_dir / run_id, expected_sha256)
    for record in records:
        errors = _compare_prompt(record, frozen)
        if errors:
            raise ValueError("Cannot audit-bind an inconsistent record: " + "; ".join(errors))
    manifest = {"schema_version": 1, "kind": "post_run_exact_input_binding", "run_id": run_id,
                "frozen_prompts": frozen.relative_to(root).as_posix(), "frozen_prompts_sha256": sha256_file(frozen),
                "record_bindings": [{"record_sha256": canonical_hash(row), "model": row["model"], "case_id": row["case_id"],
                                     "seed": row["seed"], "protocol_sha256": row.get("protocol_sha256"),
                                     "input_sha256": row["input_sha256"]} for row in records]}
    if deployment_freeze is not None:
        manifest["deployment_freeze"] = Path(deployment_freeze).resolve().relative_to(root).as_posix()
        manifest["deployment_freeze_sha256"] = sha256_file(deployment_freeze)
    target = frozen.parent / "prompt_binding.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if {k: v for k, v in existing.items() if k != "audited_at_utc"} != manifest:
            raise ValueError("Existing generation audit binding differs; refuse replacement")
        return target
    write_json(target, {**manifest, "audited_at_utc": utc_now()})
    return target


def verify_prompt_binding(record: dict, root: Path, pair_dir: Path | None = None) -> list[str]:
    """Check new direct artifacts or the audited sidecar for earlier raw records."""
    root = Path(root).resolve()
    try:
        frozen_value = record.get("artifacts", {}).get("frozen_prompts")
        expected = record.get("artifact_sha256", {}).get("frozen_prompts")
        if frozen_value is None:
            pair_dir = Path(pair_dir) if pair_dir is not None else root / "results" / "generation"
            sidecar = pair_dir / record["run_id"] / "evidence" / "prompt_binding.json"
            if not sidecar.is_file():
                return ["No frozen prompts artifact or audited prompt binding"]
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            bindings = [row for row in manifest["record_bindings"] if row["record_sha256"] == canonical_hash(record)]
            if manifest.get("run_id") != record["run_id"] or len(bindings) != 1:
                return ["Original inference record does not match its audited prompt binding"]
            frozen_value, expected = manifest["frozen_prompts"], manifest["frozen_prompts_sha256"]
            if manifest.get("deployment_freeze"):
                freeze = portable_path(root, manifest["deployment_freeze"])
                if sha256_file(freeze) != manifest.get("deployment_freeze_sha256"):
                    return ["Deployment freeze changed after prompt audit binding"]
                if json.loads(freeze.read_text(encoding="utf-8"))["dataset_sha256"] != expected:
                    return ["Frozen prompts do not match the pre-run deployment freeze"]
        frozen = portable_path(root, frozen_value).resolve()
        if not frozen.is_relative_to(root) or not expected or sha256_file(frozen) != expected:
            return ["Frozen prompts file is absent, outside the project, or has a mismatched SHA256"]
        return _compare_prompt(record, frozen)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return [f"Invalid generation prompt evidence: {type(exc).__name__}: {exc}"]
