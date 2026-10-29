import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from stylesphere.common.io import sha256_file, utc_now, write_json
from .verification import verify

ALLOWED = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".ps1", ".cmd", ".json", ".jsonl", ".csv", ".tsv", ".png", ".jpg", ".jpeg", ".svg", ".html", ".npz", ".glb", ".obj", ".mtl", ".ply", ".log", ".patch", ".config", ".sdf", ".pbtxt", ".pdf"}
EXCLUDED = {"__pycache__", ".git", ".pytest_cache", ".cache", "downloads", "vendor", "models", "submission"}


def approved_dataset_assets(root):
    """Whitelist declared redistributable assets and explicitly attributed ABO originals.

    A reconstruction source photo requires its object ID in the rights record,
    corresponding prepared row, local preparation manifest and matching raw hash.
    QA-rejected photos never enter the whitelist merely by being in assets/.
    """
    root = Path(root).resolve()
    from .reference_packaging import approved_reference_assets
    approved = approved_reference_assets(root)
    def checked(value, expected=None):
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.is_relative_to(root / "datasets"):
            raise ValueError("Dataset redistribution path must stay inside the project's datasets directory")
        if path.is_file() and expected and sha256_file(path) != expected:
            raise ValueError(f"Redistributable dataset asset hash differs: {path.name}")
        return path
    for catalogue in (root / "datasets").rglob("*.csv"):
        if catalogue.is_relative_to(root / "datasets/reconstruction_reference"):
            continue  # Separately validated source-specific GSO rights and asset hashes above.
        rights_path = catalogue.parent / "RIGHTS_AND_ATTRIBUTION.json"
        rights = json.loads(rights_path.read_text(encoding="utf-8")) if rights_path.exists() and catalogue.name == "objects.csv" else None
        if rights:
            if not rights.get("redistributable") or rights.get("licence") != "CC-BY-4.0" or not rights.get("asset_attribution") or not rights.get("dataset_attribution"):
                raise ValueError("Reconstruction rights record lacks explicit permission/licence/attribution")
            licence = catalogue.parent / "SOURCE_LICENSE_CC_BY_4_0.txt"
            if not licence.is_file() or sha256_file(licence) != rights.get("licence_sha256"):
                raise ValueError("Reconstruction redistribution licence is missing or changed")
        with catalogue.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("redistributable", "false")).lower() not in {"true", "1"}:
                    continue
                if rights and row.get("object_id") not in rights.get("objects", []):
                    raise ValueError("A reconstruction row is not authorised by the explicit rights record")
                for key, hash_key in (("image_path", "image_sha256"), ("source_image", "image_sha256"),
                                      ("reference_mesh", "mesh_sha256"), ("reference_mesh_path", "mesh_sha256")):
                    if row.get(key):
                        approved.add(checked(row[key], row.get(hash_key)))
                if not rights:
                    continue
                if not row.get("preparation_metadata"):
                    raise ValueError("Original-photo redistribution requires the corresponding preparation manifest")
                preparation_path = checked(row["preparation_metadata"])
                preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
                if row["object_id"] != "abo-" + preparation.get("source_id", ""):
                    raise ValueError("Preparation manifest does not identify the authorised ABO object")
                source = urlsplit(preparation.get("source_url", ""))
                if source.scheme != "https" or source.netloc != "amazon-berkeley-objects.s3.amazonaws.com" or not source.path.startswith("/images/original/") or source.query or source.fragment:
                    raise ValueError("Original-photo redistribution requires verified official ABO S3 provenance")
                if preparation.get("prepared_image_sha256") != row.get("image_sha256"):
                    raise ValueError("Prepared-photo hash does not match its authorised reconstruction row")
                original = checked(preparation["original_image"], preparation["original_sha256"])
                if not original.is_file():
                    raise ValueError("Authorised original reconstruction photo is missing; restore before packaging")
                approved.add(original)
    return approved


def package(root, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    checks = verify(root, output_dir)
    if not checks["integrity_passed"]:
        raise ValueError("Packaging stopped: evidence integrity errors. Run verify and resolve errors.")
    complete = (root / "results/final_evaluation/completion.json").is_file()
    if complete:
        from .completion import validate_completion_receipt
        from .final_report import validate_report_snapshot
        completion_errors = validate_completion_receipt(root)
        completion_errors.extend(validate_report_snapshot(root))
        if completion_errors:
            raise ValueError("Packaging stopped: final completion evidence is invalid: " + "; ".join(completion_errors))
    files = []
    for directory in ("src", "config", "scripts", "datasets", "docs", "tests", "environment", "figures"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file())
    files.extend(p for p in root.iterdir() if p.is_file() and (p.name in {"README.md", "AGENTS.md", "pyproject.toml", ".gitignore"} or p.name.startswith("requirements-")))
    files.extend(p for p in output_dir.rglob("*") if p.is_file())
    entries = {}
    approved_assets = approved_dataset_assets(root)
    omitted_assets = []
    for path in sorted(set(files)):
        if path.name == "package-command.log":
            continue  # Active CLI transcript cannot have a stable hash while packaging.
        if path.is_symlink() or path.suffix.lower() not in ALLOWED and path.name != ".gitignore":
            continue
        parts = path.relative_to(root).parts if path.is_relative_to(root) else path.parts
        if any(part in EXCLUDED or part.startswith(".venv") for part in parts) or path.name.startswith(".env"):
            continue
        if path.is_relative_to(root / "datasets") and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".glb", ".npz", ".svg", ".obj", ".mtl", ".ply", ".config", ".sdf", ".pbtxt"} and path.resolve() not in approved_assets:
            omitted_assets.append({"path": str(path.relative_to(root)), "reason": "Redistribution permission not explicitly granted in dataset manifest; reacquire with provenance instructions"})
            continue
        if path.stat().st_size > 100_000_000:
            raise ValueError(f"Large artifact needs redistribution review: {path.name}")
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else "results/" + path.relative_to(output_dir).as_posix()
        entries[name] = path
    status = {"timestamp_utc": utc_now(),
              "label": "COMPLETE_REQUESTED_COMPONENT" if complete else "PARTIAL_RESEARCH_PACKAGE",
              "requested_component_complete": complete, "benchmark_complete": False,
              "ai_assisted": True,
              "scope": "User-confirmed four model pairs/eight models, Evaluation Codes and Data. User expressly permits AI assistance.",
              "completion_qualification": "AI quality opinions/proxies and normalized reference-assisted shape scores remain distinct from independent human feedback and absolute physical accuracy.",
              "pending": [] if complete else checks["pending"],
              "legacy_verifier_findings": checks["pending"],
              "legacy_benchmark_complete_reason": "Original primary protocol requires independent human quality outcomes and eligible primary absolute/aligned geometry; those are not replaced by supplementary AI/normalized scores.",
              "completion_receipt_sha256": sha256_file(root / "results/final_evaluation/completion.json") if complete else None,
              "omitted_assets": omitted_assets, "files": {}}
    target = root / "submission"
    target.mkdir(exist_ok=True)
    zip_path = target / ("StyleSphere_Evaluation.zip" if complete else "StyleSphere_Evaluation_PARTIAL.zip")
    temporary = zip_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in entries.items():
                # Hash, credential-check and archive exactly the same byte snapshot.
                # Download logs may advance while a partial package is built.
                data = path.read_bytes()
                if len(data) > 100_000_000:
                    raise ValueError(f"Large artifact needs redistribution review: {path.name}")
                if path.suffix.lower() in {".txt", ".json", ".jsonl", ".md", ".yaml", ".yml", ".py", ".ps1", ".cmd", ".toml", ".csv", ".tsv", ".log", ".html", ".svg"}:
                    if re.search(r"hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|AKIA[A-Z0-9]{16}", data.decode("utf-8-sig", errors="replace")):
                        raise ValueError(f"Possible credential detected in {path.name}; inspect before packaging")
                status["files"][name] = hashlib.sha256(data).hexdigest()
                archive.writestr("StyleSphere_Evaluation/" + name, data)
            manifest_data = (json.dumps(status, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            archive.writestr("StyleSphere_Evaluation/SUBMISSION_MANIFEST.json", manifest_data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip():
                raise ValueError("ZIP integrity check failed")
        temporary.replace(zip_path)
        write_json(target / "SUBMISSION_MANIFEST.json", status)
    finally:
        temporary.unlink(missing_ok=True)
    return {"zip": str(zip_path), "files": len(entries) + 1, "size_bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "label": status["label"], "requested_component_complete": complete}
