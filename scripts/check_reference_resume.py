"""Prove same-deployment successful case resume appends no raw records."""
from datetime import datetime, timezone
import json
from pathlib import Path

from stylesphere.common.io import sha256_file
from stylesphere.reconstruction.reference_benchmark import lines, run


def main():
    root = Path.cwd()
    path = root / "results/reference_test/reconstruction/records.jsonl"
    before_hash, before_count = sha256_file(path), len(lines(path))
    diagnostics_before = {p.relative_to(root).as_posix() for p in (root / "results/reference_test/reconstruction").rglob("request.json")}
    results = {}
    for model in ("sf3d", "spar3d"):
        command = run(root, "test", model)
        response = json.loads((root / command["log"]).read_text())
        if len(response) != 1 or response[0]["resumed"] != 20 or response[0]["records"] != 20 or response[0]["success"] != 20:
            raise ValueError("Expected20successful matched resumes per model")
        results[model] = response[0]
    after_hash, after_count = sha256_file(path), len(lines(path))
    diagnostics_after = {p.relative_to(root).as_posix() for p in (root / "results/reference_test/reconstruction").rglob("request.json")}
    if before_hash != after_hash or before_count != 40 or after_count != 40 or diagnostics_before != diagnostics_after:
        raise ValueError("Resume altered raw evidence or invoked new worker cases")
    value = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "integrity_passed": True,
             "raw_records_before": before_count, "raw_records_after": after_count,
             "raw_records_sha256_before": before_hash, "raw_records_sha256_after": after_hash,
             "new_worker_requests": len(diagnostics_after - diagnostics_before), "models": results}
    (root / "environment/reference_resume_audit.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
