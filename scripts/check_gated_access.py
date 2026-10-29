"""Check accepted HF gates using the locally stored token, without downloading weights.

Run `hf auth login --force` interactively in your own terminal first. This script
does not accept tokens as arguments, print identity, or accept licence terms.
"""
from pathlib import Path
import json
import sys

from huggingface_hub import get_hf_file_metadata, hf_hub_url


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from stylesphere.common.config import load_config
    from stylesphere.common.io import utc_now, write_json
    config = load_config(root)
    checks = []
    for key in ("spar3d", "sf3d"):
        model = config["models"][key]
        row = {"model": key, "model_id": model["model_id"], "revision": model["revision"]}
        try:
            metadata = get_hf_file_metadata(hf_hub_url(model["model_id"], "config.yaml", revision=model["revision"]),
                                            token=True, timeout=20)
            row.update(access_granted=True, resolved_revision=metadata.commit_hash)
        except Exception as exc:
            # Do not retain exception text, headers, token values or signed URLs.
            response = getattr(exc, "response", None)
            row.update(access_granted=False, error_type=type(exc).__name__,
                       http_status=getattr(response, "status_code", None))
        checks.append(row)
        print(json.dumps(row))
    write_json(root / "environment/hf-gated-access.json", {"timestamp_utc": utc_now(), "checks": checks})
    return 0 if all(row["access_granted"] for row in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
