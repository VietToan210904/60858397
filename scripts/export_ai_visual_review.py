"""Export/validate/summarize supplementary AI visual evidence without model calls."""
import argparse
import json
from pathlib import Path

from stylesphere.common.io import write_json
from stylesphere.reporting.ai_visual import aggregate, export_packets, validate_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "validate", "summarize"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "results/ai_visual_assessment"
    if args.action == "export":
        result = export_packets(root, root / "results", destination)
    elif args.action == "validate":
        result = validate_export(destination)
    else:
        result = aggregate(destination)
        write_json(destination / "summary.json", result)
    print(json.dumps({key:result[key] for key in ("kind","protocol_sha256","selection_sha256") if key in result},indent=2))


if __name__ == "__main__":
    main()
