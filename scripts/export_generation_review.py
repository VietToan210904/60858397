"""Create the canonical blinded review only from both models' actual benchmarks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylesphere.generation.canonical_review import export_canonical_review


def main():
    parser = argparse.ArgumentParser(description="Export one source-hash-bound blinded human worksheet; refuses to overwrite human work.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=8410)
    parser.add_argument("--protocol-sha256", help="Select an explicit matching generation protocol if several exist")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(export_canonical_review(root, output, args.seed, args.protocol_sha256), indent=2))


if __name__ == "__main__":
    main()
