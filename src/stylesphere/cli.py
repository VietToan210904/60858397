"""Run `python -m stylesphere.cli --help` for the evidence-producing commands."""
import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path
from .common.config import PAIRS, dataset_fingerprints, load_config, pair_protocol_sha256
from .common.io import append_jsonl, canonical_hash, scrub, sha256_file, utc_now, write_json
from .common.runtime import doctor, seed_everything


def parser():
    p = argparse.ArgumentParser(description="StyleSphere: reproducible local CUDA evaluation. Blocked/incomplete work exits 2; integrity errors exit 1.")
    commands = p.add_subparsers(dest="command", required=True)
    descriptions = {"doctor": "Record actual Python, dependencies, GPU and disk diagnostics", "prepare": "Prepare versioned hypothetical inputs and local dataset manifests; never invent observations", "smoke": "Run the smallest real local inference for the selected pair; not benchmark evidence", "evaluate": "Evaluate real cases sequentially, resuming successful matching cases", "report": "Regenerate tables and available charts from observed records", "verify": "Check evidence integrity and report unfinished work", "package": "Build a verified ZIP clearly labelled partial when evaluations are unfinished", "acquire": "Estimate and explicitly download selected retrieval checkpoints (inference stays offline)"}
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description)
        sub.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory (default: current directory)")
        sub.add_argument("--config", type=Path, help="YAML overrides merged with project configuration")
        sub.add_argument("--output-dir", type=Path, help="Result directory; relative to project root")
        sub.add_argument("--seed", type=int, help="Reproducible non-negative seed (default config.seed)")
        if name in {"prepare", "smoke", "evaluate"}:
            group = sub.add_mutually_exclusive_group(required=True)
            group.add_argument("--pair", choices=PAIRS)
            group.add_argument("--all", action="store_true", help="Process all pairs sequentially")
        if name in {"smoke", "evaluate"}:
            sub.add_argument("--model", action="append", help="Model key within chosen pair; repeat for multiple keys")
        if name == "acquire":
            sub.add_argument("--model", choices=("fashion_siglip", "siglip2", "fashion_tokenizer", "phi4_mini", "qwen35_4b", "spar3d", "sf3d"), required=True)
            sub.add_argument("--dry-run", action="store_true", help="Report exact checkpoint sizes and disk space without downloading")
            sub.add_argument("--approve-large-download", action="store_true", help="Use only after explicit user approval for >5 GB")
            sub.add_argument("--ranged", action="store_true", help="Use bounded resumable HTTP ranges with official checksum validation")
    return p


def execute(args):
    root = args.root.resolve()
    config = load_config(root, args.config)
    seed = config["seed"] if args.seed is None else args.seed
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    config["seed"] = seed
    config["dataset_fingerprints"] = dataset_fingerprints(root, config)
    config["protocol_sha256"] = canonical_hash(config)
    output = args.output_dir or Path(config.get("output_dir", "results"))
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "doctor":
        result = doctor(root)
        return result, 0
    if args.command == "acquire":
        from .common.acquisition import acquire
        return acquire(root, config, args.model, approve_large=args.approve_large_download, dry_run=args.dry_run, ranged=args.ranged), 0
    if args.command == "prepare":
        from .datasets import prepare
        return [prepare(root, pair, config, seed) for pair in PAIRS if args.all or pair == args.pair], 0
    if args.command in {"smoke", "evaluate"}:
        if args.command == "evaluate" and config.get("smoke_only"):
            raise ValueError("This configuration is explicitly smoke_only; it cannot produce final benchmark records")
        results = []
        seed_everything(seed)
        for pair in PAIRS:
            if not args.all and pair != args.pair:
                continue
            keys = args.model
            if keys and any(k not in config["models"] or config["models"][k]["pair"] != pair for k in keys):
                raise ValueError("Each selected model must belong to the selected pair")
            module = importlib.import_module(f"stylesphere.{pair}.runner")
            pair_config = dict(config)
            pair_config["protocol_sha256"] = pair_protocol_sha256(config, pair, config["dataset_fingerprints"])
            results.append(module.run(root, pair_config, output, seed, smoke=args.command == "smoke", model_keys=keys))
        write_json(output / f"last_{args.command}.json", {"timestamp_utc": utc_now(), "results": results})
        return results, 2 if any(r.get("blocked", 0) or r.get("failed", 0) for r in results) else 0
    if args.command == "report":
        from .reporting.reports import report
        return report(root, output, config), 0
    if args.command == "verify":
        from .reporting.verification import verify
        result = verify(root, output)
        return result, 0 if result["integrity_passed"] else 1
    if args.command == "package":
        from .reporting.packaging import package
        return package(root, output), 0
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result, code = execute(args)
        print(json.dumps(scrub(result), indent=2, default=str))
        return code
    except Exception as exc:
        failure = {"timestamp_utc": utc_now(), "command": args.command, "type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        append_jsonl(args.root / "environment" / "command_failures.jsonl", failure)
        print(json.dumps(scrub(failure), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
