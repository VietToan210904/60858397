"""Deterministic post-run QA thumbnails, never human evaluation scores."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stylesphere.common.io import portable_path, read_jsonl, sha256_file, utc_now, write_json
from stylesphere.generation.provenance import verify_prompt_binding


def main():
    from PIL import Image, ImageDraw, ImageFont
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--model", required=True, choices=("sdxl_lightning", "flux2_klein"))
    parser.add_argument("--seed", type=int, default=8410)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    pair = root / "results" / "generation"
    records = [row for row in read_jsonl(pair / "records.jsonl") if row.get("run_id") == args.run_id
               and row.get("model") == args.model and row.get("mode") == "evaluation" and row.get("status") == "success"
               and row.get("seed") == args.seed]
    if not records:
        raise ValueError("No successful evaluation records match the requested run/model/seed")
    first = records[0]
    frozen = first.get("artifacts", {}).get("frozen_prompts")
    if frozen is None:
        binding = json.loads((pair / args.run_id / "evidence" / "prompt_binding.json").read_text(encoding="utf-8"))
        frozen = binding["frozen_prompts"]
    prompts = read_jsonl(portable_path(root, frozen))
    chosen = {}
    for prompt in sorted(prompts, key=lambda row: row["prompt_id"]):
        chosen.setdefault(prompt["category"], prompt)
    lookup = {row["case_id"]: row for row in records}
    selected = [lookup[prompt["prompt_id"]] for prompt in chosen.values()]
    for row in selected:
        errors = verify_prompt_binding(row, root, pair)
        if errors:
            raise ValueError("; ".join(errors))
        if sha256_file(portable_path(root, row["artifacts"]["image"])) != row["artifact_sha256"]["image"]:
            raise ValueError("Selected image SHA256 changed")
    font = ImageFont.load_default(size=17)
    columns, tile, gap, header, label = 2, 512, 12, 86, 50
    canvas = Image.new("RGB", (columns * tile + 3 * gap, math.ceil(len(selected) / columns) * (tile + label + gap) + header + gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), f"{args.model} | seed {args.seed} | deterministic QA selection", fill="black", font=font)
    draw.text((gap, 38), "Post-run AI-assisted inspection; no human scores or population-quality claim", fill="black", font=font)
    bindings = []
    prompt_lookup = {row["prompt_id"]: row for row in prompts}
    for index, row in enumerate(selected):
        left = gap + index % columns * (tile + gap)
        top = header + index // columns * (tile + label + gap)
        prompt = prompt_lookup[row["case_id"]]
        draw.text((left, top), f"{row['case_id']} | {prompt['category']}", fill="black", font=font)
        path = portable_path(root, row["artifacts"]["image"])
        with Image.open(path) as source:
            thumbnail = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (left, top + label))
        bindings.append({"case_id": row["case_id"], "category": prompt["category"], "seed": row["seed"],
                         "source_image": path.relative_to(root).as_posix(), "source_image_sha256": sha256_file(path),
                         "mandatory_criteria": row["mandatory_criteria"]})
    directory = pair / args.run_id / "qa"
    directory.mkdir(exist_ok=True)
    target = directory / (args.model + "_first_per_category.png")
    canvas.save(target)
    manifest = {"schema_version": 1, "created_at_utc": utc_now(), "run_id": args.run_id, "model": args.model,
                "selection_rule": "Lexicographically first prompt_id within each frozen category, seed8410 by default; no selection by output quality",
                "inspection_scope": "Post-run technical/qualitative AI-assisted QA only; not a human grade or representative quality estimate",
                "thumbnail_transform": "RGB conversion and deterministic512x512LANCZOS downsample; original images unchanged",
                "image": target.relative_to(root).as_posix(), "image_sha256": sha256_file(target), "sources": bindings}
    write_json(directory / (args.model + "_contact_sheet_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
