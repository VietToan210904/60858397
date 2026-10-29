"""Prepare separate hypothetical smoke inputs without consulting model outputs."""
import json
from pathlib import Path
import shutil


def write_once(path, content):
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing development input differs: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    knowledge = root / "datasets/assistance/knowledge_base.json"
    kb = json.loads(knowledge.read_text(encoding="utf-8"))
    folder = root / "datasets/assistance_smoke"
    folder.mkdir(exist_ok=True)
    if not (folder / "knowledge_base.json").exists():
        shutil.copyfile(knowledge, folder / "knowledge_base.json")
    product, stock_product = kb["products"][10:12]
    pid, sid = product["product_id"], stock_product["product_id"]
    stock = stock_product["inventory"][0]
    scenarios = [
        {"scenario_id": "dev-assist-001", "customer_request": f"What is the price and currency of {pid}?",
         "available_business_records": [pid], "required_facts": [{"path": f"products.{pid}.price", "value": product["price"]}, {"path": f"products.{pid}.currency", "value": product["currency"]}],
         "permitted_tools": ["get_product_details"], "expected_tool_names": ["get_product_details"], "expected_outcome": {"type": "answer"}, "evaluation_category": "development_only", "hypothetical": True},
        {"scenario_id": "dev-assist-002", "customer_request": f"What is the available quantity for {sid}, colour {stock['colour']}, size {stock['size']}?",
         "available_business_records": [sid], "required_facts": [{"path": f"products.{sid}.inventory.0.quantity", "value": stock["quantity"]}],
         "permitted_tools": ["check_inventory"], "expected_tool_names": ["check_inventory"], "expected_outcome": {"type": "answer"}, "evaluation_category": "development_only", "hypothetical": True},
    ]
    write_once(folder / "scenarios.jsonl", "".join(json.dumps(s) + "\n" for s in scenarios))
    write_once(folder / "manifest.json", json.dumps({"purpose": "development only; excluded from final80scenarios", "hypothetical_inputs": True, "model_outputs_consulted": False}, indent=2))
    generation = root / "datasets/generation_smoke"
    generation.mkdir(exist_ok=True)
    brief = {"prompt_id": "dev-gen-001", "category": "development_only", "prompt": "Studio photograph of one plain tan leather loafer on an ivory background, three-quarter view, soft side lighting, no lettering.", "mandatory_criteria": ["one tan loafer", "ivory background", "no lettering"], "hypothetical_business_input": True}
    write_once(generation / "prompts.jsonl", json.dumps(brief) + "\n")
    write_once(generation / "manifest.json", json.dumps({"purpose": "development only; excluded from final24briefs", "hypothetical_inputs": True, "model_outputs_consulted": False}, indent=2))
    print("Prepared2development assistance scenarios and1development generation brief; no observations or ratings.")


if __name__ == "__main__":
    main()
