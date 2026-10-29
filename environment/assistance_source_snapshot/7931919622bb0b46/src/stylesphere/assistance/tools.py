"""Deterministic application-side validation. No arbitrary dispatch or mutation tools."""

from __future__ import annotations

import copy
import math
import re


class ToolValidationError(ValueError):
    pass


TOOL_SCHEMAS = {
    "search_products": {"required": {"query": "string"}, "optional": {"category": "string", "max_price": "number", "limit": "integer"}},
    "get_product_details": {"required": {"product_id": "string"}, "optional": {}},
    "check_inventory": {"required": {"product_id": "string"}, "optional": {"size": "string", "colour": "string"}},
    "get_delivery_policy": {"required": {}, "optional": {}},
    "get_return_policy": {"required": {}, "optional": {}},
}


def flatten_facts(value, prefix: str = "") -> dict:
    """Canonical exact-value facts, including composite values, for auditable citations."""
    result = {prefix: copy.deepcopy(value)} if prefix else {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten_facts(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(flatten_facts(item, f"{prefix}.{index}"))
    return result


def knowledge_facts(kb: dict) -> dict:
    facts = {}
    for product in kb.get("products", []):
        facts.update(flatten_facts(product, f"products.{product['product_id']}"))
    for policy in ("delivery_policy", "return_policy"):
        facts.update(flatten_facts(kb.get(policy, {}), policy))
    return facts


def validate_knowledge_base(kb: dict) -> None:
    if not isinstance(kb, dict) or not isinstance(kb.get("products"), list):
        raise ValueError("Knowledge base requires a products array")
    identifiers = set()
    required = {"product_id", "name", "category", "description", "price", "currency", "sizes", "colours", "inventory"}
    for product in kb["products"]:
        if not isinstance(product, dict) or required - set(product):
            raise ValueError("Product is missing required business fields")
        if product["product_id"] in identifiers:
            raise ValueError("Duplicate product_id")
        identifiers.add(product["product_id"])
        if not isinstance(product["product_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", product["product_id"]):
            raise ValueError("Invalid product_id")
        if isinstance(product["price"], bool) or not isinstance(product["price"], (int, float)) or not math.isfinite(product["price"]) or product["price"] < 0:
            raise ValueError("Product price must be a nonnegative number")
        if not isinstance(product["sizes"], list) or not isinstance(product["colours"], list) or not isinstance(product["inventory"], list):
            raise ValueError("Sizes, colours and inventory must be arrays")
        seen_variants = set()
        for variant in product["inventory"]:
            if not isinstance(variant, dict) or set(variant) != {"size", "colour", "quantity"}:
                raise ValueError("Inventory entries require size, colour and quantity")
            if type(variant["quantity"]) is not int or variant["quantity"] < 0:
                raise ValueError("Inventory quantity must be a nonnegative integer")
            if variant["size"] not in product["sizes"] or variant["colour"] not in product["colours"]:
                raise ValueError("Inventory variant is not an advertised size/colour")
            key = (variant["size"], variant["colour"])
            if key in seen_variants:
                raise ValueError("Duplicate inventory variant")
            seen_variants.add(key)
    if not isinstance(kb.get("delivery_policy"), dict) or not isinstance(kb.get("return_policy"), dict):
        raise ValueError("Knowledge base requires delivery_policy and return_policy objects")


class ToolController:
    """Each scenario receives its own permissions and visible business records."""

    def __init__(self, kb: dict, permitted_tools: list[str], available_business_records: list[str] | None = None):
        validate_knowledge_base(kb)
        if not isinstance(permitted_tools, list) or any(name not in TOOL_SCHEMAS for name in permitted_tools):
            raise ToolValidationError("Scenario contains an unknown tool permission")
        self._kb = copy.deepcopy(kb)
        self._products = {p["product_id"]: p for p in self._kb["products"]}
        self.permitted_tools = set(permitted_tools)
        self.visible = set(self._products) if available_business_records is None else set(available_business_records)
        self.visible_policies = {"delivery_policy", "return_policy"} if available_business_records is None else set(available_business_records) & {"delivery_policy", "return_policy"}
        self.trace: list[dict] = []
        self.observed_facts: dict = {}

    def schemas(self) -> dict:
        return {name: copy.deepcopy(TOOL_SCHEMAS[name]) for name in sorted(self.permitted_tools)}

    def validate(self, name: str, arguments: dict) -> dict:
        if name not in self.permitted_tools or name not in TOOL_SCHEMAS:
            raise ToolValidationError("Tool is not authorised for this scenario")
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be a JSON object")
        if name in {"get_delivery_policy", "get_return_policy"} and name.removeprefix("get_") not in self.visible_policies:
            raise ToolValidationError("Policy record is unavailable in this scenario")
        schema = TOOL_SCHEMAS[name]
        if set(schema["required"]) - set(arguments):
            raise ToolValidationError("Missing required tool arguments")
        if set(arguments) - set(schema["required"]) - set(schema["optional"]):
            raise ToolValidationError("Unexpected tool arguments")
        for key, value in arguments.items():
            kind = {**schema["required"], **schema["optional"]}[key]
            valid = ((kind == "string" and isinstance(value, str) and bool(value.strip()) and len(value) <= 500)
                     or (kind == "number" and type(value) in (int, float) and 0 <= value < 1e9)
                     or (kind == "integer" and type(value) is int and 1 <= value <= 20))
            if not valid:
                raise ToolValidationError(f"Invalid value for {key}")
        if "product_id" in arguments:
            pid = arguments["product_id"]
            if pid not in self._products or pid not in self.visible:
                raise ToolValidationError("Product is unknown or unavailable in this scenario")
            product = self._products[pid]
            if "size" in arguments and arguments["size"] not in product["sizes"]:
                raise ToolValidationError("Unknown size")
            if "colour" in arguments and arguments["colour"] not in product["colours"]:
                raise ToolValidationError("Unknown colour")
        return copy.deepcopy(arguments)

    def execute(self, name: str, arguments: dict) -> dict:
        event = {"proposed_tool": name, "proposed_arguments": copy.deepcopy(arguments)}
        try:
            args = self.validate(name, arguments)
            event["validated_tool"] = {"name": name, "arguments": args}
            if name == "search_products":
                terms = set(re.findall(r"\w+", args["query"].lower()))
                hits = []
                for product in self._products.values():
                    if product["product_id"] not in self.visible:
                        continue
                    if args.get("category") and product["category"].casefold() != args["category"].casefold():
                        continue
                    if "max_price" in args and product["price"] > args["max_price"]:
                        continue
                    corpus = set(re.findall(r"\w+", " ".join(str(product.get(k, "")) for k in ("name", "category", "description", "colours")).lower()))
                    score = len(terms & corpus)
                    if score:
                        hits.append((score, product))
                hits.sort(key=lambda pair: (-pair[0], pair[1]["product_id"]))
                products = [copy.deepcopy(p) for _, p in hits[:args.get("limit", 5)]]
                observation = {"products": products}
                for product in products:
                    self.observed_facts.update(flatten_facts(product, f"products.{product['product_id']}"))
            elif name == "get_product_details":
                product = self._products[args["product_id"]]
                observation = {"product": copy.deepcopy(product)}
                self.observed_facts.update(flatten_facts(product, f"products.{product['product_id']}"))
            elif name == "check_inventory":
                product = self._products[args["product_id"]]
                inventory = [(i, item) for i, item in enumerate(product["inventory"]) if all(item[k] == args[k] for k in ("size", "colour") if k in args)]
                observation = {"product_id": args["product_id"], "inventory": [{**copy.deepcopy(item), "fact_path": f"products.{args['product_id']}.inventory.{i}"} for i, item in inventory]}
                for i, item in inventory:
                    self.observed_facts.update(flatten_facts(item, f"products.{args['product_id']}.inventory.{i}"))
            elif name == "get_delivery_policy":
                observation = {"delivery_policy": copy.deepcopy(self._kb["delivery_policy"])}
                self.observed_facts.update(flatten_facts(self._kb["delivery_policy"], "delivery_policy"))
            else:  # The validated allowlist has exactly five tools.
                observation = {"return_policy": copy.deepcopy(self._kb["return_policy"])}
                self.observed_facts.update(flatten_facts(self._kb["return_policy"], "return_policy"))
            event.update(schema_valid=True, observation=copy.deepcopy(observation))
            self.trace.append(event)
            return observation
        except ToolValidationError as exc:
            event.update(schema_valid=False, error=str(exc))
            self.trace.append(event)
            raise
