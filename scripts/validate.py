#!/usr/bin/env python3
"""Validate examples against the schema and check registry consistency.

Usage: python scripts/validate.py   (needs pyyaml, jsonschema)
Exit status is non-zero on any failure, including when a tests/invalid/*
file unexpectedly validates.
"""
import json, pathlib, sys
import yaml
from jsonschema import Draft202012Validator

root = pathlib.Path(__file__).resolve().parent.parent
schema = json.loads((root / "schemas/model.schema.json").read_text())
validator = Draft202012Validator(schema)
registry = yaml.safe_load((root / "registry/types.yaml").read_text())
portable = {t["portable"] for t in registry["types"]}
frameworks = ["gpy", "gpc", "gpmat", "gpytorch", "gpflow", "gpjax"]
failures = 0


def tags(node):
    yield node["type"]
    for p in node.get("parts", []):
        yield from tags(p)


def load(path):
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


for path in sorted((root / "examples").glob("*.*")):
    doc = load(path)
    errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    used = {doc["type"], doc["likelihood"]["type"], *tags(doc["kernel"])}
    unknown = used - portable
    for e in errs:
        print(f"FAIL {path.name}: {list(e.path)}: {e.message}"); failures += 1
    for u in sorted(unknown):
        print(f"FAIL {path.name}: type tag not in registry: {u}"); failures += 1
    if "data" in doc and "inducing_inputs" in doc:
        d = doc["input_dim"]
        for name, m in (("data.X", doc["data"]["X"]), ("inducing_inputs", doc["inducing_inputs"])):
            if any(len(r) != d for r in m):
                print(f"FAIL {path.name}: {name} width != input_dim"); failures += 1
    if not errs and not unknown:
        print(f"ok   {path.name}")

for path in sorted((root / "tests/invalid").glob("*.yaml")):
    if not list(validator.iter_errors(load(path))):
        print(f"FAIL {path.name}: invalid fixture validated"); failures += 1
    else:
        print(f"ok   invalid/{path.name} rejected")

for t in registry["types"]:
    for f in frameworks:
        if f not in t:
            print(f"FAIL registry {t['portable']}: missing framework key {f}"); failures += 1

sys.exit(1 if failures else 0)
