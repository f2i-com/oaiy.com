#!/usr/bin/env python3
"""Validate documents the relay produced (and requests a client would send) against the protocol package's JSON Schemas.

    python validate_contract.py <samples.json> [<schema dir>]

<samples.json> is a list of {"schema": "<name without .schema.json>", "doc": <document>, "label": "<what it is>"}.
A sample carries either "doc" (a parsed document) or "raw" (the exact text of a response, parsed here so that an empty
object stays an object). The schema directory defaults to platform/protocol/relay/v1. Exit 0 when every document
validates, 1 when any does not
(each failure is printed with its label and the first error), 2 on a usage or setup problem. Needs `jsonschema` >= 4.18.
"""
import json
import pathlib
import sys

try:
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
except ImportError:
    sys.stderr.write("needs jsonschema >= 4.18 (pip install jsonschema)\n")
    raise SystemExit(2)

if len(sys.argv) < 2:
    sys.stderr.write(__doc__)
    raise SystemExit(2)

here = pathlib.Path(__file__).resolve()
schema_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else here.parents[3] / "protocol" / "relay" / "v1"
samples = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))

schemas = {}
for p in sorted(schema_dir.glob("*.schema.json")):
    schemas[p.name[: -len(".schema.json")]] = json.loads(p.read_text(encoding="utf-8"))
if not schemas:
    sys.stderr.write(f"no schemas in {schema_dir}\n")
    raise SystemExit(2)

registry = Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in schemas.values()])

bad = 0
counts = {}
for s in samples:
    name = s["schema"]
    if name not in schemas:
        print(f"UNKNOWN SCHEMA {name} ({s.get('label')})")
        bad += 1
        continue
    schema = schemas[name]
    v = validator_for(schema)(schema, registry=registry)
    doc = json.loads(s["raw"]) if "raw" in s else s["doc"]
    s = dict(s, doc=doc)
    errors = sorted(v.iter_errors(doc), key=lambda e: list(e.path))
    counts[name] = counts.get(name, 0) + 1
    if errors:
        bad += 1
        e = errors[0]
        loc = "/".join(str(x) for x in e.path) or "(root)"
        print(f"INVALID {name} [{s.get('label')}] at {loc}: {e.message[:200]}")
        print("   document: " + json.dumps(s["doc"])[:300])
print(f"{len(samples)} documents against {len(counts)} schemas, {bad} invalid")
raise SystemExit(1 if bad else 0)
