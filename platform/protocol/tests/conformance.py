#!/usr/bin/env python3
"""Conformance test for the OAIY Bridge Protocol v1 schemas.

Checks three things, in order of how easily they rot:

1. **Every schema is a valid JSON Schema 2020-12** and every `$ref` resolves.
   A typo in a `$ref` makes a subschema silently unenforced, which looks exactly
   like a passing test.

2. **Valid documents are accepted.** The cheap half.

3. **Invalid documents are REJECTED.** The half that matters. A schema with a
   misplaced `allOf`, or one missing `additionalProperties: false`, accepts
   anything and sails through a positive-only suite. Each negative case below
   targets a specific invariant the README claims, so if the prose and the schema
   drift apart, this fails.

Run:  python protocol/tests/conformance.py
Exit: 0 only when every case behaves as declared.
"""
from __future__ import annotations

import json
import pathlib
import sys

try:
    from jsonschema import Draft202012Validator
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
except ImportError:
    sys.stderr.write(
        "This test needs `jsonschema` (>=4.18, which bundles `referencing`):\n"
        "    pip install jsonschema\n"
    )
    raise SystemExit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent
V1 = ROOT / "v1"

# Windows consoles still default to cp1252, which cannot encode the tick/cross
# this suite prints — so the whole run would die on its first PASSING assertion.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

passed = 0
failures: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failures.append(name)
        print(f"  ✗ {name}" + (f"  -> {detail}" if detail else ""))


def section(name: str) -> None:
    print(f"\n-- {name} --")


# ---------------------------------------------------------------------------
# Load every schema and build a registry so cross-file $refs resolve offline.
# Resolving over the network would make this test need the internet to check a
# local contract, and would pass against a stale published copy.
schemas: dict[str, dict] = {}
for path in sorted(V1.glob("*.schema.json")):
    schemas[path.name] = json.loads(path.read_text(encoding="utf-8"))

registry = Registry().with_resources(
    [(s["$id"], Resource.from_contents(s)) for s in schemas.values()]
)


def validator(name: str) -> Draft202012Validator:
    schema = schemas[name]
    cls = validator_for(schema)
    return cls(schema, registry=registry)


section("schemas are well-formed")
ok("v1/ contains schemas", len(schemas) > 0, f"found {len(schemas)}")
for name, schema in schemas.items():
    try:
        validator_for(schema).check_schema(schema)
        ok(f"{name} is valid JSON Schema 2020-12", True)
    except Exception as e:  # noqa: BLE001 - report, don't crash the suite
        ok(f"{name} is valid JSON Schema 2020-12", False, str(e)[:160])

    ok(f"{name} declares $id", isinstance(schema.get("$id"), str))
    ok(f"{name} declares a description", isinstance(schema.get("description"), str)
       or isinstance(schema.get("title"), str))

section("every $ref resolves")


def refs_of(node) -> list[str]:
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(refs_of(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(refs_of(item))
    return found


known_ids = {s["$id"] for s in schemas.values()}
for name, schema in schemas.items():
    for ref in refs_of(schema):
        target = ref.split("#")[0]
        ok(f"{name} -> {target.rsplit('/', 1)[-1]}", target in known_ids, f"unresolved: {ref}")


# ---------------------------------------------------------------------------
# Cases. Each is (schema, label, document, should_validate).
CALLER = {"product": "formlogic", "tenantId": "u_8814", "scopeId": "app:receptionist"}

CASES: list[tuple[str, str, dict, bool]] = [
    # --- health / handshake ---
    ("health.schema.json", "a healthy desktop",
     {"status": "ok", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1", "version": "0.1.0"}, True),
    ("health.schema.json", "degraded WITHOUT detail is refused",
     {"status": "degraded", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1", "version": "0.1.0"}, False),
    ("health.schema.json", "degraded WITH detail is accepted",
     {"status": "degraded", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1",
      "version": "0.1.0", "detail": "Plugin host failed to start."}, True),
    ("health.schema.json", "a foreign protocol major is refused",
     {"status": "ok", "product": "oaiy-desktop", "protocol": "oaiy-bridge/2", "version": "0.1.0"}, False),

    # --- caller: the genericity guarantee ---
    ("caller.schema.json", "product alone is enough", {"product": "cli"}, True),
    ("caller.schema.json", "an unknown field is refused",
     {"product": "formlogic", "appSlug": "receptionist"}, False),
    ("caller.schema.json", "an upper-case product is refused", {"product": "FormLogic"}, False),
    ("caller.schema.json", "an empty product is refused", {"product": ""}, False),

    # --- capabilities: unavailable must explain itself ---
    ("capability-manifest.schema.json", "an available capability",
     {"protocol": "oaiy-bridge/1", "runtime": "desktop",
      "capabilities": [{"id": "oaiy.llm.chat", "available": True, "connections": ["conn_llama"]}]}, True),
    ("capability-manifest.schema.json", "unavailable WITHOUT a reason is refused",
     {"protocol": "oaiy-bridge/1", "runtime": "desktop",
      "capabilities": [{"id": "oaiy.image.generate", "available": False}]}, False),
    ("capability-manifest.schema.json", "unavailable WITHOUT detail is refused",
     {"protocol": "oaiy-bridge/1", "runtime": "desktop",
      "capabilities": [{"id": "oaiy.image.generate", "available": False, "reason": "service_stopped"}]}, False),
    ("capability-manifest.schema.json", "unavailable, fully explained, is accepted",
     {"protocol": "oaiy-bridge/1", "runtime": "desktop",
      "capabilities": [{"id": "oaiy.image.generate", "available": False, "reason": "service_stopped",
                        "detail": "Open OAIY Desktop -> Services -> Krea-2 Turbo -> Start."}]}, True),
    ("capability-manifest.schema.json", "an invented unavailable-reason is refused",
     {"protocol": "oaiy-bridge/1", "runtime": "desktop",
      "capabilities": [{"id": "oaiy.image.generate", "available": False,
                        "reason": "because_reasons", "detail": "x"}]}, False),
    ("capability-manifest.schema.json", "an unknown runtime is refused",
     {"protocol": "oaiy-bridge/1", "runtime": "toaster", "capabilities": []}, False),

    # --- run request: idempotency and flow identity ---
    ("run-request.schema.json", "a flow invoked by id",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "customer-follow-up",
      "input": {"name": "Dana"}, "correlationId": "call_abc", "idempotencyKey": "binding:97:resp:5512"}, True),
    ("run-request.schema.json", "a flow invoked inline",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "graph": {"nodes": [], "edges": []},
      "correlationId": "c", "idempotencyKey": "k"}, True),
    ("run-request.schema.json", "NO idempotencyKey is refused",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "f", "correlationId": "c"}, False),
    ("run-request.schema.json", "neither flowId nor graph is refused",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "correlationId": "c", "idempotencyKey": "k"}, False),
    ("run-request.schema.json", "BOTH flowId and graph is refused",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "f", "graph": {"nodes": []},
      "correlationId": "c", "idempotencyKey": "k"}, False),
    ("run-request.schema.json", "FormLogic's appContext is refused",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "appContext": {"appSlug": "x"},
      "flowId": "f", "correlationId": "c", "idempotencyKey": "k"}, False),
    ("run-request.schema.json", "lineage depth 16 is allowed",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "f", "correlationId": "c",
      "idempotencyKey": "k", "lineage": {"rootRunId": "r", "depth": 16}}, True),
    ("run-request.schema.json", "lineage depth 17 is refused (the fork-bomb cap)",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "f", "correlationId": "c",
      "idempotencyKey": "k", "lineage": {"rootRunId": "r", "depth": 17}}, False),
    ("run-request.schema.json", "an unknown mode is refused",
     {"protocol": "oaiy-bridge/1", "caller": CALLER, "flowId": "f", "correlationId": "c",
      "idempotencyKey": "k", "mode": "eventually"}, False),

    # --- run result: a failure must carry an error ---
    ("run-result.schema.json", "a successful run",
     {"runId": "run_1", "status": "succeeded", "output": {"message": "done"}}, True),
    ("run-result.schema.json", "failed WITHOUT an error is refused",
     {"runId": "run_1", "status": "failed"}, False),
    ("run-result.schema.json", "timed_out WITHOUT an error is refused",
     {"runId": "run_1", "status": "timed_out"}, False),
    ("run-result.schema.json", "failed WITH a typed error is accepted",
     {"runId": "run_1", "status": "failed",
      "error": {"code": "node_failed", "message": "Krea-2 refused the prompt.", "nodeId": "img_1"}}, True),
    ("run-result.schema.json", "an invented status is refused",
     {"runId": "run_1", "status": "mostly_fine"}, False),
    ("run-result.schema.json", "'done' (the FormLogic status) is refused",
     {"runId": "run_1", "status": "done"}, False),

    # --- errors: the taxonomy is closed, and one code must be actionable ---
    ("error.schema.json", "a node failure",
     {"code": "node_failed", "message": "boom", "retryable": False}, True),
    ("error.schema.json", "capability_unavailable WITHOUT detail is refused",
     {"code": "capability_unavailable", "message": "The image service is not running."}, False),
    ("error.schema.json", "capability_unavailable WITH actionable detail is accepted",
     {"code": "capability_unavailable", "message": "The image service is not running.",
      "detail": "Open OAIY Desktop -> Services -> Krea-2 Turbo -> Start.",
      "capability": "oaiy.image.generate", "retryable": True}, True),
    ("error.schema.json", "an invented code is refused", {"code": "oops", "message": "x"}, False),
    ("error.schema.json", "an empty message is refused", {"code": "internal", "message": ""}, False),

    # --- events: dedupe fields are not optional ---
    ("event-envelope.schema.json", "an incoming call",
     {"schemaVersion": 1, "source": "aokie", "name": "aokie.call.incoming",
      "correlationId": "call_abc", "idempotencyKey": "aokie:call_abc:incoming:v1",
      "occurredAt": "2026-07-30T04:12:09Z", "data": {"callerNumber": "+61400000000"}}, True),
    ("event-envelope.schema.json", "NO idempotencyKey is refused",
     {"schemaVersion": 1, "source": "aokie", "name": "aokie.call.incoming",
      "correlationId": "call_abc", "occurredAt": "2026-07-30T04:12:09Z", "data": {}}, False),
    ("event-envelope.schema.json", "NO correlationId is refused",
     {"schemaVersion": 1, "source": "aokie", "name": "aokie.call.incoming",
      "idempotencyKey": "k", "occurredAt": "2026-07-30T04:12:09Z", "data": {}}, False),
    ("event-envelope.schema.json", "an un-namespaced event name is refused",
     {"schemaVersion": 1, "source": "aokie", "name": "incoming", "correlationId": "c",
      "idempotencyKey": "k", "occurredAt": "2026-07-30T04:12:09Z", "data": {}}, False),
    ("event-envelope.schema.json", "a future schemaVersion is refused",
     {"schemaVersion": 2, "source": "aokie", "name": "aokie.call.incoming", "correlationId": "c",
      "idempotencyKey": "k", "occurredAt": "2026-07-30T04:12:09Z", "data": {}}, False),

    # --- triggers ---
    ("trigger-binding.schema.json", "an event-driven binding",
     {"id": "b1", "event": "aokie.call.incoming", "flowId": "caller-lookup", "mode": "async",
      "inputMap": {"callerPhone": "$event.data.callerNumber"}}, True),
    ("trigger-binding.schema.json", "a terminal flow event is bindable",
     {"id": "b2", "event": "flow.failed", "flowId": "notify-me", "mode": "background"}, True),
    ("trigger-binding.schema.json", "an unknown mode is refused",
     {"id": "b3", "event": "flow.failed", "flowId": "f", "mode": "whenever"}, False),

    # --- connectors ---
    ("connector-request.schema.json", "answering a call",
     {"connectorId": "aokie", "command": "call.answer", "payload": {"callId": "abc"}}, True),
    ("connector-request.schema.json", "an un-namespaced command is refused",
     {"connectorId": "aokie", "command": "answer"}, False),
    ("connector-request.schema.json", "an upper-case connectorId is refused",
     {"connectorId": "Aokie", "command": "call.answer"}, False),
    ("connector-response.schema.json", "a failure WITHOUT an error is refused", {"ok": False}, False),
    ("connector-response.schema.json", "a failure WITH an error is accepted",
     {"ok": False, "error": {"code": "node_failed", "message": "no dongle"}}, True),

    # --- artifacts: the inline cap ---
    ("artifact-ref.schema.json", "a loopback artifact ref",
     {"kind": "artifact", "id": "art_1", "mediaType": "image/png", "bytes": 184320,
      "url": "http://127.0.0.1:17972/api/bridge/artifacts/art_1"}, True),
    ("artifact-ref.schema.json", "a small inline payload is allowed",
     {"kind": "artifact", "id": "art_2", "mediaType": "text/plain", "inline": "aGVsbG8="}, True),
    ("artifact-ref.schema.json", "an oversized inline payload is refused",
     {"kind": "artifact", "id": "art_3", "mediaType": "image/png", "inline": "A" * 90000}, False),

    # --- devices ---
    ("device-registration.schema.json", "a desktop announcing itself",
     {"deviceId": "dev_1", "runtime": "desktop", "protocol": "oaiy-bridge/1",
      "deviceName": "Reception PC", "platform": "windows-x86_64",
      "capabilities": ["oaiy.llm.chat"], "lastSeenAt": "2026-07-30T04:12:09Z"}, True),

    # --- flow descriptors ---
    ("flow-descriptor.schema.json", "a flow a picker can render",
     {"flowId": "caller-lookup", "name": "Caller lookup", "revision": 7,
      "inputs": [{"name": "callerPhone", "required": True, "type": "string"}],
      "capabilities": ["oaiy.llm.chat"], "runtimes": ["desktop", "browser"]}, True),
    ("flow-descriptor.schema.json", "an unknown input type is refused",
     {"flowId": "f", "name": "F", "inputs": [{"name": "x", "type": "blob"}]}, False),
]

section("documents validate exactly as declared")
for schema_name, label, doc, should_pass in CASES:
    if schema_name not in schemas:
        ok(f"{schema_name}: {label}", False, "schema file missing")
        continue
    errs = sorted(validator(schema_name).iter_errors(doc), key=lambda e: list(e.path))
    valid = not errs
    if should_pass:
        ok(f"{schema_name}: {label}", valid, errs[0].message[:150] if errs else "")
    else:
        ok(f"{schema_name}: {label}", not valid, "ACCEPTED a document it must reject")

# ---------------------------------------------------------------------------
# The README is normative prose over these schemas. If it stops naming a schema
# that exists, the contract has an undocumented surface.
section("README covers every schema")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
for name in schemas:
    stem = name.replace(".schema.json", "")
    ok(f"README mentions {stem}", stem in readme or stem.replace("-", " ") in readme.lower())

print("\n" + "-" * 60)
print(f"bridge protocol conformance: {passed} passed, {len(failures)} failed")
if failures:
    print("failed:\n  - " + "\n  - ".join(failures))
raise SystemExit(1 if failures else 0)
