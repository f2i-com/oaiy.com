"""Write golden cases for the Rust DeepSeek-V4.1 chat encoding
(crates/dsv41/src/chat.rs) from the reference encoding/encoding.py.

Two kinds of case: `encode` (messages + options -> prompt text, image
records) and `parse` (a completion -> the assistant message, or an error).

Usage: python tools/dsv41/chat_golden.py [--out E:/deepseek/golden/chat_cases.json]
"""

import argparse
import copy
import glob
import json
import os
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default=r"E:\deepseek\reference")
ap.add_argument("--out", default=r"E:\deepseek\golden\chat_cases.json")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.ref, "encoding"))
import encoding as enc  # noqa: E402

cases = []


def encode_case(messages, mode, effort=None, drop_thinking=True):
    try:
        prompt, media = enc.encode_messages(
            copy.deepcopy(messages), thinking_mode=mode, reasoning_effort=effort,
            drop_thinking=drop_thinking, return_multi_modal_data=True)
        cases.append({"kind": "encode", "messages": messages, "mode": mode, "effort": effort,
                      "drop_thinking": drop_thinking, "prompt": prompt, "images": media["images"]})
    except (AssertionError, ValueError, NotImplementedError, KeyError) as e:
        cases.append({"kind": "encode", "messages": messages, "mode": mode, "effort": effort,
                      "drop_thinking": drop_thinking, "error": str(e)})


def parse_case(text, mode):
    try:
        parsed = enc.parse_message_from_completion_text(text, thinking_mode=mode)
        cases.append({"kind": "parse", "text": text, "mode": mode, "message": parsed})
    except (AssertionError, ValueError) as e:
        cases.append({"kind": "parse", "text": text, "mode": mode, "error": str(e)})


# ---- the reference fixtures -------------------------------------------------
for path in sorted(glob.glob(os.path.join(a.ref, "encoding", "tests", "test_input_*.json"))):
    for case in enc.load_cases(path):
        for mode in ["chat", "thinking"]:
            encode_case(case["messages"], case.get("thinking_mode") or mode, case.get("reasoning_effort"))

# ---- building blocks ----------------------------------------------------------
lookup = {"type": "function", "function": {"name": "lookup", "description": "Look up a value",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                          "limit": {"type": "integer", "minimum": 1, "default": 5.0}}}}}
read_file = {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk",
             "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "路径"}},
                            "required": ["path"]}}}
ns_str = {"type": "function", "namespace": "search", "function": {"name": "lookup", "description": "ns lookup",
          "parameters": {"type": "object"}}}
ns_obj = {"type": "function", "namespace": {"name": "web", "description": "Web tools."},
          "function": {"name": "fetch", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}
ns_inner = {"type": "function", "function": {"name": "db::query", "description": "SQL",
            "parameters": {"type": "object"}}}
ns_nodesc = {"type": "function", "namespace": {"name": "fs"},
             "function": {"name": "ls", "parameters": {"type": "object"}}}


def call(name, args, id_=None, namespace=None):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if id_:
        c["id"] = id_
    if namespace:
        c["namespace"] = namespace
    return c


arg_variants = [
    '{"query": "value", "limit": 2}',
    '{"query":"línea\\n\\"quoted\\"","limit":2.5,"flag":true,"none":null,"list":[1,2.0,"x"],"obj":{"a":{"b":[]}}}',
    '"{\\"query\\": \\"double encoded\\"}"',
    'not json at all',
    '5',
    {"query": "a dict, not a string", "n": 1e-05},
    '{}',
]

sys_msg = {"role": "system", "content": "You are a coding assistant."}
for mode in ["chat", "thinking"]:
    for effort in [None, "low", "high", "max", 1, 42, 100]:
        encode_case([sys_msg, {"role": "user", "content": "hi"}], mode, effort)
    encode_case([{"role": "user", "content": "no system"}], mode)
    encode_case([{"role": "user", "content": ""}], mode)
    encode_case([{"role": "system", "content": None}, {"role": "user", "content": "x"}], mode)

    # multi-turn, with and without reasoning, drop_thinking both ways
    turns = [sys_msg, {"role": "user", "content": "q1"},
             {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
             {"role": "user", "content": "q2"},
             {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
             {"role": "user", "content": "q3"}]
    for drop in [True, False]:
        encode_case(turns, mode, drop_thinking=drop)
        encode_case(turns[:5], mode, drop_thinking=drop)

    # tools on the system message, tool calls, tool results out of order
    for tools in [[lookup], [lookup, read_file], [ns_str, ns_obj, ns_inner, ns_nodesc]]:
        encode_case([dict(sys_msg, tools=tools), {"role": "user", "content": "use tools"}], mode)
    for args in arg_variants:
        encode_case([dict(sys_msg, tools=[lookup]), {"role": "user", "content": "q"},
                     {"role": "assistant", "content": "", "reasoning_content": "think",
                      "tool_calls": [call("lookup", args, "c1")]},
                     {"role": "tool", "tool_call_id": "c1", "content": "result"}], mode)
    encode_case([dict(sys_msg, tools=[lookup, read_file]), {"role": "user", "content": "q"},
                 {"role": "assistant", "content": "calling two", "tool_calls": [
                     call("lookup", '{"query": "x"}', "a"), call("read_file", '{"path": "/tmp/f"}', "b")]},
                 {"role": "tool", "tool_call_id": "b", "content": "file text"},
                 {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "part 1"},
                                                                  {"type": "text", "text": "part 2"}]},
                 {"role": "user", "content": "and now?"},
                 {"role": "assistant", "content": "done", "reasoning_content": "because"},
                 {"role": "user", "content": "thanks"}], mode)
    encode_case([dict(sys_msg, tools=[ns_str, ns_obj]), {"role": "user", "content": "q"},
                 {"role": "assistant", "content": "", "tool_calls": [
                     call("lookup", '{"q": 1}', "x", namespace="search"), call("web::fetch", '{"url": "u"}', "y")]},
                 {"role": "tool", "tool_call_id": "x", "content": "r1"},
                 {"role": "tool", "tool_call_id": "y", "content": "r2"}], mode)
    encode_case([sys_msg, {"role": "user", "content": "q"},
                 {"role": "assistant", "tool_calls": [call("bad::name::x", '{}')]}], mode)
    encode_case([dict(sys_msg, tools=[{"type": "function", "namespace": "a", "function": {"name": "b::c"}}]),
                 {"role": "user", "content": "conflict"}], mode)

    # consecutive users, mid-conversation system, latest reminder, tasks, response format
    encode_case([{"role": "user", "content": "one"}, {"role": "user", "content": "two"}], mode)
    encode_case([sys_msg, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                 {"role": "system", "content": "new rule"}], mode)
    encode_case([sys_msg, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                 {"role": "system", "content": "new rule"}, {"role": "user", "content": "q2"}], mode)
    encode_case([sys_msg, {"role": "latest_reminder", "content": "Today is Monday."},
                 {"role": "user", "content": "q"}], mode)
    encode_case([sys_msg, {"role": "user", "content": "q"},
                 {"role": "latest_reminder", "content": "reminder"}], mode)
    for task in ["action", "query", "authority", "domain", "read_url", "bogus"]:
        encode_case([{"role": "user", "content": "q", "task": task}], mode)
    encode_case([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "task": "title"},
                 {"role": "assistant", "content": "A title"}], mode)
    encode_case([{"role": "user", "content": "q", "task": "action"}, {"role": "assistant", "content": "Search"},
                 {"role": "user", "content": "next"}], mode)
    encode_case([dict(sys_msg, response_format={"type": "json_schema", "schema": {"type": "object"}}),
                 {"role": "user", "content": "q"}], mode)
    encode_case([{"role": "user", "content": "q"}, {"role": "assistant", "content": "partial", "wo_eos": True}], mode)

    # images: OpenAI parts, nested in tool results, rejected placeholders
    encode_case([{"role": "user", "content": [{"type": "text", "text": "first"},
                                              {"type": "image_url", "image_url": {"url": "a.png"}},
                                              {"type": "text", "text": "what is this?"},
                                              {"type": "image_url", "image_url": "data:image/png;base64,AAA"}]}], mode)
    encode_case([{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]}], mode)
    encode_case([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                              "content": [{"type": "text", "text": "see"},
                                                          {"type": "image_url", "image_url": {"url": "b.png"}}]}]}], mode)
    encode_case([{"role": "user", "content": "hi <｜deepseek_image｜>"}], mode)
    encode_case([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}], mode)
    encode_case([{"role": "system", "content": [{"type": "text", "text": "sys a"}, {"type": "text", "text": "sys b"}]},
                 {"role": "user", "content": [{"type": "input_audio"}, {"type": "text", "text": "x"}]}], mode)

# random conversations
rng = random.Random(7)
words = ["alpha", "βeta", "中文", "code()", "  spaced  ", "line\nbreak", "{\"j\": 1}", "", "émoji 😀"]
for i in range(300):
    mode = rng.choice(["chat", "thinking"])
    msgs = []
    if rng.random() < 0.8:
        msgs.append({"role": "system", "content": rng.choice(words), **({"tools": [lookup]} if rng.random() < 0.4 else {})})
    for _ in range(rng.randint(1, 5)):
        msgs.append({"role": "user", "content": rng.choice(words)})
        r = rng.random()
        if r < 0.3:
            msgs.append({"role": "assistant", "content": rng.choice(words), "reasoning_content": rng.choice(words)})
        elif r < 0.5:
            msgs.append({"role": "assistant", "content": rng.choice(words),
                         "tool_calls": [call("lookup", rng.choice(arg_variants[:2]), f"id{i}")]})
            msgs.append({"role": "tool", "tool_call_id": f"id{i}", "content": rng.choice(words)})
    encode_case(msgs, mode, rng.choice([None, "low", 60]), rng.random() < 0.8)

# ---- parse cases --------------------------------------------------------------
dsml = enc.dsml_token
tool_block = (f"\n\n<{dsml} calls>\n<{dsml} invoke name=\"lookup\">\n"
              f"<{dsml} parameter name=\"query\" string=\"true\">va\"lue\nx</{dsml} parameter>\n"
              f"<{dsml} parameter name=\"limit\" string=\"false\">2</{dsml} parameter>\n</{dsml} invoke>\n"
              f"<{dsml} invoke name=\"web::fetch\">\n</{dsml} invoke>\n</{dsml} calls>")
eos = enc.eos_token
for text, mode in [
    ("Hello there." + eos, "chat"),
    ("reasoning here</think>Answer." + eos, "thinking"),
    ("  reason  </think>summary" + tool_block + eos, "thinking"),
    ("summary" + tool_block + eos, "chat"),
    (tool_block[2:] + eos, "chat"),
    ("no eos", "chat"),
    ("no think end" + eos, "thinking"),
    ("text" + tool_block + "trailing" + eos, "chat"),
    ("a <think> b" + eos, "chat"),
    (f"x\n\n<{dsml} calls>\n<{dsml} invoke name=\"f\">\n<{dsml} parameter name=\"a\" string=\"maybe\">1</{dsml} parameter>\n</{dsml} invoke>\n</{dsml} calls>" + eos, "chat"),
    (f"x\n\n<{dsml} calls>\n<{dsml} invoke name=\"f\">\n<{dsml} parameter name=\"a\" string=\"true\">1</{dsml} parameter>\n<{dsml} parameter name=\"a\" string=\"true\">2</{dsml} parameter>\n</{dsml} invoke>\n</{dsml} calls>" + eos, "chat"),
    (f"x\n\n<{dsml} calls>\n<{dsml} invoke name=\"f\">\n<{dsml} parameter name=\"p\" string=\"true\">a\" string=\"true\">b</{dsml} parameter>\n</{dsml} invoke>\n</{dsml} calls>" + eos, "chat"),
]:
    parse_case(text, mode)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=1)
n_err = sum(1 for c in cases if "error" in c)
print(f"{len(cases)} cases ({n_err} errors) -> {a.out}")
