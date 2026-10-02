#!/usr/bin/env python3
"""exact_check.py save|compare <ref.json> [base-url]

Replies must stay bit-identical under a speed change. Sends a fixed set of greedy (temperature 0) requests - prose,
code, JSON, arithmetic, a tool call, a long-context recall, a multi-turn chat - and stores (save) or compares
(compare) the exact reply text and finish reason. Exit code 1 on any difference.
"""
import json
import sys
import urllib.request

mode, ref = sys.argv[1], sys.argv[2]
base = (sys.argv[3] if len(sys.argv) > 3 else "http://10.0.1.110:8888").rstrip("/")
MODEL = "GLM-5.3-Flash-EXL3"
TOOLS = [{"type": "function", "function": {
    "name": "add_tags", "description": "Attach tags to a document.",
    "parameters": {"type": "object", "required": ["doc_id", "tags"], "properties": {
        "doc_id": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}}}}]
FILLER = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7 % 13}." for i in range(400))
CASES = {
    "prose": [{"role": "user", "content": "Write two paragraphs on why the sky is blue."}],
    "code": [{"role": "user", "content": "Write a Python function that merges two sorted lists, with a docstring."}],
    "json": [{"role": "user", "content": "Return only JSON: an array of three objects with keys name and age."}],
    "math": [{"role": "user", "content": "What is 17 * 23 + 45? Show the steps briefly."}],
    "recall": [{"role": "user", "content": FILLER + "\n\nWhat was item 321's dog number? Answer with the number only."}],
    "chat": [{"role": "user", "content": "My name is Ada."}, {"role": "assistant", "content": "Hello Ada."},
             {"role": "user", "content": "What is my name and what rhymes with it?"}],
    "tool": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}],
}


def ask(msgs, tools):
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]
    m = ch["message"]
    return {"text": m.get("content") or "", "calls": [c["function"] for c in (m.get("tool_calls") or [])],
            "finish": ch["finish_reason"]}


got = {k: ask(v, k == "tool") for k, v in CASES.items()}
for k, v in got.items():
    print(f"{k}: finish={v['finish']} chars={len(v['text'])} calls={len(v['calls'])}")
if mode == "save":
    json.dump(got, open(ref, "w"), indent=1)
    print("saved", ref)
    sys.exit(0)
want = json.load(open(ref))
bad = [k for k in CASES if want.get(k) != got[k]]
print("IDENTICAL" if not bad else "DIFFERENT: " + ", ".join(bad))
sys.exit(1 if bad else 0)
