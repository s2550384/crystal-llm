#!/usr/bin/env python3
"""
ELM proxy probe: checks how "native OpenAI" the ELM proxy behaves.

Standard library only. Usage:

    export ELM_API_KEY=elm-xxxxxxxx-xxxxxxxxxxxxxxxx
    python elm_proxy_test.py
    python elm_proxy_test.py --model gpt-4o-mini --search-model gpt-5.2 --local-model Qwen/Qwen3.5-397B-A17B-FP8

Each test makes a tiny request (capped output tokens). The search tests cost a
little more because OpenAI bills per search call. The script prints your /costs
before and after so you can see what the whole run spent.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1").rstrip("/")
KEY = os.environ.get("ELM_API_KEY")

RESULTS = []  # (status, name, detail)


# ---------------------------------------------------------------- helpers
def call(method, path, body=None, stream=False, timeout=120):
    """Return (status, lowercase_headers, parsed_body_or_stream)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {KEY}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status = resp.status
        headers = {k.lower(): v for k, v in resp.headers.items()}
        if stream:
            return status, headers, resp
        raw = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        headers = {k.lower(): v for k, v in e.headers.items()}
        raw = e.read()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = raw.decode(errors="replace")
    return status, headers, parsed


def record(status, name, detail=""):
    RESULTS.append((status, name, detail))
    print(f"[{status:<4}] {name}" + (f"\n         {detail}" if detail else ""))


def interesting_headers(headers):
    keep = []
    for k, v in headers.items():
        if (k.startswith("openai-") or k.startswith("x-ratelimit")
                or k in ("x-request-id", "server", "via", "cf-ray", "retry-after")):
            keep.append(f"{k}: {v}")
    return keep


def err_msg(body):
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            return str(e.get("message"))
    return str(body)[:300]


def pick(available, preferences):
    for p in preferences:
        if p in available:
            return p
    return None


def costs():
    s, _, b = call("GET", "/costs")
    if s == 200 and isinstance(b, dict):
        return b
    return None


# ---------------------------------------------------------------- tests
def test_models():
    s, h, b = call("GET", "/models")
    if s != 200:
        record("FAIL", "GET /models", f"HTTP {s}: {err_msg(b)}")
        return set()
    models = {m["id"]: m.get("owned_by", "?") for m in b.get("data", [])}
    openai_models = sorted(k for k, v in models.items() if v == "openai")
    other = sorted(k for k, v in models.items() if v != "openai")
    record("PASS", f"GET /models: {len(models)} models enabled",
           f"non-OpenAI: {other or 'none'}\n         OpenAI (first 15): {openai_models[:15]}")
    return set(models)


def test_basic_chat(model):
    s, h, b = call("POST", "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_completion_tokens": 20,
    })
    if s != 200:
        record("FAIL", f"chat/completions ({model})", f"HTTP {s}: {err_msg(b)}")
        return
    snapshot = b.get("model")
    detail = [f"returned model id: {snapshot}  (a dated snapshot suggests real OpenAI)",
              f"usage: {b.get('usage', {}).get('prompt_tokens')} prompt / "
              f"{b.get('usage', {}).get('completion_tokens')} completion tokens",
              f"fingerprint: {b.get('system_fingerprint')}"]
    hdrs = interesting_headers(h)
    detail.append("headers: " + ("; ".join(hdrs) if hdrs else "none of the usual OpenAI headers present"))
    record("PASS", f"chat/completions ({model})", "\n         ".join(detail))


def test_injected_prompt(model):
    """A bare 'hi' should cost only a few input tokens. Hundreds => something is prepended."""
    s, h, b = call("POST", "/responses", {
        "model": model, "input": "hi", "max_output_tokens": 50, "store": False})
    if s != 200:
        record("FAIL", f"responses bare prompt ({model})", f"HTTP {s}: {err_msg(b)}")
        return
    n = b.get("usage", {}).get("input_tokens")
    if n is None:
        record("INFO", "injected prompt check", "no usage returned")
    elif n <= 25:
        record("PASS", "no injected system prompt (responses)", f"input_tokens for 'hi' = {n}")
    else:
        record("WARN", "possible injected content (responses)",
               f"input_tokens for 'hi' = {n}; expected well under 25. Inspect the full request/response.")


def test_unknown_param(model):
    """Real OpenAI rejects unknown params with a recognisable message."""
    s, h, b = call("POST", "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "nonsense_param": 1,
        "max_completion_tokens": 5,
    })
    msg = err_msg(b)
    if s == 400 and "nonsense_param" in msg:
        record("PASS", "unknown parameter rejected with OpenAI-style error", msg)
    elif s == 200:
        record("INFO", "unknown parameter silently accepted/stripped",
               "ELM (or the model backend) is not passing this through strictly")
    else:
        record("INFO", "unknown parameter result", f"HTTP {s}: {msg}")


def test_stream_usage(model):
    """We do NOT ask for stream_options; docs say the proxy injects it."""
    s, h, stream = call("POST", "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "stream": True,
        "max_completion_tokens": 30,
    }, stream=True)
    if s != 200:
        record("FAIL", "streaming chat", f"HTTP {s}")
        return
    chunks, usage = 0, None
    for line in stream:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        chunks += 1
        if obj.get("usage"):
            usage = obj["usage"]
    if usage:
        record("PASS", "streaming works, usage injected in final chunk",
               f"{chunks} chunks; usage={usage.get('total_tokens')} total tokens")
    else:
        record("WARN", "streaming works but no usage chunk", f"{chunks} chunks")


def test_response_chaining(model):
    s1, _, b1 = call("POST", "/responses", {
        "model": model,
        "input": "Remember this codeword: PINEAPPLE. Reply 'ok'.",
        "max_output_tokens": 30, "store": True})
    if s1 != 200:
        record("FAIL", "responses store=true", f"HTTP {s1}: {err_msg(b1)}")
        return
    s2, _, b2 = call("POST", "/responses", {
        "model": model,
        "input": "What was the codeword?",
        "previous_response_id": b1.get("id"),
        "max_output_tokens": 30})
    if s2 != 200:
        record("WARN", "previous_response_id chaining failed",
               f"HTTP {s2}: {err_msg(b2)}  (possible effect of zero data retention)")
        return
    text = b2.get("output_text") or json.dumps(b2.get("output"))
    ok = "PINEAPPLE" in text.upper()
    record("PASS" if ok else "WARN", "previous_response_id chaining",
           f"reply: {text[:120]!r}")


def test_responses_search(model):
    s, h, b = call("POST", "/responses", {
        "model": model,
        "input": "What is the top UK news headline right now? One sentence, cite the source.",
        "tools": [{"type": "web_search"}],
        "max_output_tokens": 600,
        "store": False,
    })
    if s != 200:
        record("FAIL", f"responses + web_search tool ({model})",
               f"HTTP {s}: {err_msg(b)}")
        return
    items = b.get("output", [])
    types = [i.get("type") for i in items if isinstance(i, dict)]
    citations = 0
    for i in items:
        if isinstance(i, dict) and i.get("type") == "message":
            for part in i.get("content", []):
                for a in part.get("annotations", []) or []:
                    if a.get("type") == "url_citation":
                        citations += 1
    native = "web_search_call" in types and citations > 0
    record("PASS" if native else "WARN",
           f"responses + web_search tool ({model})",
           f"output item types: {types}\n         url_citation annotations: {citations}"
           + ("\n         => looks like OpenAI's native search" if native
              else "\n         => no native web_search_call/url_citation seen; tool may be ignored or replaced"))


def test_chat_search(model):
    s, h, b = call("POST", "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "One-sentence summary of today's top UK news, with source."}],
        "web_search_options": {},
        "max_completion_tokens": 600,
    })
    if s != 200:
        record("FAIL", f"chat/completions + web_search_options ({model})",
               f"HTTP {s}: {err_msg(b)}")
        return
    msg = (b.get("choices") or [{}])[0].get("message", {})
    ann = [a for a in (msg.get("annotations") or []) if a.get("type") == "url_citation"]
    record("PASS" if ann else "WARN",
           f"chat/completions + web_search_options ({model})",
           f"url_citation annotations: {len(ann)}\n         answer: {(msg.get('content') or '')[:160]!r}")


def test_local(model, available):
    if model not in available:
        record("SKIP", f"local model ({model})", "not enabled for your key")
        return
    s, h, b = call("POST", "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 20,
        "reasoning_effort": "none" if ("Qwen" in model or "Mistral-Small-4" in model) else None,
    } if ("Qwen" in model or "Mistral-Small-4" in model) else {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 20,
    })
    if s != 200:
        record("FAIL", f"local model ({model})", f"HTTP {s}: {err_msg(b)}")
        return
    content = (b.get("choices") or [{}])[0].get("message", {}).get("content")
    record("PASS", f"local model ({model})",
           f"reply: {content!r}; returned model: {b.get('model')}; "
           f"fingerprint: {b.get('system_fingerprint')}")


def test_responses_on_local(model, available):
    """Docs say /responses is OpenAI-only; confirm a local model is refused there."""
    if model not in available:
        return
    s, h, b = call("POST", "/responses", {"model": model, "input": "hi", "max_output_tokens": 20})
    record("INFO", f"/responses with local model ({model})", f"HTTP {s}: {err_msg(b) if s != 200 else 'accepted'}")


# ---------------------------------------------------------------- main
def main():
    if not KEY:
        sys.exit("Set ELM_API_KEY first (export ELM_API_KEY=elm-...)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="cheap OpenAI model for basic tests")
    ap.add_argument("--search-model", help="OpenAI model for Responses API web_search")
    ap.add_argument("--chat-search-model", default="gpt-5-search-api")
    ap.add_argument("--local-model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--skip-search", action="store_true", help="skip tests that incur search-call fees")
    args = ap.parse_args()

    print(f"Base URL: {BASE}\n")
    before = costs()
    if before:
        print(f"/costs before: {before['usage']} / {before['limit']} {before['currency']} "
              f"(as of {before['asOf']})\n")

    available = test_models()
    if not available:
        sys.exit("Cannot continue without a working /models call.")

    cheap = args.model or pick(available, ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-4.1-nano"])
    searchy = args.search_model or pick(available, ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-5.2", "gpt-5"])
    if not cheap:
        sys.exit("No suitable cheap OpenAI model found; pass --model.")

    print(f"\nUsing {cheap} for basic tests, {searchy} for Responses search\n")

    test_basic_chat(cheap)
    test_injected_prompt(cheap)
    test_unknown_param(cheap)
    test_stream_usage(cheap)
    test_response_chaining(cheap)

    if not args.skip_search:
        if searchy:
            test_responses_search(searchy)
        if args.chat_search_model in available:
            test_chat_search(args.chat_search_model)
        else:
            record("SKIP", f"chat search ({args.chat_search_model})", "not enabled for your key")

    test_local(args.local_model, available)
    test_responses_on_local(args.local_model, available)

    after = costs()
    print("\n" + "=" * 60)
    for status, name, _ in RESULTS:
        print(f"{status:<5} {name}")
    if before and after:
        print(f"\nSpend this run (approx, /costs syncs periodically): "
              f"{after['usage'] - before['usage']:.5f} {after['currency']}")
        print(f"Now: {after['usage']} / {after['limit']} {after['currency']} (as of {after['asOf']})")


if __name__ == "__main__":
    main()