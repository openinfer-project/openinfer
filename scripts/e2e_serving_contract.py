#!/usr/bin/env python3
"""E2E checks for the Kimi-K2 serving contract against a running server.

Three checks through `/v1/completions` (issues #238, #237):

1. EOS stop: the answer must end early with `finish_reason: "stop"` and fewer
   than `max_tokens` completion tokens;
2. `ignore_eos: true`: generation must run to `finish_reason: "length"` with
   exactly `max_tokens` completion tokens (benchmarks rely on this);
3. non-greedy reject: a `temperature>0` request must fail with an HTTP error
   and produce no completion — silently-greedy output is the one forbidden
   state. (The vllm-server HTTP layer collapses engine rejections into a
   generic 500; the "decodes greedy only" reason appears in the server log.)

The default prompt is in Kimi-K2 chat format; pass --prompt for other models.

Usage:
    python3 scripts/e2e_serving_contract.py --base-url http://127.0.0.1:8124 --model kimi-k2.5
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

KIMI_PROMPT = (
    "<|im_system|>system<|im_middle|>You are a helpful assistant.<|im_end|>"
    "<|im_user|>user<|im_middle|>What is 2+2? Reply with just the number.<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|>"
)


def post_completion(base_url: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def request_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool, timeout: float
) -> dict:
    status, payload = post_completion(
        base_url,
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": ignore_eos,
        },
        timeout,
    )
    if status != 200:
        raise RuntimeError(f"greedy request failed with HTTP {status}: {payload}")
    choice = payload["choices"][0]
    return {
        "finish_reason": choice["finish_reason"],
        "completion_tokens": payload["usage"]["completion_tokens"],
        "text": choice["text"],
    }


def check(label: str, condition: bool, detail: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8124")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=KIMI_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    stop_run = request_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, ignore_eos=False, timeout=args.timeout
    )
    print(f"stop run: {stop_run['finish_reason']}, {stop_run['completion_tokens']} tokens")
    print(f"  text: {stop_run['text']!r}")

    length_run = request_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, ignore_eos=True, timeout=args.timeout
    )
    print(f"ignore_eos run: {length_run['finish_reason']}, {length_run['completion_tokens']} tokens")

    sampling_status, sampling_payload = post_completion(
        args.base_url,
        {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
        },
        args.timeout,
    )
    sampling_body = json.dumps(sampling_payload, ensure_ascii=False)
    print(f"non-greedy run: HTTP {sampling_status}")
    print(f"  body: {sampling_body[:300]}")
    sampling_text = "".join(
        choice.get("text") or "" for choice in sampling_payload.get("choices", [])
    )

    ok = True
    ok &= check(
        "EOS stops generation",
        stop_run["finish_reason"] == "stop",
        f"finish_reason={stop_run['finish_reason']}",
    )
    ok &= check(
        "stop run ends early",
        stop_run["completion_tokens"] < args.max_tokens,
        f"{stop_run['completion_tokens']} < {args.max_tokens}",
    )
    ok &= check(
        "ignore_eos runs to length",
        length_run["finish_reason"] == "length",
        f"finish_reason={length_run['finish_reason']}",
    )
    ok &= check(
        "ignore_eos generates max_tokens",
        length_run["completion_tokens"] == args.max_tokens,
        f"{length_run['completion_tokens']} == {args.max_tokens}",
    )
    ok &= check(
        "non-greedy is explicitly rejected",
        sampling_status != 200 and not sampling_text,
        f"HTTP {sampling_status}, generated text: {sampling_text!r}",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
