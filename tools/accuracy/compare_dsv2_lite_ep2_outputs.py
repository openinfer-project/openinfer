#!/usr/bin/env python3
"""Compare HF, host-staged, and NCCL DeepSeek-V2-Lite EP=2 greedy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_u32_le(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_or_stdout(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise


@dataclass
class Output:
    name: str
    model_path: str | None
    backend: str | None
    prompt: str | None
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    reported_token_sha256: str | None
    reported_text_sha256: str | None
    token_sha256: str
    text_sha256: str
    raw: dict[str, Any]


def normalize(name: str, payload: dict[str, Any]) -> Output:
    token_ids = payload.get("generated_token_ids")
    if token_ids is None:
        token_ids = payload.get("output_tokens")
    if token_ids is None:
        token_ids = payload.get("tokens")
    if token_ids is None:
        token_ids = []

    generated_text = payload.get("generated_text")
    if generated_text is None:
        generated_text = payload.get("output_text")
    if generated_text is None:
        generated_text = payload.get("output")
    if generated_text is None:
        generated_text = ""

    reported_token_hash = payload.get("token_sha256")
    if reported_token_hash is None:
        reported_token_hash = payload.get("output_token_sha256")
    reported_text_hash = payload.get("text_sha256")
    if reported_text_hash is None:
        reported_text_hash = payload.get("output_text_sha256")

    prompt_token_ids = payload.get("prompt_token_ids") or []

    return Output(
        name=name,
        model_path=payload.get("model_path"),
        backend=payload.get("ep_backend") or payload.get("backend"),
        prompt=payload.get("prompt"),
        prompt_token_ids=[int(value) for value in prompt_token_ids],
        generated_token_ids=[int(value) for value in token_ids],
        generated_text=str(generated_text),
        reported_token_sha256=reported_token_hash,
        reported_text_sha256=reported_text_hash,
        token_sha256=sha256_u32_le([int(value) for value in token_ids]),
        text_sha256=sha256_text(str(generated_text)),
        raw=payload,
    )


def first_token_diff(left: Output, right: Output) -> dict[str, Any] | None:
    limit = min(len(left.generated_token_ids), len(right.generated_token_ids))
    for index in range(limit):
        left_token = left.generated_token_ids[index]
        right_token = right.generated_token_ids[index]
        if left_token != right_token:
            return {
                "index": index,
                left.name: left_token,
                right.name: right_token,
                "reason": "token_mismatch",
            }
    if len(left.generated_token_ids) != len(right.generated_token_ids):
        return {
            "index": limit,
            left.name: left.generated_token_ids[limit]
            if len(left.generated_token_ids) > limit
            else None,
            right.name: right.generated_token_ids[limit]
            if len(right.generated_token_ids) > limit
            else None,
            "reason": "length_mismatch",
        }
    return None


def pair_summary(left: Output, right: Output) -> dict[str, Any]:
    token_exact = left.generated_token_ids == right.generated_token_ids
    text_exact = left.generated_text == right.generated_text
    return {
        "token_exact": token_exact,
        "text_exact": text_exact,
        "first_different_token": None if token_exact else first_token_diff(left, right),
    }


def classify(pairs: dict[str, dict[str, Any]]) -> str:
    host_nccl_exact = pairs["host_staged_vs_nccl"]["token_exact"] and pairs[
        "host_staged_vs_nccl"
    ]["text_exact"]
    hf_host_exact = pairs["hf_vs_host_staged"]["token_exact"] and pairs[
        "hf_vs_host_staged"
    ]["text_exact"]
    hf_nccl_exact = pairs["hf_vs_nccl"]["token_exact"] and pairs["hf_vs_nccl"][
        "text_exact"
    ]
    if host_nccl_exact and hf_host_exact and hf_nccl_exact:
        return "all_token_text_exact"
    if not host_nccl_exact:
        return "nccl_transport_regression"
    return "pegainfer_baseline_accuracy_gap"


def short(text: str, width: int = 72) -> str:
    one_line = text.replace("\n", "\\n")
    if len(one_line) <= width:
        return one_line
    return one_line[: width - 3] + "..."


def table(outputs: list[Output]) -> str:
    rows = [
        "| Source | Backend | Tokens | Token SHA256 | Text SHA256 | Text |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for output in outputs:
        rows.append(
            "| {name} | {backend} | {tokens} | `{token_hash}` | `{text_hash}` | `{text}` |".format(
                name=output.name,
                backend=output.backend or "-",
                tokens=len(output.generated_token_ids),
                token_hash=output.token_sha256,
                text_hash=output.text_sha256,
                text=short(output.generated_text),
            )
        )
    return "\n".join(rows)


def hash_warnings(outputs: list[Output]) -> list[str]:
    warnings = []
    for output in outputs:
        if (
            output.reported_token_sha256
            and output.reported_token_sha256 != output.token_sha256
        ):
            warnings.append(
                f"{output.name}: reported token hash {output.reported_token_sha256} "
                f"does not match recomputed {output.token_sha256}"
            )
        if output.reported_text_sha256 and output.reported_text_sha256 != output.text_sha256:
            warnings.append(
                f"{output.name}: reported text hash {output.reported_text_sha256} "
                f"does not match recomputed {output.text_sha256}"
            )
    return warnings


def context_warnings(hf: Output, host: Output, nccl: Output) -> list[str]:
    warnings = []
    outputs = [hf, host, nccl]

    prompts = {output.prompt for output in outputs if output.prompt is not None}
    if len(prompts) > 1:
        warnings.append(f"prompt mismatch across outputs: {sorted(prompts)!r}")

    prompt_token_ids = {
        tuple(output.prompt_token_ids)
        for output in outputs
        if output.prompt_token_ids
    }
    if len(prompt_token_ids) > 1:
        warnings.append("prompt_token_ids mismatch across outputs")

    model_paths = {output.model_path for output in outputs if output.model_path is not None}
    if len(model_paths) > 1:
        warnings.append(
            "model_path labels differ; verify all outputs use the same snapshot: "
            f"{sorted(model_paths)!r}"
        )

    hf_output_len = hf.raw.get("output_len")
    peg_output_lens = {
        output.raw.get("max_new_tokens")
        for output in [host, nccl]
        if output.raw.get("max_new_tokens") is not None
    }
    output_lens = set()
    if hf_output_len is not None:
        output_lens.add(hf_output_len)
    output_lens.update(peg_output_lens)
    if len(output_lens) > 1:
        warnings.append(f"output length labels differ across outputs: {sorted(output_lens)!r}")

    hf_generation_mode = hf.raw.get("generation_mode")
    if hf_generation_mode not in (
        None,
        "incremental_past_key_values",
        "transformers_generate_use_cache",
    ):
        warnings.append(
            "HF output generation_mode is not a recognized HF generation mode: "
            f"{hf_generation_mode}"
        )
    if host.backend not in (None, "host-staged"):
        warnings.append(f"host-staged file reports backend {host.backend!r}")
    if nccl.backend not in (None, "nccl"):
        warnings.append(f"NCCL file reports backend {nccl.backend!r}")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="HF JSON output")
    parser.add_argument("--host-staged", required=True, help="host-staged pegainfer JSON output")
    parser.add_argument("--nccl", required=True, help="NCCL pegainfer JSON output")
    parser.add_argument("--out", help="Optional path for structured comparison JSON")
    parser.add_argument(
        "--require-all-exact",
        action="store_true",
        help="Exit nonzero unless HF, host-staged, and NCCL are token/text exact",
    )
    args = parser.parse_args()

    hf = normalize("hf", load_json_or_stdout(Path(args.hf)))
    host = normalize("host_staged", load_json_or_stdout(Path(args.host_staged)))
    nccl = normalize("nccl", load_json_or_stdout(Path(args.nccl)))
    outputs = [hf, host, nccl]

    pairs = {
        "hf_vs_host_staged": pair_summary(hf, host),
        "hf_vs_nccl": pair_summary(hf, nccl),
        "host_staged_vs_nccl": pair_summary(host, nccl),
    }
    classification = classify(pairs)
    warnings = hash_warnings(outputs) + context_warnings(hf, host, nccl)
    result = {
        "classification": classification,
        "outputs": {
            output.name: {
                "model_path": output.model_path,
                "backend": output.backend,
                "prompt": output.prompt,
                "prompt_token_ids": output.prompt_token_ids,
                "generated_token_ids": output.generated_token_ids,
                "generated_text": output.generated_text,
                "token_sha256": output.token_sha256,
                "text_sha256": output.text_sha256,
                "reported_token_sha256": output.reported_token_sha256,
                "reported_text_sha256": output.reported_text_sha256,
            }
            for output in outputs
        },
        "pairs": pairs,
        "warnings": warnings,
    }

    print(table(outputs))
    print()
    print(f"Classification: {classification}")
    print(json.dumps({"pairs": pairs, "warnings": warnings}, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")

    if args.require_all_exact and classification != "all_token_text_exact":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
