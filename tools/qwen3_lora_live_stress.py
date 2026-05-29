#!/usr/bin/env python3
"""Live Qwen3 LoRA API stress check.

This script focuses on serving semantics rather than HF parity:

- load multiple adapters into one process;
- verify loaded adapters appear in /v1/models;
- issue concurrent completions against base and multiple adapters;
- verify duplicate load fails unless load_inplace=true;
- verify load_inplace replacement keeps the adapter usable;
- verify unload removes the adapter from /v1/models and future requests fail.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--server-url")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Write one short sentence about systems software.")
    return parser.parse_args()


def read_config(model_path: Path) -> dict:
    return json.loads((model_path / "config.json").read_text())


def tensor_name(layer_idx: int, path_segment: str, lora_side: str) -> str:
    return f"base_model.model.model.layers.{layer_idx}.{path_segment}.{lora_side}.weight"


def patterned_tensor(torch, shape: tuple[int, ...], seed: int, scale: float):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.empty(shape, dtype=torch.float32)
    tensor.uniform_(-scale, scale, generator=generator)
    return tensor.to(torch.bfloat16)


def write_adapter(model_path: Path, adapter_path: Path, seed_offset: int, scale: float) -> None:
    from safetensors.torch import save_file
    import torch

    config = read_config(model_path)
    rank = 1
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(model_path),
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "lora_alpha": 1,
                "lora_dropout": 0.0,
                "peft_type": "LORA",
                "r": rank,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            },
            indent=2,
        )
    )

    hidden = int(config["hidden_size"])
    q_out = int(config["num_attention_heads"]) * int(config["head_dim"])
    v_out = int(config["num_key_value_heads"]) * int(config["head_dim"])
    tensors = {}
    for layer_idx in range(int(config["num_hidden_layers"])):
        base_seed = seed_offset + 1000 + layer_idx * 17
        tensors[tensor_name(layer_idx, "self_attn.q_proj", "lora_A")] = patterned_tensor(
            torch, (rank, hidden), base_seed, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.q_proj", "lora_B")] = patterned_tensor(
            torch, (q_out, rank), base_seed + 1, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.v_proj", "lora_A")] = patterned_tensor(
            torch, (rank, hidden), base_seed + 2, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.v_proj", "lora_B")] = patterned_tensor(
            torch, (v_out, rank), base_seed + 3, scale
        )
    save_file(tensors, str(adapter_path / "adapter_model.safetensors"))


def request_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 120.0,
    expect_error: bool = False,
) -> tuple[int, dict | str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        if not expect_error:
            raise
        body = error.read().decode("utf-8")
        status = error.code
    parsed: dict | str
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        return status, parsed
    parsed = body
    return status, parsed


def wait_for_health(server_url: str, timeout_s: float, process: subprocess.Popen | None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            request_json("GET", f"{server_url}/health", timeout=2.0)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {server_url}/health: {last_error}")


def start_server(args: argparse.Namespace, repo_root: Path) -> subprocess.Popen:
    assert_port_available(args.port)
    env = os.environ.copy()
    env.setdefault("PEGAINFER_CUDA_SM", "80")
    compat = "/usr/local/cuda-12.9/compat"
    if Path(compat).exists():
        old = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = compat if not old else f"{compat}:{old}"
    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "pegainfer-server",
        "--",
        "--model-path",
        args.model_path,
        "--enable-lora",
        "--served-model-name",
        "qwen3-base",
        "--tp-size",
        str(args.tp_size),
        "--port",
        str(args.port),
    ]
    log = tempfile.NamedTemporaryFile(
        prefix="pegainfer-qwen3-lora-stress-server-",
        suffix=".log",
        mode="w+",
        delete=False,
    )
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.pegainfer_log_path = log.name  # type: ignore[attr-defined]
    print(f"server_log={log.name}", file=sys.stderr)
    log.close()
    return process


def assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"port {port} is already accepting connections; stop the stale server or choose another port"
            )


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def tail_server_output(process: subprocess.Popen | None) -> str:
    if process is None:
        return ""
    log_path = getattr(process, "pegainfer_log_path", None)
    if not log_path:
        return ""
    with contextlib.suppress(Exception):
        return Path(log_path).read_text(errors="replace")[-4000:]
    return ""


def load_adapter(server_url: str, name: str, path: Path, load_inplace: bool = False) -> str:
    status, body = request_json(
        "POST",
        f"{server_url}/v1/load_lora_adapter",
        {"lora_name": name, "lora_path": str(path), "load_inplace": load_inplace},
        timeout=180.0,
    )
    if status != 200 or not isinstance(body, str):
        raise RuntimeError(f"load {name} failed: status={status}, body={body!r}")
    return body


def unload_adapter(server_url: str, name: str) -> str:
    status, body = request_json(
        "POST",
        f"{server_url}/v1/unload_lora_adapter",
        {"lora_name": name},
        timeout=180.0,
    )
    if status != 200 or not isinstance(body, str):
        raise RuntimeError(f"unload {name} failed: status={status}, body={body!r}")
    return body


def list_model_ids(server_url: str) -> list[str]:
    status, body = request_json("GET", f"{server_url}/v1/models", timeout=10.0)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"/v1/models failed: status={status}, body={body!r}")
    return sorted(entry["id"] for entry in body.get("data", []))


def completion(server_url: str, model: str, prompt: str, max_tokens: int) -> str:
    status, body = request_json(
        "POST",
        f"{server_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=180.0,
    )
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"completion failed: model={model}, status={status}, body={body!r}")
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"completion has no choices: model={model}, body={body!r}")
    return choices[0].get("text", "")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    server_url = args.server_url or f"http://127.0.0.1:{args.port}"
    process = None

    with tempfile.TemporaryDirectory(prefix="pegainfer-qwen3-lora-stress-") as tmp:
        root = Path(tmp)
        adapters = {
            "stress-a": root / "stress-a",
            "stress-b": root / "stress-b",
            "stress-c": root / "stress-c",
            "stress-b-replacement": root / "stress-b-replacement",
        }
        for index, path in enumerate(adapters.values()):
            write_adapter(model_path, path, seed_offset=index * 100, scale=0.001 + index * 0.0005)

        if args.server_url is None:
            process = start_server(args, repo_root)
        try:
            wait_for_health(server_url, args.startup_timeout_s, process)
            load_responses = {
                name: load_adapter(server_url, name, path)
                for name, path in list(adapters.items())[:3]
            }
            models_after_load = list_model_ids(server_url)
            expected_models = {"qwen3-base", "stress-a", "stress-b", "stress-c"}
            missing = expected_models.difference(models_after_load)
            if missing:
                raise RuntimeError(f"/v1/models missing loaded adapters: {sorted(missing)}")

            jobs = []
            request_models = ["qwen3-base", "stress-a", "stress-b", "stress-c"]
            for round_idx in range(args.rounds):
                for worker_idx in range(args.concurrency):
                    model = request_models[(round_idx * args.concurrency + worker_idx) % len(request_models)]
                    prompt = f"{args.prompt} round={round_idx} worker={worker_idx}"
                    jobs.append((model, prompt))

            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [
                    pool.submit(completion, server_url, model, prompt, args.max_tokens)
                    for model, prompt in jobs
                ]
                texts = [future.result() for future in concurrent.futures.as_completed(futures)]
            elapsed_s = time.monotonic() - started
            if len(texts) != len(jobs):
                raise RuntimeError(f"expected {len(jobs)} completions, got {len(texts)}")

            duplicate_status, duplicate_body = request_json(
                "POST",
                f"{server_url}/v1/load_lora_adapter",
                {"lora_name": "stress-b", "lora_path": str(adapters["stress-b"])},
                timeout=180.0,
                expect_error=True,
            )
            if duplicate_status == 200:
                raise RuntimeError("duplicate load without load_inplace unexpectedly succeeded")

            inplace_response = load_adapter(
                server_url,
                "stress-b",
                adapters["stress-b-replacement"],
                load_inplace=True,
            )
            inplace_text = completion(
                server_url,
                "stress-b",
                "After replacement, answer with a terse sentence.",
                args.max_tokens,
            )

            unload_response = unload_adapter(server_url, "stress-c")
            models_after_unload = list_model_ids(server_url)
            if "stress-c" in models_after_unload:
                raise RuntimeError("unloaded adapter still appears in /v1/models")
            unloaded_status, unloaded_body = request_json(
                "POST",
                f"{server_url}/v1/completions",
                {
                    "model": "stress-c",
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                },
                timeout=180.0,
                expect_error=True,
            )
            if unloaded_status == 200:
                raise RuntimeError("completion for unloaded adapter unexpectedly succeeded")
        except Exception:  # noqa: BLE001
            print(tail_server_output(process), file=sys.stderr)
            raise
        finally:
            stop_server(process)

    summary = {
        "server_url": server_url,
        "load_responses": load_responses,
        "models_after_load": models_after_load,
        "concurrent_requests": len(jobs),
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "elapsed_s": elapsed_s,
        "requests_per_s": len(jobs) / elapsed_s if elapsed_s > 0 else None,
        "duplicate_load_without_inplace": {
            "status": duplicate_status,
            "body": duplicate_body,
        },
        "load_inplace_response": inplace_response,
        "load_inplace_completion_text_len": len(inplace_text),
        "unload_response": unload_response,
        "models_after_unload": models_after_unload,
        "unloaded_completion": {
            "status": unloaded_status,
            "body": unloaded_body,
        },
        "ok": True,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
