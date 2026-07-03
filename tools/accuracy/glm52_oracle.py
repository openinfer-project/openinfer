#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.13.0.dev0",
#   "safetensors>=0.4",
#   "numpy",
# ]
# ///
"""GLM5.2 layer-0 oracle harness: emit hardcodable probes for the Rust gate.

The ground-truth math is transformers' own `glm_moe_dsa` module (5.13.0.dev0+,
PR #46842 — fixes indexer RoPE interleave) — this script writes NO model math
of its own. It reads the layer-0 attention weights
straight from the FP8 checkpoint, instantiates the official
`GlmMoeDsaAttention` (which includes the DSA indexer), feeds it a seeded input,
and captures intermediate tensors ("taps") via forward hooks.

What IS hand-written here, and why it is trustworthy anyway:
  * fp8 block dequant (weight * weight_scale_inv per 128x128 block) — the
    checkpoint's documented quant contract; a mistake produces garbage, not
    subtle drift.
  * `Fp8SimLinear` — precision emulation of the engine's fp8 GEMM path
    (per-128-group activation quant -> dequant -> f32 matmul), so the oracle
    lives in the same precision regime as the Rust TRTLLM kernels and the gate
    tolerance can be tight. `--precision bf16` disables it and runs the pure
    official bf16 path (looser tolerance, zero hand-written quant math) as a
    cross-check.
  * splitmix64 input generation — integer-only (no transcendentals), so Python
    and Rust produce bit-identical bf16 inputs; the emitted input digest lets
    the Rust gate fail fast on PRNG drift before touching any kernel.

Cache-fidelity emulation (fp8sim mode): the engine caches kv_c as fp8 per-128
group + f32 scales, so a hook quant-dequants the `kv_a_layernorm` output before
the official kv_b decompression consumes it — every key/value the reference
attends over has exactly the cache's precision. k_pe stays bf16 (both sides).

Floating-point probes are asserted with tolerance (fp8 vs bf16 accumulation
order makes bit-equality impossible); sha256 digests are provenance only —
never assert a float digest across implementations.

Usage (paste the emitted block into openinfer-glm52/tests/mla_decode_oracle.rs):

    uv run tools/accuracy/glm52_oracle.py \
        --model-path /data/models/GLM-5.2-FP8 --emit rust

    # full tensor dump for debugging a divergence
    uv run tools/accuracy/glm52_oracle.py \
        --model-path /data/models/GLM-5.2-FP8 --emit safetensors --out /tmp/taps.safetensors

    # negative control: the gate MUST go red against fault-injected probes
    uv run tools/accuracy/glm52_oracle.py \
        --model-path /data/models/GLM-5.2-FP8 --emit rust --inject-fault rope-swap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

FP8_GROUP = 128
E4M3_MAX = 448.0
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Cross-language deterministic input (mirror of the Rust test's generator).
# splitmix64 -> 53-bit uniform -> (u - 0.5) * 4.0, all exact in f64, then
# f32 -> bf16 round-to-nearest-even. No transcendentals anywhere, so both
# languages produce bit-identical bf16.
# ---------------------------------------------------------------------------


def splitmix64_uniform(seed: int, count: int) -> np.ndarray:
    state = np.uint64(seed)
    out = np.empty(count, dtype=np.float64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    m1 = np.uint64(0xBF58476D1CE4E5B9)
    m2 = np.uint64(0x94D049BB133111EB)
    states = (np.arange(1, count + 1, dtype=np.uint64) * golden) + state
    with np.errstate(over="ignore"):
        z = states
        z = (z ^ (z >> np.uint64(30))) * m1
        z = (z ^ (z >> np.uint64(27))) * m2
        z = z ^ (z >> np.uint64(31))
    out = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return out


def seeded_hidden(seed: int, ctx: int, hidden: int) -> torch.Tensor:
    u = splitmix64_uniform(seed, ctx * hidden)
    x = ((u - 0.5) * 4.0).astype(np.float32)
    return torch.from_numpy(x).reshape(ctx, hidden).to(torch.bfloat16)


def bf16_digest(t: torch.Tensor) -> str:
    raw = t.contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def i32_digest(t: torch.Tensor) -> str:
    raw = t.contiguous().to(torch.int32).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Checkpoint access + fp8 block dequant
# ---------------------------------------------------------------------------


class Checkpoint:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._open_shards: dict[str, object] = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open_shards:
            self._open_shards[shard] = safe_open(
                self.model_path / shard, framework="pt", device="cpu"
            )
        return self._open_shards[shard].get_tensor(name)


def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 [n,k] * per-128x128-block f32 scale -> f32 [n,k]."""
    n, k = weight.shape
    sn, sk = scale_inv.shape
    assert sn == -(-n // FP8_GROUP) and sk == -(-k // FP8_GROUP), (
        f"scale shape {scale_inv.shape} does not match weight {weight.shape}"
    )
    scale_full = scale_inv.repeat_interleave(FP8_GROUP, 0).repeat_interleave(FP8_GROUP, 1)
    return weight.to(torch.float32) * scale_full[:n, :k]


def quant_dequant_groups(x: torch.Tensor) -> torch.Tensor:
    """Per-128-group fp8 e4m3 quant->dequant along the last dim (engine's
    activation-quant contract: scale = group amax / 448, f32 scale)."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    shape = xf.shape
    assert shape[-1] % FP8_GROUP == 0, f"width {shape[-1]} not a multiple of {FP8_GROUP}"
    g = xf.reshape(-1, shape[-1] // FP8_GROUP, FP8_GROUP)
    scale = (g.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
    dq = (g / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale
    return dq.reshape(shape).to(orig_dtype)


class Fp8SimLinear(torch.nn.Module):
    """Engine fp8 GEMM precision emulation: quant-dequant the activation per
    128-group, f32 matmul against the block-dequantized weight, bf16 out."""

    def __init__(self, weight_f32: torch.Tensor):
        super().__init__()
        self.register_buffer("w", weight_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xdq = quant_dequant_groups(x).to(torch.float32)
        return torch.nn.functional.linear(xdq, self.w).to(torch.bfloat16)


class Bf16Linear(torch.nn.Module):
    """Plain bf16 linear on dequantized weights (the `--precision bf16` path,
    and the kv_b decompression in both modes — the engine holds kv_b's absorb
    factors as bf16, not fp8)."""

    def __init__(self, weight_f32: torch.Tensor):
        super().__init__()
        self.register_buffer("w", weight_f32.to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.w)


# ---------------------------------------------------------------------------
# Model assembly from the official transformers module
# ---------------------------------------------------------------------------


def load_config(model_path: Path):
    from transformers.models.glm_moe_dsa import GlmMoeDsaConfig

    raw = json.loads((model_path / "config.json").read_text())
    known = set(GlmMoeDsaConfig.__dataclass_fields__)
    # __post_init__ consumes these via kwargs to derive indexer_types.
    passthrough = {"index_topk_pattern", "index_topk_freq", "index_skip_topk_offset"}
    filtered = {k: v for k, v in raw.items() if k in known or k in passthrough}
    config = GlmMoeDsaConfig(**filtered)
    config._attn_implementation = "eager"
    return config


def patch_attention(attn, ckpt: Checkpoint, layer: int, precision: str, fault: str):
    """Swap an official `GlmMoeDsaAttention`'s linears for checkpoint-loaded,
    precision-emulated ones. Shared by the bare-attention and decoder-layer
    builders."""
    prefix = f"model.layers.{layer}.self_attn"

    def proj(stem: str) -> torch.Tensor:
        w = dequant_fp8_block(
            ckpt.tensor(f"{prefix}.{stem}.weight"),
            ckpt.tensor(f"{prefix}.{stem}.weight_scale_inv"),
        )
        if fault == "qb-head-negate" and stem == "q_b_proj":
            # Negate head 0's full q_b slice (rows 0..256). A single 128x128
            # block is NOT enough: softmax smoothing + o_proj's 64-head mixing
            # dilute it below the gate tolerance (measured 2026-07-02).
            w[0:256, :] *= -1.0
            print("// !!! FAULT INJECTED: q_b head-0 negated — DO NOT COMMIT", file=sys.stderr)
        return w

    linear_cls = Fp8SimLinear if precision == "fp8sim" else Bf16Linear
    attn.q_a_proj = linear_cls(proj("q_a_proj"))
    attn.q_b_proj = linear_cls(proj("q_b_proj"))
    attn.kv_a_proj_with_mqa = linear_cls(proj("kv_a_proj_with_mqa"))
    attn.o_proj = linear_cls(proj("o_proj"))
    # kv_b is bf16 in the engine (host-dequanted absorb factors), never fp8 GEMM.
    attn.kv_b_proj = Bf16Linear(proj("kv_b_proj"))

    attn.q_a_layernorm.weight.data = ckpt.tensor(f"{prefix}.q_a_layernorm.weight").to(torch.bfloat16)
    attn.kv_a_layernorm.weight.data = ckpt.tensor(f"{prefix}.kv_a_layernorm.weight").to(torch.bfloat16)

    if attn.indexer is not None:
        attn.indexer.wq_b = linear_cls(proj("indexer.wq_b"))
        attn.indexer.wk = linear_cls(proj("indexer.wk"))
        attn.indexer.k_norm.weight.data = ckpt.tensor(f"{prefix}.indexer.k_norm.weight").to(torch.bfloat16)
        attn.indexer.k_norm.bias.data = ckpt.tensor(f"{prefix}.indexer.k_norm.bias").to(torch.bfloat16)
        # transformers keeps indexer.weights_proj in fp32 (_keep_in_fp32_modules).
        attn.indexer.weights_proj.weight.data = ckpt.tensor(
            f"{prefix}.indexer.weights_proj.weight"
        ).to(torch.float32)
        attn.indexer.weights_proj = attn.indexer.weights_proj.to(torch.float32)
    return attn


def build_attention(ckpt: Checkpoint, config, layer: int, precision: str, fault: str):
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

    assert config.indexer_types[layer] == "full", (
        f"layer {layer} is a shared-indexer layer; pick a 'full' layer for the oracle"
    )
    attn = GlmMoeDsaAttention(config, layer_idx=layer).to(torch.bfloat16).eval()
    return patch_attention(attn, ckpt, layer, precision, fault)


class Fp8SimExperts(torch.nn.Module):
    """Routed-experts forward with the engine's fp8 precision regime: the token
    activation is quant-dequanted per 128-group before the gate|up GEMM, and the
    SwiGLU product before the down GEMM (the engine's W2-input re-quant; the
    route weight commutes with the per-group dynamic scale, so applying it after
    down as the official code does is equivalent up to f32 scale rounding).

    The loop itself mirrors `GlmMoeDsaExperts.forward` (transport, no new math);
    expert weights stay fp8 in memory and are block-dequanted lazily per hit
    expert, so the 256-expert layer never materializes ~19 GB of f32 weights.
    """

    def __init__(self, ckpt: Checkpoint, layer: int, num_experts: int, precision: str):
        super().__init__()
        self.ckpt = ckpt
        self.prefix = f"model.layers.{layer}.mlp.experts"
        self.num_experts = num_experts
        self.precision = precision
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _expert(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx not in self._cache:
            def dq(stem: str) -> torch.Tensor:
                return dequant_fp8_block(
                    self.ckpt.tensor(f"{self.prefix}.{idx}.{stem}.weight"),
                    self.ckpt.tensor(f"{self.prefix}.{idx}.{stem}.weight_scale_inv"),
                )

            gate_up = torch.cat([dq("gate_proj"), dq("up_proj")], dim=0)  # [2I, H]
            self._cache[idx] = (gate_up, dq("down_proj"))
        return self._cache[idx]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        if self.precision == "fp8sim":
            hidden_states = quant_dequant_groups(hidden_states)
        act = hidden_states.to(torch.float32)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in expert_mask.sum(dim=(-1, -2)).nonzero()[:, 0].tolist():
            gate_up, down = self._expert(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            gate, up = torch.nn.functional.linear(act[token_idx], gate_up).chunk(2, dim=-1)
            prod = torch.nn.functional.silu(gate) * up
            if self.precision == "fp8sim":
                prod = quant_dequant_groups(prod.to(torch.bfloat16)).to(torch.float32)
            out = torch.nn.functional.linear(prod, down)
            out = out * top_k_weights[token_idx, top_k_pos, None].to(torch.float32)
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


def build_decoder_layer(ckpt: Checkpoint, config, layer: int, precision: str, fault: str):
    """Official `GlmMoeDsaDecoderLayer` with checkpoint weights + fp8sim linears:
    the full-layer oracle (attention + norms + residuals + dense-or-MoE MLP)."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer,
        GlmMoeDsaMLP,
        GlmMoeDsaMoE,
    )

    assert config.indexer_types[layer] == "full", (
        f"layer {layer} is a shared-indexer layer; pick a 'full' layer for the oracle"
    )
    dl = GlmMoeDsaDecoderLayer(config, layer_idx=layer).to(torch.bfloat16).eval()
    patch_attention(dl.self_attn, ckpt, layer, precision, fault)
    prefix = f"model.layers.{layer}"
    dl.input_layernorm.weight.data = ckpt.tensor(f"{prefix}.input_layernorm.weight").to(torch.bfloat16)
    dl.post_attention_layernorm.weight.data = ckpt.tensor(
        f"{prefix}.post_attention_layernorm.weight"
    ).to(torch.bfloat16)

    linear_cls = Fp8SimLinear if precision == "fp8sim" else Bf16Linear

    def mlp_proj(module: GlmMoeDsaMLP, stem_prefix: str):
        for stem in ("gate_proj", "up_proj", "down_proj"):
            w = dequant_fp8_block(
                ckpt.tensor(f"{stem_prefix}.{stem}.weight"),
                ckpt.tensor(f"{stem_prefix}.{stem}.weight_scale_inv"),
            )
            setattr(module, stem, linear_cls(w))

    if isinstance(dl.mlp, GlmMoeDsaMoE):
        dl.mlp.gate.weight.data = ckpt.tensor(f"{prefix}.mlp.gate.weight").to(torch.bfloat16)
        dl.mlp.gate.e_score_correction_bias.data = ckpt.tensor(
            f"{prefix}.mlp.gate.e_score_correction_bias"
        ).to(torch.float32)
        dl.mlp.experts = Fp8SimExperts(ckpt, layer, config.n_routed_experts, precision)
        mlp_proj(dl.mlp.shared_experts, f"{prefix}.mlp.shared_experts")
    else:
        mlp_proj(dl.mlp, f"{prefix}.mlp")
    return dl


# ---------------------------------------------------------------------------
# Forward + taps
# ---------------------------------------------------------------------------


def run(attn, config, hidden: torch.Tensor, precision: str, fault: str) -> dict[str, torch.Tensor]:
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding

    taps: dict[str, torch.Tensor] = {"hidden": hidden}
    ctx = hidden.shape[0]
    hidden_b = hidden.unsqueeze(0)
    position_ids = torch.arange(ctx).unsqueeze(0)

    rope = GlmMoeDsaRotaryEmbedding(config)
    cos, sin = rope(hidden_b, position_ids)
    if fault == "rope-swap":
        cos, sin = sin, cos
        print("// !!! FAULT INJECTED: cos/sin swapped — DO NOT COMMIT", file=sys.stderr)
    taps["cos"], taps["sin"] = cos.squeeze(0), sin.squeeze(0)

    def record(name):
        def hook(_mod, _args, out):
            taps[name] = out.detach().squeeze(0)

        return hook

    attn.q_a_layernorm.register_forward_hook(record("q_resid"))
    attn.q_b_proj.register_forward_hook(record("q_full"))
    attn.kv_a_proj_with_mqa.register_forward_hook(record("ckv"))

    def kv_c_hook(_mod, _args, out):
        # Engine cache fidelity: kv_c lives in the paged cache as fp8 per-128
        # group; everything downstream (kv_b decompress -> K/V) must see the
        # quantized value, exactly like FlashMLA reading the cache.
        if precision == "fp8sim":
            out = quant_dequant_groups(out)
        taps["kv_c_cached"] = out.detach().squeeze(0)
        return out

    attn.kv_a_layernorm.register_forward_hook(kv_c_hook)
    attn.o_proj.register_forward_pre_hook(
        lambda _mod, args: taps.__setitem__("attn_v", args[0].detach().squeeze(0))
    )

    with torch.no_grad():
        o, _attn_weights, topk = attn(
            hidden_states=hidden_b,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
    taps["o"] = o.squeeze(0)
    taps["topk_indices"] = topk.squeeze(0)
    return taps


def run_layer(dl, config, hidden: torch.Tensor, fault: str) -> dict[str, torch.Tensor]:
    """Full decoder-layer forward (norms + attention + residuals + MLP/MoE)."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMoE,
        GlmMoeDsaRotaryEmbedding,
    )

    taps: dict[str, torch.Tensor] = {"hidden": hidden}
    ctx = hidden.shape[0]
    hidden_b = hidden.unsqueeze(0)
    position_ids = torch.arange(ctx).unsqueeze(0)

    rope = GlmMoeDsaRotaryEmbedding(config)
    cos, sin = rope(hidden_b, position_ids)
    if fault == "rope-swap":
        cos, sin = sin, cos
        print("// !!! FAULT INJECTED: cos/sin swapped — DO NOT COMMIT", file=sys.stderr)
    taps["cos"], taps["sin"] = cos.squeeze(0), sin.squeeze(0)

    if isinstance(dl.mlp, GlmMoeDsaMoE):
        def gate_hook(_mod, _args, out):
            router_logits, topk_weights, topk_indices = out
            taps["router_logits"] = router_logits.detach()
            taps["router_topk_weights"] = topk_weights.detach()
            taps["router_topk_indices"] = topk_indices.detach().to(torch.int32)

        dl.mlp.gate.register_forward_hook(gate_hook)

    with torch.no_grad():
        out, dsa_topk = dl(
            hidden_states=hidden_b,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
    taps["layer_out"] = out.squeeze(0)
    taps["topk_indices"] = dsa_topk.squeeze(0)
    return taps


def seeded_token_ids(seed: int, ctx: int, vocab: int) -> torch.Tensor:
    """splitmix64-derived token ids (integer-only, bit-identical across languages)."""
    u = splitmix64_uniform(seed ^ 0xB00C, ctx)
    ids = (u * vocab).astype(np.int64).clip(0, vocab - 1)
    return torch.from_numpy(ids)


def run_bookend(ckpt: Checkpoint, config, hidden: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    """Bookends: embed gather (exact), final RMSNorm + lm_head logits + argmax.
    The norm math is the official `GlmMoeDsaRMSNorm`; embed/lm_head are plain
    bf16 gather/linear (no fp8 anywhere in the bookends)."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm

    ctx = hidden.shape[0]
    taps: dict[str, torch.Tensor] = {"hidden": hidden}

    embed = ckpt.tensor("model.embed_tokens.weight")  # bf16 [V, H]
    token_ids = seeded_token_ids(seed, ctx, config.vocab_size)
    taps["token_ids"] = token_ids.to(torch.int32)
    taps["embed_rows"] = embed[token_ids]  # exact gather, digest-assertable

    norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16).eval()
    norm.weight.data = ckpt.tensor("model.norm.weight").to(torch.bfloat16)
    lm_head = ckpt.tensor("lm_head.weight")  # bf16 [V, H]
    with torch.no_grad():
        normed = norm(hidden.unsqueeze(0)).squeeze(0)
        logits = torch.nn.functional.linear(normed.to(torch.float32), lm_head.to(torch.float32))
    taps["normed"] = normed
    taps["logits"] = logits.to(torch.bfloat16)
    taps["argmax"] = logits.argmax(dim=-1).to(torch.int32)
    return taps


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit_rust(taps, args, versions: str) -> str:
    hidden, o = taps["hidden"], taps["o"]
    flat = o.to(torch.float32).flatten()
    rms = float(flat.square().mean().sqrt())
    rng = random.Random(f"{args.seed}:o")
    idxs = sorted(rng.sample(range(flat.numel()), args.probes))
    probes = ",\n".join(f"    ({i}, {flat[i].item():.9e})" for i in idxs)
    fault_banner = (
        f"// !!! FAULT-INJECTED ({args.inject_fault}) — negative control only, DO NOT COMMIT\n"
        if args.inject_fault != "none"
        else ""
    )
    return f"""\
{fault_banner}// ---- BEGIN GENERATED: glm52_oracle probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path {args.model_path} \\
//     --ctx {args.ctx} --seed {args.seed:#x} --layer {args.layer} --precision {args.precision}
// {versions}
const ORACLE_SEED: u64 = {args.seed:#x};
const ORACLE_CTX: usize = {args.ctx};
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "{bf16_digest(hidden)}";
// tap `o` [{args.ctx}, 6144] bf16 digest={bf16_digest(o)} (provenance only, never assert)
const ORACLE_O_RMS: f32 = {rms:.9e};
const ORACLE_O_REL_TOL: f32 = {args.rel_tol};
const ORACLE_O_PROBES: &[(usize, f32)] = &[
{probes},
];
// ---- END GENERATED ----"""


def emit_rust_indexer(taps, args, versions: str) -> str:
    """Emit topk_indices as a sorted set for the Rust overlap gate.

    The indexer gate asserts set-overlap >= 2047/2048 (allow 1 tie-break
    divergence between FlashInfer top-k and torch.topk).
    Also emits q_resid, cos, sin, hidden digests so the Rust side can
    verify input fidelity before running the indexer forward.
    """
    hidden = taps["hidden"]
    topk = taps["topk_indices"]  # [ctx, 2048] or [2048] for bs=1
    if topk.dim() == 2:
        topk = topk[-1]  # bs=1 → last position [2048]
    topk_set = sorted(int(v) for v in topk.tolist() if v >= 0)
    q_resid = taps.get("q_resid")
    cos = taps.get("cos")
    sin = taps.get("sin")

    fault_banner = (
        f"// !!! FAULT-INJECTED ({args.inject_fault}) — negative control only, DO NOT COMMIT\n"
        if args.inject_fault != "none"
        else ""
    )
    q_digest = bf16_digest(q_resid) if q_resid is not None else "N/A"
    cos_digest = bf16_digest(cos) if cos is not None else "N/A"
    sin_digest = bf16_digest(sin) if sin is not None else "N/A"
    hidden_digest = bf16_digest(hidden)
    topk_digest = i32_digest(topk)

    return f"""\
{fault_banner}// ---- BEGIN GENERATED: glm52_oracle indexer probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path {args.model_path} \\
//     --ctx {args.ctx} --seed {args.seed:#x} --layer {args.layer} --precision {args.precision} --stage indexer
// {versions}
const ORACLE_SEED: u64 = {args.seed:#x};
const ORACLE_CTX: usize = {args.ctx};
// Input digests — verify before running the indexer forward.
const ORACLE_HIDDEN_DIGEST: &str = "{hidden_digest}";
const ORACLE_Q_RESID_DIGEST: &str = "{q_digest}";
const ORACLE_COS_DIGEST: &str = "{cos_digest}";
const ORACLE_SIN_DIGEST: &str = "{sin_digest}";
// topk_indices [2048] i32 (provenance only — the gate asserts set-overlap, not bit-equality)
const ORACLE_TOPK_DIGEST: &str = "{topk_digest}";
const ORACLE_TOPK_SET: &[i32] = &{topk_set};
// ---- END GENERATED ----"""


def sampled_probes(tap: torch.Tensor, args, salt: str) -> tuple[float, str]:
    """Sample `args.probes` flat indices from a float tap; returns (rms, rust rows)."""
    flat = tap.to(torch.float32).flatten()
    rms = float(flat.square().mean().sqrt())
    rng = random.Random(f"{args.seed}:{salt}")
    idxs = sorted(rng.sample(range(flat.numel()), args.probes))
    rows = ",\n".join(f"    ({i}, {flat[i].item():.9e})" for i in idxs)
    return rms, rows


def emit_rust_layer(taps, args, versions: str) -> str:
    """Full decoder-layer probes (+ router top-k of the last position on MoE
    layers). The Rust gate replays the same input through
    `glm52_decoder_layer_forward` position by position and asserts `layer_out`."""
    hidden = taps["hidden"]
    rms, probes = sampled_probes(taps["layer_out"], args, "layer_out")
    fault_banner = (
        f"// !!! FAULT-INJECTED ({args.inject_fault}) — negative control only, DO NOT COMMIT\n"
        if args.inject_fault != "none"
        else ""
    )
    router_block = ""
    if "router_topk_indices" in taps:
        ids = taps["router_topk_indices"][-1]
        weights = taps["router_topk_weights"][-1].to(torch.float32)
        pairs = sorted(zip(ids.tolist(), weights.tolist()))
        rows = ",\n".join(f"    ({i}, {w:.9e})" for i, w in pairs)
        router_block = f"""
// Router selection of the LAST position: (expert_id, normalized x2.5 weight),
// sorted by expert id. Ids assert exactly; weights within REL_TOL of their max.
const ORACLE_ROUTER_LAST: &[(i32, f32)] = &[
{rows},
];"""
    return f"""\
{fault_banner}// ---- BEGIN GENERATED: glm52_oracle layer probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path {args.model_path} \\
//     --ctx {args.ctx} --seed {args.seed:#x} --layer {args.layer} --precision {args.precision} --stage layer
// {versions}
const ORACLE_SEED: u64 = {args.seed:#x};
const ORACLE_CTX: usize = {args.ctx};
const ORACLE_LAYER: usize = {args.layer};
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "{bf16_digest(hidden)}";
// tap `layer_out` [{args.ctx}, 6144] bf16 digest={bf16_digest(taps["layer_out"])} (provenance only)
const ORACLE_LAYER_RMS: f32 = {rms:.9e};
const ORACLE_LAYER_REL_TOL: f32 = {args.rel_tol};
const ORACLE_LAYER_PROBES: &[(usize, f32)] = &[
{probes},
];{router_block}
// ---- END GENERATED ----"""


def emit_rust_bookend(taps, args, versions: str) -> str:
    """Bookend probes: embed rows digest (exact — a bf16 gather), logits probes,
    and the per-position argmax (exact — the thing greedy decode consumes)."""
    rms, probes = sampled_probes(taps["logits"], args, "logits")
    argmax = ", ".join(str(int(v)) for v in taps["argmax"].tolist())
    return f"""\
// ---- BEGIN GENERATED: glm52_oracle bookend probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path {args.model_path} \\
//     --ctx {args.ctx} --seed {args.seed:#x} --precision {args.precision} --stage bookend
// {versions}
const ORACLE_SEED: u64 = {args.seed:#x};
const ORACLE_CTX: usize = {args.ctx};
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "{bf16_digest(taps["hidden"])}";
// Token-id gather is exact: assert the embed-rows digest bit-for-bit.
const ORACLE_EMBED_ROWS_DIGEST: &str = "{bf16_digest(taps["embed_rows"])}";
const ORACLE_LOGITS_RMS: f32 = {rms:.9e};
const ORACLE_LOGITS_REL_TOL: f32 = {args.rel_tol};
const ORACLE_LOGITS_PROBES: &[(usize, f32)] = &[
{probes},
];
// Greedy argmax per position — exact.
const ORACLE_ARGMAX: &[u32] = &[{argmax}];
// ---- END GENERATED ----"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=200, help="positions; <=2048 so full top-k == DSA")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED_604D)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--precision", choices=["fp8sim", "bf16"], default="fp8sim")
    p.add_argument("--emit", choices=["rust", "safetensors", "both"], default="rust")
    p.add_argument("--out", type=Path, help="safetensors dump path")
    p.add_argument("--probes", type=int, default=64)
    p.add_argument("--rel-tol", type=float, default=0.05)
    p.add_argument("--inject-fault", choices=["none", "qb-head-negate", "rope-swap"], default="none")
    p.add_argument("--stage", choices=["mla", "indexer", "layer", "bookend"], default="mla",
                   help="mla: MLA o-probes. indexer: topk set for the overlap gate. "
                   "layer: full decoder-layer probes (dense layer 0 / MoE layer 6). "
                   "bookend: embed/final-norm/lm_head probes.")
    args = p.parse_args()

    assert args.ctx <= 2048 or args.stage in ("indexer",), "full top-k == DSA only holds at ctx <= 2048"

    import transformers

    versions = f"transformers={transformers.__version__} torch={torch.__version__}"
    config = load_config(args.model_path)
    ckpt = Checkpoint(args.model_path)
    hidden = seeded_hidden(args.seed, args.ctx, config.hidden_size)

    if args.stage == "layer":
        dl = build_decoder_layer(ckpt, config, args.layer, args.precision, args.inject_fault)
        taps = run_layer(dl, config, hidden, args.inject_fault)
    elif args.stage == "bookend":
        taps = run_bookend(ckpt, config, hidden, args.seed)
    else:
        attn = build_attention(ckpt, config, args.layer, args.precision, args.inject_fault)
        taps = run(attn, config, hidden, args.precision, args.inject_fault)

    for name, t in taps.items():
        digest = i32_digest(t) if t.dtype == torch.int32 else bf16_digest(t.to(torch.bfloat16))
        rms = float(t.to(torch.float32).square().mean().sqrt())
        print(f"// tap {name:<12} shape={tuple(t.shape)} rms={rms:.6f} digest={digest}", file=sys.stderr)

    if args.emit in ("rust", "both"):
        if args.stage == "indexer":
            print(emit_rust_indexer(taps, args, versions))
        elif args.stage == "layer":
            print(emit_rust_layer(taps, args, versions))
        elif args.stage == "bookend":
            print(emit_rust_bookend(taps, args, versions))
        else:
            print(emit_rust(taps, args, versions))
    if args.emit in ("safetensors", "both"):
        assert args.out, "--out required for safetensors emission"
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in taps.items()}, args.out)
        print(f"// taps written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
