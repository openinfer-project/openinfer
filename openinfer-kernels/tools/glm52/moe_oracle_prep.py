#!/usr/bin/env python
"""Build the GLM5.2 MoE (FFN) full-precision oracle for the first sparse layer
(layer 3), the ground truth a Rust GPU test validates the MoE forward against.
Mirror of the MLA oracle (`/data/models/glm52_mla_ref/layer0.npz`): torch CPU,
float32, `torch.manual_seed(0)`, input `hidden = randn(1, 8, 6144) * 0.1`.

We import HF's OWN modules so the oracle is authoritative: `GlmMoeDsaMoE` (its
`gate` / `route_tokens_to_experts` / `experts` / `shared_experts`) runs the real
routing + expert math. The eager NaiveMoe expert loop is forced via
`config._experts_implementation = "eager"` (no GPU/triton kernel dispatch).

Run on the H200 build node (torch CPU is enough; weights stay fp8 on disk and are
dequantized here):

    export https_proxy=http://127.0.0.1:1083 http_proxy=http://127.0.0.1:1083
    uv run --no-project --with torch --with numpy --with safetensors \
        --with ml_dtypes --with transformers==5.12.0 python -u \
        openinfer-kernels/tools/glm52/moe_oracle_prep.py

GLM5.2 MoE facts honored (verified against config.json):
  256 routed experts, top-8, sigmoid scoring + noaux_tc (n_group=1, topk_group=1
  -> group masking is a no-op, flat top-8 over biased scores). The 8 selected
  weights are the UNBIASED sigmoid scores gathered at the top-8 indices, then
  normalized to sum 1 (norm_topk_prob=true) and multiplied by routed_scaling_factor
  = 2.5 (so each row sums to 2.5). moe_intermediate_size=2048, one shared expert
  (intermediate 2048). final moe_output = routed_output + shared_output.

fp8 dequant matches the MLA oracle convention: 128x128 multiplicative block scale
(`weight_scale_inv`), CEIL block count for partial last blocks. Layer-3 experts are
128-exact in both dims (2048/6144), so no partial-block subtlety here, but we
ceil-divide anyway for parity. Router gate.weight is bf16 (it is in the
checkpoint's `modules_to_not_convert`), e_score_correction_bias is f32 -- neither
is fp8.
"""

import json
import os
import struct

import ml_dtypes
import numpy as np
import torch

MODEL = os.environ.get("GLM52_MODEL", "/data/models/GLM-5.2-FP8")
OUT = os.environ.get("GLM52_MOE_ORACLE", "/data/models/glm52_mla_ref/moe_layer3.npz")
LAYER = 3  # first sparse MoE layer (first_k_dense_replace=3 -> layers 0..2 dense)
T = 7  # spotlight token, mirroring the MLA oracle (validates token 7)
P = f"model.layers.{LAYER}.mlp."

IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def load_raw(name):
    shard = IDX[name]
    with open(f"{MODEL}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        f.seek(8 + n + meta["data_offsets"][0])
        buf = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return buf, meta["shape"], meta["dtype"]


def deq_fp8(name):
    """fp8 e4m3 weight * 128x128 block scale (multiplicative, ceil block count)."""
    wb, ws, dt = load_raw(name)
    assert dt == "F8_E4M3", f"{name} expected F8_E4M3 got {dt}"
    sb, _, sdt = load_raw(name + "_scale_inv")
    assert sdt == "F32"
    w = np.frombuffer(wb, ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(ws)
    bn, bk = (ws[0] + 127) // 128, (ws[1] + 127) // 128
    s = np.frombuffer(sb, np.float32).reshape(bn, bk)
    sfull = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[: ws[0], : ws[1]]
    return w * sfull


def plain(name, np_dtype):
    b, sh, _ = load_raw(name)
    return np.frombuffer(b, np_dtype).astype(np.float32).reshape(sh)


def main():
    torch.set_num_threads(min(64, os.cpu_count() or 8))
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE

    cfg = GlmMoeDsaConfig(**json.load(open(f"{MODEL}/config.json")))
    cfg._experts_implementation = "eager"  # force the naive eager expert loop on CPU
    assert cfg.num_local_experts == 256 and cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 2048 and cfg.hidden_size == 6144
    assert cfg.norm_topk_prob is True and abs(cfg.routed_scaling_factor - 2.5) < 1e-9
    assert cfg.n_group == 1 and cfg.topk_group == 1

    torch.manual_seed(0)
    hidden = (torch.randn(1, 8, 6144) * 0.1).to(torch.float32)

    print("building GlmMoeDsaMoE + loading dequantized weights ...", flush=True)
    moe = GlmMoeDsaMoE(cfg).to(torch.float32).eval()

    # router (bf16 gate.weight, f32 bias -- neither is fp8)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.from_numpy(plain(P + "gate.weight", ml_dtypes.bfloat16)))
        moe.gate.e_score_correction_bias.copy_(
            torch.from_numpy(plain(P + "gate.e_score_correction_bias", np.float32))
        )

        # routed experts: gate_up_proj[e] = [gate_proj; up_proj] (rows 0:2048 gate,
        # 2048:4096 up -- forward does linear(x, w).chunk(2,-1)); down_proj[e] = down.
        for e in range(cfg.num_local_experts):
            g = deq_fp8(f"{P}experts.{e}.gate_proj.weight")
            u = deq_fp8(f"{P}experts.{e}.up_proj.weight")
            d = deq_fp8(f"{P}experts.{e}.down_proj.weight")
            moe.experts.gate_up_proj.data[e].copy_(torch.from_numpy(np.concatenate([g, u], axis=0)))
            moe.experts.down_proj.data[e].copy_(torch.from_numpy(d))
            del g, u, d
            if (e + 1) % 64 == 0:
                print(f"  loaded {e + 1}/{cfg.num_local_experts} experts", flush=True)

        # shared expert
        moe.shared_experts.gate_proj.weight.copy_(torch.from_numpy(deq_fp8(P + "shared_experts.gate_proj.weight")))
        moe.shared_experts.up_proj.weight.copy_(torch.from_numpy(deq_fp8(P + "shared_experts.up_proj.weight")))
        moe.shared_experts.down_proj.weight.copy_(torch.from_numpy(deq_fp8(P + "shared_experts.down_proj.weight")))

        # --- authoritative HF forward, decomposed to capture intermediates ---
        gate_logits = moe.gate(hidden)  # [8,256] raw gate projection (pre-sigmoid)
        router_probs = gate_logits.sigmoid()  # [8,256] post-sigmoid, pre-bias
        topk_indices, topk_weights = moe.route_tokens_to_experts(gate_logits)  # [8,8], [8,8]
        flat = hidden.view(-1, hidden.shape[-1])  # [8,6144]
        routed = moe.experts(flat, topk_indices, topk_weights)  # [8,6144]
        shared = moe.shared_experts(flat)  # [8,6144]
        moe_full = moe.forward(hidden).view(8, 6144)  # authoritative end-to-end [8,6144]

    # consistency: decomposed pieces must reconstruct the authoritative forward
    recon = routed + shared
    assert torch.allclose(moe_full, recon, atol=1e-4, rtol=0), (
        f"decompose mismatch max|d|={(moe_full - recon).abs().max().item()}"
    )

    hidden_np = hidden.numpy().astype(np.float32)
    router_logits_np = router_probs.numpy().astype(np.float32)  # post-sigmoid, pre-bias
    gate_logits_np = gate_logits.numpy().astype(np.float32)  # raw, pre-sigmoid
    topk_idx_np = topk_indices.numpy().astype(np.int32)
    topk_w_np = topk_weights.numpy().astype(np.float32)
    shared_np = shared.numpy().astype(np.float32)
    routed_np = routed.numpy().astype(np.float32)
    moe_np = moe_full.numpy().astype(np.float32)
    bias_np = moe.gate.e_score_correction_bias.numpy().astype(np.float32)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        hidden=hidden_np,  # [1,8,6144] MoE block input
        router_logits=router_logits_np,  # [8,256] sigmoid(gate@h), pre-bias
        gate_logits=gate_logits_np,  # [8,256] raw gate projection (pre-sigmoid)
        e_score_correction_bias=bias_np,  # [256] additive routing bias (f32)
        topk_indices=topk_idx_np,  # [8,8] selected expert ids per token
        topk_weights=topk_w_np,  # [8,8] normalized sigmoid weights * 2.5
        shared_output=shared_np,  # [8,6144]
        routed_output=routed_np,  # [8,6144]
        moe_output=moe_np,  # [8,6144] = routed + shared
    )

    # ---- report ----
    z = np.load(OUT)
    print(f"\nnpz -> {OUT}")
    for k in z.files:
        print(f"  {k:24s} {tuple(z[k].shape)} {z[k].dtype}")
    sel = topk_idx_np[T].tolist()
    print(f"\ntoken {T} selected expert ids: {sel}")
    print(f"token {T} topk_weights:        {np.round(topk_w_np[T], 5).tolist()}")
    rowsum = topk_w_np.sum(axis=1)
    print(f"topk_weights row-sums (want ~2.5): min={rowsum.min():.6f} max={rowsum.max():.6f}")
    assert np.allclose(rowsum, 2.5, atol=1e-4), "row-sum != 2.5"
    print(
        f"\n|moe_output[{T}]|   max={np.abs(moe_np[T]).max():.5f} mean={np.abs(moe_np[T]).mean():.5f}"
    )
    print(
        f"|shared_output[{T}]| max={np.abs(shared_np[T]).max():.5f} mean={np.abs(shared_np[T]).mean():.5f}"
    )
    print(
        f"|routed_output[{T}]| max={np.abs(routed_np[T]).max():.5f} mean={np.abs(routed_np[T]).mean():.5f}"
    )
    print(f"decompose check (routed+shared vs forward) max|d|={(moe_full - recon).abs().max().item():.3e}")
    print("OK")


if __name__ == "__main__":
    main()
