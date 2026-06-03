"""
Kimi-K2 TP1.DP8.EP8 decode -- TPOT(B) cost decomposition.

WHERE does the per-step time grow as batch B increases?

This is a THEORY-DRIVEN MOCK. Magnitudes are illustrative (loosely anchored to
the B=8 / ctx=1 ratios in tp1-dp8-ep8-decode-optimization-master.md); only the
SHAPES are the message. Real numbers wait for an H20 sweep of --active-rows.

Each component carries its theoretical B-scaling law:
  - BF16 GEMM weight reads : flat while memory-bound (B<ridge), then ~B (compute)
  - MoE routed INT4 reads  : coupon-collector ramp in #active experts, then ~B
  - activation / control   : ~B (norm, swiglu, residual, rope, router-topk, ...)
  - MLA attention KV       : ~B*ctx  (the only ctx-coupled term)
  - EP all-to-all          : latency floor + ~B
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- illustrative constants (MOCK; not measured) -------------------------
DP = 8                      # data-parallel ranks -> global batch = DP * B
N_EXPERTS = 384
TOPK = 8
RIDGE_B = 30.83             # H20 BF16 ridge = 148 TFLOP/s / 4.8 TB/s
INT4_RIDGE = RIDGE_B / 4.0  # INT4 weights -> 4x AI -> crossover at ridge/4

B = np.arange(1, 65)        # per-rank batch (decode-realistic + a little WideEP headroom)
Bg = DP * B                 # global batch

# coupon-collector: expected distinct experts touched by Bg tokens (each picks TOPK)
p_miss = 1.0 - TOPK / N_EXPERTS
E_active_global = N_EXPERTS * (1.0 - p_miss ** Bg)
E_active_local = E_active_global / DP                       # experts owned per rank (EP8)
rows_per_expert = (Bg * TOPK) / np.maximum(E_active_global, 1.0)


def gemm_base(B):
    floor = 9300.0                                   # BF16 weight-read time, ~master B=8 sum
    return floor * np.maximum(1.0, B / RIDGE_B)      # flat to ridge, then linear


def moe_routed(B):
    per_expert_read = 437.0                          # us per active local expert (INT4 weights)
    read = per_expert_read * E_active_local          # coupon-collector ramp (saturates at 48)
    compute_factor = np.maximum(1.0, rows_per_expert / INT4_RIDGE)
    return read * compute_factor


def activation(B):
    return 300.0 + 300.0 * B                          # norm/swiglu/residual/rope/router-topk ~B


def ep_comm(B):
    return 400.0 + 120.0 * B                          # latency floor + ~B (bytes); master excludes


def mla_attention(B, ctx):
    ctrl = 78.0 * B                                  # per-row launch/control (ctx-independent)
    kv = 1.57 * B * ctx                              # KV streaming ~ B*ctx
    return ctrl + kv


COMPONENTS = [
    ("BF16 GEMM weight reads (qkv_a / o_proj / dense / shared / lm_head)", "#4C72B0"),
    ("MoE routed-expert INT4 weight reads (coupon-collector)",            "#C44E52"),
    ("Activation / control (norm x61, swiglu, residual, router-topk)",    "#55A868"),
    ("MLA attention KV streaming  (~ B*ctx)",                             "#8172B3"),
    ("EP all-to-all (dispatch + combine, mock)",                         "#CCB974"),
]


def stacks_for(ctx):
    return [gemm_base(B), moe_routed(B), activation(B), mla_attention(B, ctx), ep_comm(B)]


fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
for ax, ctx in zip(axes, [1, 8192]):
    ys = stacks_for(ctx)
    ax.stackplot(B, *ys, labels=[c[0] for c in COMPONENTS],
                 colors=[c[1] for c in COMPONENTS], alpha=0.92)
    top = sum(y[-1] for y in ys)
    ax.axvline(RIDGE_B, ls="--", c="k", lw=1.1)
    ax.text(RIDGE_B + 0.8, top * 0.05, f"ridge B$\\approx${RIDGE_B:.0f}", fontsize=8, va="bottom", rotation=90)
    ax.axvline(8, ls=":", c="dimgray", lw=1.1)
    ax.text(8 + 0.6, top * 0.985, "B=8\nanchor", fontsize=7.5, color="dimgray", va="top")
    sub = "ctx = 1  (decode anchor: MLA negligible)" if ctx == 1 else "ctx = 8192  (long context)"
    ax.set_title(sub, fontsize=11)
    ax.set_xlabel("per-rank batch  B    (global = 8B)")
    ax.set_xlim(1, 64)
    ax.set_ylim(0, top * 1.03)
    ax.grid(alpha=0.25)
    ax.margins(x=0)

axes[0].set_ylabel("decode-step operator time, summed  (us, MOCK)")

# growth-source annotations on the anchor (ctx=1) panel
axes[0].annotate("MoE INT4: coupon-collector ramp\n(larger B -> more experts active\n-> more weight to read)",
                 xy=(9, 20000), xytext=(20, 52000), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#C44E52"))
axes[0].annotate("BF16 GEMM base: FLAT until ridge\n(batching is 'free' here:\nweight read fixed, throughput ~B)",
                 xy=(14, 4800), xytext=(2, 60000), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#4C72B0"))
axes[1].annotate("at long ctx, MLA KV (~ B*ctx)\nswamps the whole step",
                 xy=(45, 4.0e5), xytext=(6, 6.6e5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#8172B3"))
axes[1].text(0.97, 0.03, "note: y-scale ~10x the left panel", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=7.5, style="italic", color="dimgray")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Kimi-K2 TP1.DP8.EP8 decode: where does TPOT grow with batch B?   "
             "[MOCK data -- shapes only, pending H20 sweep]", fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0.10, 1, 0.96])
fig.savefig("tpot_decomposition.png", dpi=130)
fig.savefig("tpot_decomposition.svg")
print("wrote tpot_decomposition.png / .svg")
print(f"E_active_global at B=1/8/64 (global 8/64/512): "
      f"{E_active_global[0]:.0f} / {E_active_global[7]:.0f} / {E_active_global[63]:.0f}")
print(f"rows/expert at B=8/64: {rows_per_expert[7]:.1f} / {rows_per_expert[63]:.1f}")
