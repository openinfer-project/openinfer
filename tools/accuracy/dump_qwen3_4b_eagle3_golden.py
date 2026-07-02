#!/usr/bin/env python
"""Generate the EAGLE-3 drafter golden fixture for the Qwen3-4B EAGLE-3 gate.

The reference is the OFFICIAL SafeAILab/EAGLE drafter forward (`eagle/model/cnets.py`,
`Model`), run on a seed-pinned synthetic input, so the Rust gate validates its own
drafter numerics (fc + rope + attention + mlp + head, i.e. the eagle3_rope and
single_decode/single_prefill_nhd kernels) against an INDEPENDENT implementation —
not against a sibling kernel.

What it pins (teacher-forced batched prefill over N positions, start_position=0):
  * `tokens`   [N]           i32  — the draft input tokens (embedded via the target's
                                     embed_tokens, which the drafter reuses)
  * `features` [N, 3*hidden] bf16 — the per-position fused target hidden fed to `fc`
                                     (token-major = HiddenStates layout data[s*dim+h])
  * `logits`   [N, dvoc]     f32  — the official drafter's per-position draft logits

Run once (needs torch + the AngelSlim eagle3 checkpoint + the Qwen3-4B embed weights
+ a checkout of https://github.com/SafeAILab/EAGLE):

  git clone --depth 1 https://github.com/SafeAILab/EAGLE /path/to/EAGLE
  python tools/accuracy/dump_qwen3_4b_eagle3_golden.py \
      --eagle-repo /path/to/EAGLE \
      --target models/Qwen3-4B --draft models/Qwen3-4B_eagle3 \
      --out test_data/qwen3-4b-eagle3-golden.safetensors

NOTE: the stock EAGLE `cnets.LlamaAttention` assumes head_dim = hidden // heads; Qwen3
uses head_dim=128 decoupled from hidden, so we patch ONLY the module construction
(dims) — the forward math (the oracle) is untouched. We also fix one upstream reshape
that hard-codes hidden_size for the non-square head_dim case.
"""
import argparse, json, os, sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

SEED = 0
N = 8  # draft positions


def patch_cnets(cnets):
    """head_dim=128 support + the non-square reshape fix (see module docstring)."""
    def attn_init(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
    cnets.LlamaAttention.__init__ = attn_init


def load_target_embed(target_dir):
    """Read model.embed_tokens.weight from the Qwen3-4B safetensors (no full load)."""
    idx = os.path.join(target_dir, "model.safetensors.index.json")
    key = "model.embed_tokens.weight"
    shard = json.load(open(idx))["weight_map"][key]
    with safe_open(os.path.join(target_dir, shard), "pt") as f:
        return f.get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle-repo", required=True, help="checkout of SafeAILab/EAGLE")
    ap.add_argument("--target", default="models/Qwen3-4B")
    ap.add_argument("--draft", default="models/Qwen3-4B_eagle3")
    ap.add_argument("--out", default="test_data/qwen3-4b-eagle3-golden.safetensors")
    args = ap.parse_args()

    # One-line upstream fix for non-square head_dim (Qwen3 head_dim=128): the attn
    # output reshape hard-codes hidden_size; it must be num_heads*head_dim. Apply it
    # to the clone in place (idempotent — no-op if already patched).
    cnets_path = os.path.join(args.eagle_repo, "eagle", "model", "cnets.py")
    src = open(cnets_path).read()
    bad = "attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)"
    good = "attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)"
    if bad in src:
        open(cnets_path, "w").write(src.replace(bad, good))

    sys.path.insert(0, os.path.join(args.eagle_repo, "eagle", "model"))
    import cnets
    patch_cnets(cnets)

    dc = json.load(open(os.path.join(args.draft, "config.json")))
    cfg = SimpleNamespace(
        vocab_size=dc["vocab_size"], draft_vocab_size=dc["draft_vocab_size"],
        hidden_size=dc["hidden_size"], intermediate_size=dc["intermediate_size"],
        num_hidden_layers=1, num_attention_heads=dc["num_attention_heads"],
        num_key_value_heads=dc["num_key_value_heads"], head_dim=dc["head_dim"],
        hidden_act=dc["hidden_act"], rms_norm_eps=dc["rms_norm_eps"],
        rope_theta=dc["rope_theta"], rope_scaling=dc.get("rope_scaling"),
        max_position_embeddings=dc["max_position_embeddings"],
        pad_token_id=None, pretraining_tp=1, target_hidden_size=dc["hidden_size"],
    )
    hidden = cfg.hidden_size
    dvoc = cfg.draft_vocab_size

    from safetensors.torch import load_file
    drafter = cnets.Model(cfg, load_emb=False)
    drafter.load_state_dict(load_file(os.path.join(args.draft, "model.safetensors")), strict=False)
    drafter.embed_tokens.weight.data = load_target_embed(args.target)
    drafter = drafter.to("cuda", torch.bfloat16).eval()

    gen = torch.Generator().manual_seed(SEED)
    tokens = torch.randint(1, 150000, (N,), generator=gen)
    features = (torch.randn(N, 3 * hidden, generator=gen)).to(torch.bfloat16)  # fc input

    with torch.no_grad():
        hs = features.to("cuda", torch.bfloat16).unsqueeze(0)  # [1, N, 3h]
        ids = tokens.to("cuda").unsqueeze(0)                   # [1, N]
        dec = drafter(hs, input_ids=ids, use_cache=False)      # [1, N, hidden]
        logits = drafter.lm_head(drafter.norm(dec))[0].float().cpu()  # [N, dvoc]

    tensors = {
        "tokens": tokens.to(torch.int32).contiguous(),
        "features": features.contiguous(),          # bf16 [N, 3h], token-major
        "logits": logits.contiguous(),              # f32  [N, dvoc]
    }
    meta = {
        "source": "SafeAILab/EAGLE cnets.Model (official EAGLE-3 drafter)",
        "seed": str(SEED), "n": str(N), "hidden": str(hidden),
        "draft_vocab_size": str(dvoc), "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata=meta)
    print(f"wrote {args.out}: N={N}, hidden={hidden}, dvoc={dvoc}")
    # sanity: report argmax of a couple positions
    am = logits.argmax(-1).tolist()
    print("reference argmax (draft-vocab ids):", am)


if __name__ == "__main__":
    main()
