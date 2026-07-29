# glm52-kernel-lab

GLM5.2 decode 逐内核调优台（Blackwell-only；一期 = EP 无关 per-rank 内核，rows {1,2,4,8,16,32,64} 轴：
1–8 = decode bucket，16–64 = MTP span-mapped verify 行 / 未来大 batch）。
设计合同：`docs/models/glm52/decode-op-bench-harness.md`。内核唯一事实源仍是
`openinfer-kernels/csrc/glm52/*.cu` — 本 harness 通过 ctypes dlopen
`libglm52_kernel_lab.so`（`openinfer-kernels/build.rs` 在 `OPENINFER_KERNEL_LAB=1`
时用生产 nvcc flags 把同一批 obj 链成 .so；不设 env 的默认构建零变化），
保证调优对象 = 生产 SASS。

## Quickstart

```bash
# build（在 repo 根；产出 target/release/libglm52_kernel_lab.so）
PYTHONPATH=openinfer-glm52/benches python3 -m kernel_lab build

# list（CPU-only，无 torch 也能跑）
PYTHONPATH=openinfer-glm52/benches python3 -m kernel_lab list

# check（对拍 torch 参考，按 manifest 容差判 PASS/FAIL）
PYTHONPATH=openinfer-glm52/benches python3 -m kernel_lab check mla_front.q_b_gemv --rows 64

# bench（capacity shape 计时；--save 写 baselines/<unit>.json 账本）
PYTHONPATH=openinfer-glm52/benches python3 -m kernel_lab bench mla_front.q_b_gemv --rows 8 --rows 64 --save

# compare（默认对账本出 delta；--baseline-so 同会话交替测量旧 .so）
PYTHONPATH=openinfer-glm52/benches python3 -m kernel_lab compare mla_front.q_b_gemv --rows 64
```

装进 repo `.venv` 后可直接用 `kernel_lab` console script：
`uv pip install -e openinfer-glm52/benches`。

## 环境（tray03）

- tray03 = GB300 NVL72 tray（sm_103 Blackwell，4×GPU）。本 harness 只调 Blackwell；
  账本按 arch 分桶，H200 数字不迁移（`capability.blackwell_only` fail-closed）。
- torch 用 repo `.venv` 里 oracle 已有的 pinned 版本即可，harness 自身只有
  stdlib + 惰性 torch：`list` / pytest / manifest 链路在无 torch 的 CPU 机器可用，
  `check`/`bench`/`compare` 缺 torch 时报清晰错误。
- 迭代循环：改 `.cu` → `kernel_lab build` → `check` → `bench` → `compare`
  → 顶门是同会话 `glm52_step_bench` A/B。锁频是 bench owner 的纪律
  （harness 只记录 `nvidia-smi clocks.sm`，不强制）。

## Tests

```bash
python3 -m pytest openinfer-glm52/benches/tests/ -q   # CPU-only，不 import torch
```
