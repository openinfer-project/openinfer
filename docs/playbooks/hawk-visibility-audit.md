# Hawk Visibility Audit Playbook

> **TL;DR:** 用 [hawk](https://github.com/astral-sh/hawk) 做 workspace 级可见性审计。**必须跑 all-features**:单 feature profile 下 `dead_public` 有 61% 是 feature 盲区造成的假阳性（2026-07-22 基线：默认 profile 345 条 vs all-features 134 条）。

hawk 把 workspace 当闭世界，从 `hawk.toml` 声明的 production 二进制出发做可达性分析，报三种 lint:

- `hawk::dead_public` — 从 production 和全部非 production target(tests/benches/examples/doctests）都不可达。**report-only**,`--fix` 不动它；正确姿势是先收紧成 `pub(crate)`，让 rustc `dead_code` 二次确认后再删。
- `hawk::unnecessary_public` — `pub` 可降为 `pub(crate)`（含"只有测试在用"的子类）。
- `hawk::unnecessary_restricted_visibility` — `pub(crate)` 等受限可见性可进一步私有化。

## 安装

hawk 用 `rustc_private`，钉死 Rust 1.97.1；预编译包不要求 `rustc-dev`:

```sh
rustup toolchain install 1.97.1
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/astral-sh/hawk/releases/latest/download/cargo-hawk-installer.sh | sh
```

## 运行

```sh
RUSTC_BOOTSTRAP=1 \
OPENINFER_NCCL_ROOT=/data/opt/nccl-2.30.4 \
OPENINFER_TRITON_PYTHON=.venv/bin/python \
cargo +1.97.1 hawk check
```

三个环境变量缺一不可：

- `RUSTC_BOOTSTRAP=1` — 仓库钉 nightly(`generic_const_exprs`),hawk 只认 1.97.1 stable，需要它放开 feature gate。
- `OPENINFER_NCCL_ROOT` — all-features 下 `openinfer-kernels/moe` 会编 DeepEP shim，需要 NCCL >= 2.30.4。
- `OPENINFER_TRITON_PYTHON` — `qwen35-4b` 的 Triton AOT build-time codegen(build.rs 也会自动回退到 `.venv/bin/python`)。

预期耗时：全量首轮 ~20 分钟（含 Triton AOT 和 Marlin nvcc)；产物缓存在 `/tmp/cargo-hawk-target/<workspace>-<hash>`，重跑几分钟。

## 解读与坑

- `hawk.toml` 声明了 `openinfer` + `bench_serving` 两个 production bin 和单个 all-features profile。**普通 bin 不属于非 production 面**——未声明的 bin 独占的 API 会被误判 dead，新增 shipped bin 时记得加 `[[production]]`。
- kvbm-logical 是上游 dynamo 的 fork：它的 finding(2026-07 基线占 ~1/3）要不要收紧，取决于是否接受与上游 API 分叉，别批量处理。
- `--fix` 可机器应用可见性收紧（走 `cargo fix`)，但只在单 profile 下可用。
- hawk 的 all-features 全 target 编译顺带是 CI 盲区探测器：2026-07-22 首轮就跑出了 #741 修的 3 个 feature-gated 编译错误。

## 基线（2026-07-22,all-features）

946 findings:666 `unnecessary_public` / 134 `dead_public` / 146 `unnecessary_restricted_visibility`。`dead_public` 按 crate: kernels 33、qwen3 25、kvbm 25、core 23、qwen35 13，其余零星。
