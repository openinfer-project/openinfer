# GLM5.2 TRT-LLM fused-MoE artifacts

These generated interface headers and SM100f cubins come from
`flashinfer-cubin 0.6.13` and match the vendored FlashInfer commit
`19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23`. The runtime embeds only the
FC1/FC2 tile-8 and tile-64 kernels selected by the GLM5.2 TP4 configuration;
their SHA-256 values are checked in `build.rs`. `flashinferMetaInfo.h` is
trimmed to those same four entries; the full 2,108-kernel artifact is not
vendored. Serving does not require a Python FlashInfer installation.

License: Apache-2.0; see the vendored
[FlashInfer license](../../../third_party/flashinfer/LICENSE).
