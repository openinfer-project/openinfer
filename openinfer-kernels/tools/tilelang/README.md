# TileLang Generators

This directory owns TileLang-based CUDA source generators used by
`openinfer-kernels`.

Keep the technology boundary here and put model- or shape-family-specific
programs in subdirectories:

| Path | Role |
| --- | --- |
| `deepseek_v4/generate.py` | DeepSeek V4 MP8 FP8/FP4, sparse attention, and HC helper kernels. |

Generated CUDA is a build artifact under Cargo `OUT_DIR`; it should not be
checked into the repository.
