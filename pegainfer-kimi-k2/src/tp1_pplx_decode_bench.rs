mod attention;
mod moe_compute;
mod pplx_comm;

use serde::Serialize;

pub const TP1_PPLX_ARENA_ROWS: usize = 8;
pub(crate) const TP1_PPLX_EP_WORLD: usize = 8;
pub(crate) const TP1_PPLX_LOCAL_EXPERTS: usize = 48;
pub(crate) const TP1_PPLX_EXPERT_PADDING: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Embedding,
    Attention,
    Dense,
    MoeShared,
    MoeRouter,
    MoePplxCompute,
    MoePplxComm,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundKind {
    Compute,
    Memory,
    Mixed,
    Comm,
    Control,
}

#[must_use]
#[derive(Clone, Debug, Serialize)]
pub struct BenchSpec {
    pub op: &'static str,
    pub label: &'static str,
    pub stage: Stage,
    pub active_rows: usize,
    pub arena_rows: usize,
    pub ctx_len: usize,
    pub calls_per_decode_step: usize,
    pub shape: Option<String>,
    pub m: Option<usize>,
    pub n: Option<usize>,
    pub k: Option<usize>,
    pub elem_count: Option<usize>,
    pub bytes_per_decode_step: u128,
    pub flops_per_decode_step: u128,
    pub bound: BoundKind,
    pub measure: MeasureKind,
    pub notes: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasureKind {
    ExistingProvider,
    EstimateOnly,
}

pub trait WorkAmount {
    fn into_u128(self) -> u128;
}

impl WorkAmount for u128 {
    fn into_u128(self) -> u128 {
        self
    }
}

impl WorkAmount for usize {
    fn into_u128(self) -> u128 {
        self as u128
    }
}

impl WorkAmount for u64 {
    fn into_u128(self) -> u128 {
        self as u128
    }
}

impl WorkAmount for u32 {
    fn into_u128(self) -> u128 {
        self as u128
    }
}

impl WorkAmount for i32 {
    fn into_u128(self) -> u128 {
        u128::try_from(self).expect("work amount must be non-negative")
    }
}

impl BenchSpec {
    pub fn new(
        op: &'static str,
        stage: Stage,
        active_rows: usize,
        arena_rows: usize,
        ctx_len: usize,
    ) -> Self {
        Self {
            op,
            label: op,
            stage,
            active_rows,
            arena_rows,
            ctx_len,
            calls_per_decode_step: 1,
            shape: None,
            m: None,
            n: None,
            k: None,
            elem_count: None,
            bytes_per_decode_step: 0,
            flops_per_decode_step: 0,
            bound: BoundKind::Control,
            measure: MeasureKind::EstimateOnly,
            notes: String::new(),
        }
    }

    pub fn label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    pub fn calls(mut self, calls: usize) -> Self {
        self.calls_per_decode_step = calls;
        self
    }

    pub fn calls_per_decode_step(self, calls: usize) -> Self {
        self.calls(calls)
    }

    pub fn shape(mut self, shape: impl Into<String>) -> Self {
        self.shape = Some(shape.into());
        self
    }

    pub fn shape_mnk(mut self, m: usize, n: usize, k: usize) -> Self {
        self.m = Some(m);
        self.n = Some(n);
        self.k = Some(k);
        self.shape = Some(format!("rows={m}, out={n}, in={k}"));
        self
    }

    pub fn m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self.refresh_mnk_shape();
        self
    }

    pub fn n(mut self, n: usize) -> Self {
        self.n = Some(n);
        self.refresh_mnk_shape();
        self
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self.refresh_mnk_shape();
        self
    }

    pub fn hidden_batch(mut self, hidden: usize, batch: usize) -> Self {
        self.m = Some(hidden);
        self.n = Some(batch);
        self.elem_count = Some(hidden * batch);
        self.shape = Some(format!("hidden={hidden}, batch={batch}"));
        self
    }

    pub fn elements(mut self, elem_count: usize) -> Self {
        self.elem_count = Some(elem_count);
        if self.shape.is_none() {
            self.shape = Some(format!("elems={elem_count}"));
        }
        self
    }

    pub fn elem(self, elem_count: usize) -> Self {
        self.elements(elem_count)
    }

    pub fn bytes(mut self, bytes_per_decode_step: impl WorkAmount) -> Self {
        self.bytes_per_decode_step = bytes_per_decode_step.into_u128();
        self
    }

    pub fn flops(mut self, flops_per_decode_step: impl WorkAmount) -> Self {
        self.flops_per_decode_step = flops_per_decode_step.into_u128();
        self
    }

    pub fn bound(mut self, bound: BoundKind) -> Self {
        self.bound = bound;
        self
    }

    pub fn measured(mut self) -> Self {
        self.measure = MeasureKind::ExistingProvider;
        self
    }

    pub fn estimate_only(mut self) -> Self {
        self.measure = MeasureKind::EstimateOnly;
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        let note = note.into();
        if self.notes.is_empty() {
            self.notes = note;
        } else if !note.is_empty() {
            self.notes.push_str("; ");
            self.notes.push_str(&note);
        }
        self
    }

    pub fn notes(self, note: impl Into<String>) -> Self {
        self.note(note)
    }

    fn refresh_mnk_shape(&mut self) {
        if let (Some(m), Some(n), Some(k)) = (self.m, self.n, self.k) {
            self.shape = Some(format!("rows={m}, out={n}, in={k}"));
        }
    }

    pub fn gemm(
        label: &'static str,
        stage: Stage,
        active_rows: usize,
        arena_rows: usize,
        ctx_len: usize,
        calls: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self {
        let flops = 2_u128 * m as u128 * n as u128 * k as u128 * calls as u128;
        let bytes = 2_u128
            * calls as u128
            * (m as u128 * k as u128 + k as u128 * n as u128 + m as u128 * n as u128);
        Self::new("gemm_graphsafe", stage, active_rows, arena_rows, ctx_len)
            .label(label)
            .calls(calls)
            .shape_mnk(m, n, k)
            .flops(flops)
            .bytes(bytes)
            .bound(BoundKind::Compute)
            .measured()
    }
}

pub fn specs(active_rows: usize, arena_rows: usize, ctx_len: usize) -> Vec<BenchSpec> {
    let mut specs = Vec::new();
    specs.extend(embedding_dense_specs(active_rows, arena_rows, ctx_len));
    specs.extend(attention::specs(active_rows, arena_rows, ctx_len));
    specs.extend(moe_compute::specs(active_rows, arena_rows, ctx_len));
    specs.extend(pplx_comm::specs(active_rows, arena_rows, ctx_len));
    specs
}

fn embedding_dense_specs(active_rows: usize, arena_rows: usize, ctx_len: usize) -> Vec<BenchSpec> {
    let dense_gate_up = 2 * crate::config::KIMI_K2_DENSE_INTERMEDIATE;
    vec![
        BenchSpec::new(
            "embedding_batch_vocab_shard",
            Stage::Embedding,
            active_rows,
            arena_rows,
            ctx_len,
        )
        .label("decode.embedding")
        .calls(1)
        .shape(format!(
            "vocab={}, hidden={}, rows={arena_rows}",
            crate::config::KIMI_K2_VOCAB,
            crate::config::KIMI_K2_HIDDEN
        ))
        .elements(arena_rows * crate::config::KIMI_K2_HIDDEN)
        .bytes(
            bf16_bytes(arena_rows * crate::config::KIMI_K2_HIDDEN) * 2
                + (arena_rows * std::mem::size_of::<u32>()) as u128,
        )
        .flops(0_u128)
        .bound(BoundKind::Memory)
        .measured()
        .note("token embedding fills the full TP1 DP-rank decode arena before active_rows is applied to MoE/top1"),
        dense_gemm_spec(
            "gemm_dm_typed_to_hs_graphsafe",
            "decode.dense.gate_up",
            active_rows,
            arena_rows,
            ctx_len,
            arena_rows,
            dense_gate_up,
            crate::config::KIMI_K2_HIDDEN,
        )
        .note("layer0 dense MLP gate/up consumes the post-attention normed arena rows"),
        BenchSpec::new(
            "silu_mul_hs_fused_into",
            Stage::Dense,
            active_rows,
            arena_rows,
            ctx_len,
        )
        .label("decode.dense.swiglu")
        .calls(1)
        .hidden_batch(crate::config::KIMI_K2_DENSE_INTERMEDIATE, arena_rows)
        .bytes(bf16_bytes(
            arena_rows * dense_gate_up
                + arena_rows * crate::config::KIMI_K2_DENSE_INTERMEDIATE,
        ))
        .flops(0_u128)
        .bound(BoundKind::Memory)
        .measured()
        .note("layer0 dense MLP fused SiLU-mul over gate/up halves"),
        dense_gemm_spec(
            "gemm_dm_hs_to_typed_graphsafe",
            "decode.dense.down",
            active_rows,
            arena_rows,
            ctx_len,
            arena_rows,
            crate::config::KIMI_K2_HIDDEN,
            crate::config::KIMI_K2_DENSE_INTERMEDIATE,
        )
        .note("layer0 dense MLP down projection writes hidden rows"),
        BenchSpec::new("add_batch", Stage::Dense, active_rows, arena_rows, ctx_len)
            .label("decode.dense.residual_add")
            .calls(1)
            .hidden_batch(crate::config::KIMI_K2_HIDDEN, arena_rows)
            .bytes(bf16_bytes(arena_rows * crate::config::KIMI_K2_HIDDEN * 3))
            .flops(0_u128)
            .bound(BoundKind::Memory)
            .measured()
            .note("layer0 dense residual add; TP1 has no dense MLP all-reduce"),
    ]
}

fn dense_gemm_spec(
    op: &'static str,
    label: &'static str,
    active_rows: usize,
    arena_rows: usize,
    ctx_len: usize,
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> BenchSpec {
    BenchSpec::new(op, Stage::Dense, active_rows, arena_rows, ctx_len)
        .label(label)
        .calls(1)
        .shape_mnk(rows, out_dim, in_dim)
        .elements(rows * out_dim)
        .flops(2_u128 * rows as u128 * out_dim as u128 * in_dim as u128)
        .bytes(bf16_bytes(
            rows * in_dim + out_dim * in_dim + rows * out_dim,
        ))
        .bound(BoundKind::Compute)
        .measured()
}

pub fn default_active_rows() -> &'static [usize] {
    &[1, 2, 4, TP1_PPLX_ARENA_ROWS]
}

pub fn default_ctx_lens() -> &'static [usize] {
    &[1, 128, 1024, 4096, 8192]
}

pub fn pplx_recv_capacity(arena_rows: usize) -> usize {
    let max_total_tokens = arena_rows * TP1_PPLX_EP_WORLD;
    let max_routes = max_total_tokens * crate::config::KIMI_K2_TOPK;
    let active_experts = max_routes.min(TP1_PPLX_LOCAL_EXPERTS);
    max_routes + active_experts * (TP1_PPLX_EXPERT_PADDING - 1)
}

pub(crate) fn bf16_bytes(elems: usize) -> u128 {
    elems as u128 * 2
}
