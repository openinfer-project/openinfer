//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::config::Config35;
use openinfer_core::tensor::{DeviceContext, DeviceVec};

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [num_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d.
    pub(crate) conv_state: DeviceVec,
}

/// Recurrent state for all linear attention layers.
pub(crate) struct RecurrentState {
    pub(crate) layers: Vec<LayerRecurrentState>,
    /// Number of tokens processed so far (for prefill/decode tracking).
    pub(crate) seq_len: usize,
}

impl RecurrentState {
    pub(crate) fn estimate_bytes(config: &Config35) -> usize {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let state_size = config
            .linear_num_value_heads
            .saturating_mul(config.linear_key_head_dim)
            .saturating_mul(config.linear_value_head_dim)
            .saturating_mul(std::mem::size_of::<f32>());
        let conv_state_size = config
            .linear_attn_qkv_dim()
            .saturating_mul(config.linear_conv_kernel_dim.saturating_sub(1))
            .saturating_mul(std::mem::size_of::<half::bf16>());
        num_linear_layers.saturating_mul(state_size.saturating_add(conv_state_size))
    }

    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();

        let state_size = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let qkv_dim = config.linear_attn_qkv_dim();
        let conv_state_size = qkv_dim * (config.linear_conv_kernel_dim - 1);

        let mut layers = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            let state: CudaSlice<f32> = ctx
                .stream
                .alloc_zeros(state_size)
                .map_err(|e| anyhow::anyhow!("Alloc recurrent state failed: {}", e))?;
            layers.push(LayerRecurrentState {
                state,
                conv_state: DeviceVec::zeros(ctx, conv_state_size)?,
            });
        }

        Ok(Self { layers, seq_len: 0 })
    }

    /// D2D copy all recurrent and convolution state from `src`.
    pub(crate) fn copy_from(&mut self, ctx: &DeviceContext, src: &RecurrentState) -> Result<()> {
        anyhow::ensure!(
            self.layers.len() == src.layers.len(),
            "Qwen3.5 recurrent copy layer mismatch: dst={}, src={}",
            self.layers.len(),
            src.layers.len()
        );
        for (layer_idx, (dst_layer, src_layer)) in
            self.layers.iter_mut().zip(src.layers.iter()).enumerate()
        {
            anyhow::ensure!(
                dst_layer.state.len() == src_layer.state.len(),
                "Qwen3.5 recurrent state length mismatch at layer {layer_idx}: dst={}, src={}",
                dst_layer.state.len(),
                src_layer.state.len()
            );
            anyhow::ensure!(
                dst_layer.conv_state.len == src_layer.conv_state.len,
                "Qwen3.5 conv state length mismatch at layer {layer_idx}: dst={}, src={}",
                dst_layer.conv_state.len,
                src_layer.conv_state.len
            );
            ctx.stream
                .memcpy_dtod(&src_layer.state, &mut dst_layer.state)
                .map_err(|e| anyhow::anyhow!("copy recurrent state layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_dtod(&src_layer.conv_state.data, &mut dst_layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("copy conv state layer {layer_idx}: {e}"))?;
        }
        self.seq_len = src.seq_len;
        Ok(())
    }
}
