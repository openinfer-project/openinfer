use super::*;

thread_local! {
    static FP32_ALL_REDUCE_SCRATCH: std::cell::RefCell<Vec<Option<(usize, usize, CudaSlice<f32>)>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub fn all_reduce_hidden_in_place(hidden: &mut Bf16HiddenStates, comm: &Comm) -> Result<()> {
    comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
        .map_err(|err| anyhow::anyhow!("NCCL all-reduce failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_f32_hidden_in_place(hidden: &mut F32HiddenStates, comm: &Comm) -> Result<()> {
    comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
        .map_err(|err| anyhow::anyhow!("NCCL F32 all-reduce failed: {err:?}"))?;
    Ok(())
}

pub(crate) fn all_gather_bf16_hidden_into(
    ctx: &RankGpuContext,
    comm: &Comm,
    local: &Bf16HiddenStates,
    world_size: usize,
    gathered: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        world_size > 0,
        "BF16 hidden all-gather world size must be positive"
    );
    ensure!(local.hidden_dim > 0, "BF16 hidden dim must be positive");
    ensure!(local.seq_len > 0, "BF16 hidden seq_len must be positive");
    let gathered_seq_len = local.seq_len * world_size;
    ensure!(
        gathered.hidden_dim == local.hidden_dim,
        "BF16 all-gather scratch dim mismatch: gathered={}, local={}",
        gathered.hidden_dim,
        local.hidden_dim
    );
    ensure!(
        gathered.seq_capacity() >= gathered_seq_len,
        "BF16 all-gather scratch capacity too small: gathered={}, required={gathered_seq_len}",
        gathered.seq_capacity()
    );
    gathered.seq_len = gathered_seq_len;
    let local_len = local.hidden_dim * local.seq_len;
    let gathered_len = local.hidden_dim * gathered_seq_len;
    let local_view = local.data.slice(0..local_len);
    let mut gathered_view = gathered.data.slice_mut(0..gathered_len);
    comm.all_gather(&local_view, &mut gathered_view)
        .map_err(|err| anyhow::anyhow!("NCCL BF16 hidden all-gather failed: {err:?}"))?;
    Ok(())
}

pub(crate) fn all_gather_u32_into(
    ctx: &RankGpuContext,
    comm: &Comm,
    local: &CudaSlice<u32>,
    world_size: usize,
    gathered: &mut CudaSlice<u32>,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(world_size > 0, "u32 all-gather world size must be positive");
    ensure!(!local.is_empty(), "u32 all-gather input must be non-empty");
    let required = local.len() * world_size;
    ensure!(
        gathered.len() >= required,
        "u32 all-gather scratch capacity too small: gathered={}, required={required}",
        gathered.len()
    );
    let mut gathered_view = gathered.slice_mut(0..required);
    comm.all_gather(local, &mut gathered_view)
        .map_err(|err| anyhow::anyhow!("NCCL u32 all-gather failed: {err:?}"))?;
    Ok(())
}

pub(crate) fn reduce_scatter_f32_hidden_into(
    ctx: &RankGpuContext,
    comm: &Comm,
    global: &F32HiddenStates,
    world_size: usize,
    local: &mut F32HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        world_size > 0,
        "F32 hidden reduce-scatter world size must be positive"
    );
    ensure!(global.hidden_dim > 0, "F32 hidden dim must be positive");
    ensure!(
        global.seq_len > 0,
        "F32 hidden reduce-scatter seq_len must be positive"
    );
    ensure!(
        global.seq_len.is_multiple_of(world_size),
        "F32 hidden reduce-scatter seq_len {} must be divisible by world size {}",
        global.seq_len,
        world_size
    );
    let local_seq_len = global.seq_len / world_size;
    ensure!(
        local.hidden_dim == global.hidden_dim,
        "F32 reduce-scatter local dim mismatch: local={}, global={}",
        local.hidden_dim,
        global.hidden_dim
    );
    ensure!(
        local.seq_capacity() >= local_seq_len,
        "F32 reduce-scatter local capacity too small: local={}, required={local_seq_len}",
        local.seq_capacity()
    );
    local.seq_len = local_seq_len;
    let global_len = global.hidden_dim * global.seq_len;
    let local_len = global.hidden_dim * local_seq_len;
    let global_view = global.data.slice(0..global_len);
    let mut local_view = local.data.slice_mut(0..local_len);
    comm.reduce_scatter(&global_view, &mut local_view, &ReduceOp::Sum)
        .map_err(|err| anyhow::anyhow!("NCCL F32 hidden reduce-scatter failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_fp32_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    comm: &Comm,
) -> Result<()> {
    FP32_ALL_REDUCE_SCRATCH.with(|scratch| -> Result<()> {
        let mut scratch = scratch.borrow_mut();
        if scratch.is_empty() {
            scratch.push(None);
        }

        let len = hidden.hidden_dim * hidden.seq_len;
        ctx.set_current()?;
        let slot = &mut scratch[0];
        let needs_alloc = slot
            .as_ref()
            .map(|(device_ordinal, capacity, _)| {
                *device_ordinal != ctx.device_ordinal || *capacity < len
            })
            .unwrap_or(true);
        if needs_alloc {
            *slot = Some((ctx.device_ordinal, len, unsafe {
                ctx.stream.alloc::<f32>(len)?
            }));
        }

        let (_, _, scratch_buf) = slot
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("missing FP32 all-reduce scratch"))?;
        let mut temp = scratch_buf.slice_mut(0..len);
        {
            let (input_ptr, _input_guard) = hidden.data.device_ptr(&ctx.stream);
            let (temp_ptr, _temp_guard) = temp.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                ffi::deepseek_bf16_to_f32_cuda(
                    input_ptr as *const ffi::Half,
                    temp_ptr as *mut f32,
                    len as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }

        if let Err(err) = comm.all_reduce_in_place(&mut temp, &ReduceOp::Sum) {
            return Err(anyhow::anyhow!("NCCL FP32 all-reduce failed: {err:?}"));
        }

        let (temp_ptr, _temp_guard) = temp.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_f32_to_bf16_cuda(
                temp_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
        Ok(())
    })
}

pub(crate) fn all_reduce_hidden_fp32_hc_post_view_into(
    ctx: &RankGpuContext,
    branch_out: &Bf16HiddenStates,
    residual: &HcHiddenStates,
    pre_state: &HcPreStateView<'_>,
    comm: &Comm,
    scratch: &mut CudaSlice<f32>,
    out: &mut HcHiddenStates,
) -> Result<()> {
    ensure!(
        branch_out.hidden_dim == residual.hidden_dim,
        "HC post all-reduce hidden dim mismatch: branch={}, residual={}",
        branch_out.hidden_dim,
        residual.hidden_dim
    );
    ensure!(
        branch_out.seq_len == residual.seq_len,
        "HC post all-reduce seq len mismatch: branch={}, residual={}",
        branch_out.seq_len,
        residual.seq_len
    );
    ensure!(
        pre_state.seq_len == branch_out.seq_len,
        "HC post all-reduce pre-state seq len mismatch: state={}, branch={}",
        pre_state.seq_len,
        branch_out.seq_len
    );
    ensure!(
        pre_state.hc == residual.hc,
        "HC post all-reduce pre-state multiplier mismatch: state={}, residual={}",
        pre_state.hc,
        residual.hc
    );
    ensure!(
        out.hidden_dim == branch_out.hidden_dim,
        "HC post all-reduce output hidden dim mismatch: out={}, branch={}",
        out.hidden_dim,
        branch_out.hidden_dim
    );
    ensure!(
        out.seq_capacity() >= branch_out.seq_len,
        "HC post all-reduce output capacity too small: out={}, branch={}",
        out.seq_capacity(),
        branch_out.seq_len
    );
    ensure!(
        out.hc == residual.hc,
        "HC post all-reduce output multiplier mismatch: out={}, residual={}",
        out.hc,
        residual.hc
    );

    let len = branch_out.hidden_dim * branch_out.seq_len;
    ensure!(
        scratch.len() >= len,
        "HC post all-reduce scratch too small: scratch={}, required={len}",
        scratch.len()
    );
    out.seq_len = branch_out.seq_len;
    ctx.set_current()?;
    let mut temp = scratch.slice_mut(0..len);
    {
        let (input_ptr, _input_guard) = branch_out.data.device_ptr(&ctx.stream);
        let (temp_ptr, _temp_guard) = temp.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_to_f32_cuda(
                input_ptr as *const ffi::Half,
                temp_ptr as *mut f32,
                len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if let Err(err) = comm.all_reduce_in_place(&mut temp, &ReduceOp::Sum) {
        return Err(anyhow::anyhow!("NCCL FP32 all-reduce failed: {err:?}"));
    }

    {
        let (temp_ptr, _temp_guard) = temp.device_ptr(&ctx.stream);
        let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = pre_state.post.device_ptr(&ctx.stream);
        let (comb_ptr, _comb_guard) = pre_state.comb.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_post_f32_branch_cuda(
                temp_ptr as *const f32,
                residual_ptr as *const ffi::Half,
                post_ptr as *const f32,
                comb_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                branch_out.seq_len as i32,
                residual.hc as i32,
                branch_out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}
