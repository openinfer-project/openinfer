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

pub fn all_reduce_hidden_group(
    comms_and_hidden: &mut [(&Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for (comm, hidden) in comms_and_hidden {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_group_bf16(
    contexts_comms_and_hidden: &mut [(&RankGpuContext, &Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for (_, comm, hidden) in contexts_comms_and_hidden {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL BF16 all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_group_fp32(
    contexts_comms_and_hidden: &mut [(&RankGpuContext, &Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    FP32_ALL_REDUCE_SCRATCH.with(|scratch| -> Result<()> {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < contexts_comms_and_hidden.len() {
            scratch.resize_with(contexts_comms_and_hidden.len(), || None);
        }

        for (slot, (ctx, _, hidden)) in scratch.iter_mut().zip(contexts_comms_and_hidden.iter_mut())
        {
            let len = hidden.hidden_dim * hidden.seq_len;
            ctx.set_current()?;
            let needs_alloc = slot
                .as_ref()
                .map(|(device_ordinal, capacity, _)| {
                    *device_ordinal != ctx.device_ordinal || *capacity < len
                })
                .unwrap_or(true);
            if needs_alloc {
                *slot = Some((ctx.device_ordinal, len, ctx.stream.alloc_zeros::<f32>(len)?));
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
        }

        group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
        for ((_, comm, hidden), slot) in
            contexts_comms_and_hidden.iter_mut().zip(scratch.iter_mut())
        {
            let len = hidden.hidden_dim * hidden.seq_len;
            let (_, _, scratch_buf) = slot
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("missing FP32 all-reduce scratch"))?;
            let mut temp = scratch_buf.slice_mut(0..len);
            if let Err(err) = comm.all_reduce_in_place(&mut temp, &ReduceOp::Sum) {
                let _ = group_end();
                return Err(anyhow::anyhow!("NCCL FP32 all-reduce failed: {err:?}"));
            }
        }
        group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

        for ((ctx, _, hidden), slot) in contexts_comms_and_hidden.iter_mut().zip(scratch.iter()) {
            let len = hidden.hidden_dim * hidden.seq_len;
            ctx.set_current()?;
            let (_, _, scratch_buf) = slot
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing FP32 all-reduce scratch"))?;
            let temp = scratch_buf.slice(0..len);
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
        }
        Ok(())
    })
}
