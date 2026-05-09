use super::*;

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
    let mut temps = Vec::<CudaSlice<f32>>::with_capacity(contexts_comms_and_hidden.len());
    for (ctx, _, hidden) in contexts_comms_and_hidden.iter_mut() {
        let len = hidden.hidden_dim * hidden.seq_len;
        ctx.set_current()?;
        let mut temp = ctx.stream.alloc_zeros::<f32>(len)?;
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
        temps.push(temp);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, comm, _), temp) in contexts_comms_and_hidden.iter_mut().zip(temps.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(temp, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL FP32 all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    for ((ctx, _, hidden), temp) in contexts_comms_and_hidden.iter_mut().zip(temps.iter()) {
        let len = hidden.hidden_dim * hidden.seq_len;
        ctx.set_current()?;
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
}
