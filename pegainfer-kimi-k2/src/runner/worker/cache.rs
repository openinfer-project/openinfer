use super::{load::*, runtime::*, *};

impl KimiOneTokenForwardCache {
    pub(super) fn from_gpu_weights(
        ctx: &KimiRankGpuContext,
        weights: &KimiRankGpuWeights,
        names: &KimiRankWeightNames,
    ) -> Result<Self> {
        ensure!(
            weights.rank == names.rank,
            "Kimi forward cache rank mismatch: weights={}, names={}",
            weights.rank,
            names.rank
        );
        ensure!(
            names.layers.len() == KIMI_K2_LAYERS,
            "Kimi forward cache needs {} layers, got {}",
            KIMI_K2_LAYERS,
            names.layers.len()
        );

        let vocab_rows = names.plan.vocab_range.len();
        let token_embedding = GpuTensor::from_device_matrix_rows(
            raw_tensor(weights, &names.top.token_embedding)?.copy_bf16_matrix(
                ctx,
                vocab_rows,
                KIMI_K2_HIDDEN,
                "token_embedding",
            )?,
        )?;
        let final_norm = NormWeight::from_device_vec(
            raw_tensor(weights, &names.top.final_norm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_HIDDEN,
                "final_norm",
            )?,
        )?;
        let lm_head = GpuTensor::from_device_matrix_rows(
            raw_tensor(weights, &names.top.lm_head)?.copy_bf16_matrix(
                ctx,
                vocab_rows,
                KIMI_K2_HIDDEN,
                "lm_head",
            )?,
        )?;
        let layers = names
            .layers
            .iter()
            .map(|layer| load_layer_forward_cache(ctx, weights, layer))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vocab_start: names.plan.vocab_range.start,
            vocab_rows,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl KimiWorkerDecodeArena {
    pub(super) fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        batch_size: usize,
        page_size: usize,
        vocab_rows: usize,
    ) -> Result<Self> {
        ensure!(num_layers > 0, "Kimi decode arena needs layers");
        ensure!(
            batch_size > 0,
            "Kimi decode arena batch_size must be positive"
        );
        ensure!(
            KIMI_DECODE_PAGES_PER_REQUEST > 0,
            "Kimi decode arena pages per request must be positive"
        );
        let max_pages = batch_size
            .checked_mul(KIMI_DECODE_PAGES_PER_REQUEST)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode arena max_pages overflow"))?;
        let append_capacity = batch_size
            .checked_mul(KIMI_DECODE_ROPE_CACHE_TOKENS)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode arena append metadata overflow"))?;
        let layout = KimiMlaPagedKvLayout::separate_contiguous(max_pages, page_size, batch_size);
        let initial_positions = vec![0usize; batch_size];
        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            batch_size,
            page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &initial_positions,
        )?;
        let batch_indices = (0..batch_size).map(|idx| idx as i32).collect::<Vec<_>>();
        let positions = initial_positions
            .iter()
            .map(|position| *position as i32)
            .collect::<Vec<_>>();
        let mut batch_indices_padded = vec![0i32; append_capacity];
        let mut positions_padded = vec![0i32; append_capacity];
        batch_indices_padded[..batch_size].copy_from_slice(&batch_indices);
        positions_padded[..batch_size].copy_from_slice(&positions);
        let request_indices = (0..batch_size).map(|idx| idx as i32).collect::<Vec<_>>();
        let kv_tile_indices = vec![0i32; batch_size];
        let kv_chunk_size = vec![1i32; batch_size];
        let token_ids = vec![0u32; batch_size];
        let (cos_host, sin_host) = build_yarn_rope_cache(KIMI_DECODE_ROPE_CACHE_TOKENS);

        let mut layer_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_caches.push(KimiWorkerMlaLayerCache {
                ckv_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(layout.required_ckv_len()?)?,
                kpe_cache: ctx
                    .stream
                    .alloc_zeros::<half::bf16>(layout.required_kpe_len()?)?,
            });
        }

        Ok(Self {
            batch_size,
            page_size,
            max_pages,
            append_capacity,
            layout,
            page_indices_d: ctx.stream.clone_htod(&page_indices)?,
            page_indptr_d: ctx.stream.clone_htod(&page_indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&last_page_len)?,
            batch_indices_d: ctx.stream.clone_htod(&batch_indices_padded)?,
            positions_d: ctx.stream.clone_htod(&positions_padded)?,
            request_indices_d: ctx.stream.clone_htod(&request_indices)?,
            kv_tile_indices_d: ctx.stream.clone_htod(&kv_tile_indices)?,
            kv_chunk_size_d: ctx.stream.clone_htod(&kv_chunk_size)?,
            token_ids_d: ctx.stream.clone_htod(&token_ids)?,
            cos_d: ctx.stream.clone_htod(&cos_host)?,
            sin_d: ctx.stream.clone_htod(&sin_host)?,
            layer_caches,
            scratch: KimiWorkerDecodeScratch::new(ctx, batch_size)?,
            logits: HiddenStates::zeros(ctx, vocab_rows, batch_size)?,
            graph: CudaGraphState::new(),
        })
    }

    pub(super) fn configure_slot_prefill(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        seq_len: usize,
    ) -> Result<()> {
        ensure!(seq_len > 0, "Kimi prefill KV write requires tokens");
        ensure!(
            slot < self.batch_size,
            "Kimi prefill slot {slot} exceeds batch_size {}",
            self.batch_size
        );
        ensure!(
            seq_len <= KIMI_DECODE_ROPE_CACHE_TOKENS,
            "Kimi prefill seq_len {seq_len} exceeds per-request decode KV capacity {}",
            KIMI_DECODE_ROPE_CACHE_TOKENS
        );
        ensure!(
            seq_len <= self.append_capacity,
            "Kimi prefill seq_len {seq_len} exceeds append metadata capacity {}",
            self.append_capacity
        );
        let mut append_positions = vec![0usize; self.batch_size];
        append_positions[slot] = seq_len - 1;
        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            self.batch_size,
            self.page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &append_positions,
        )?;
        ensure!(
            page_indices.len() == self.max_pages,
            "Kimi prefill page table length {} must match arena max_pages {}",
            page_indices.len(),
            self.max_pages
        );
        let batch_indices = vec![slot as i32; seq_len];
        let positions = (0..seq_len as i32).collect::<Vec<_>>();
        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_len, &mut self.last_page_len_d)?;
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..seq_len);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..seq_len);
            ctx.stream.memcpy_htod(&positions, &mut positions_d)?;
        }
        Ok(())
    }

    pub(super) fn configure_batch_decode(
        &mut self,
        ctx: &DeviceContext,
        slots: &[usize],
        append_positions: &[usize],
    ) -> Result<()> {
        ensure!(!slots.is_empty(), "Kimi batch decode requires active slots");
        ensure!(
            slots.len() == append_positions.len(),
            "Kimi batch decode slots/positions mismatch: slots={}, positions={}",
            slots.len(),
            append_positions.len()
        );
        ensure!(
            slots.len() <= self.batch_size,
            "Kimi batch decode active slots {} exceeds batch_size {}",
            slots.len(),
            self.batch_size
        );
        let mut slot_positions = vec![0usize; self.batch_size];
        let mut batch_indices = vec![0i32; self.batch_size];
        let mut row_positions = vec![0i32; self.batch_size];
        let mut request_indices = vec![0i32; self.batch_size];
        let kv_tile_indices = vec![0i32; self.batch_size];
        let mut kv_chunk_size = vec![1i32; self.batch_size];
        let mut occupied_slots = vec![false; self.batch_size];

        for (row, (&slot, &position)) in slots.iter().zip(append_positions.iter()).enumerate() {
            ensure!(
                slot < self.batch_size,
                "Kimi batch decode slot {slot} exceeds batch_size {}",
                self.batch_size
            );
            ensure!(
                !occupied_slots[slot],
                "Kimi batch decode slot {slot} appears more than once"
            );
            ensure!(
                position < KIMI_DECODE_ROPE_CACHE_TOKENS,
                "Kimi decode append_position {position} exceeds per-request KV capacity {}",
                KIMI_DECODE_ROPE_CACHE_TOKENS
            );
            occupied_slots[slot] = true;
            slot_positions[slot] = position;
            batch_indices[row] = slot as i32;
            row_positions[row] = position as i32;
            request_indices[row] = slot as i32;
            kv_chunk_size[row] = (position + 1) as i32;
        }
        let free_slots = occupied_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, occupied)| (!occupied).then_some(slot))
            .collect::<Vec<_>>();
        for row in slots.len()..self.batch_size {
            let padding_slot = free_slots[row - slots.len()];
            batch_indices[row] = padding_slot as i32;
            request_indices[row] = padding_slot as i32;
        }

        let (page_indices, page_indptr, last_page_len) = build_decode_append_page_metadata(
            self.batch_size,
            self.page_size,
            KIMI_DECODE_PAGES_PER_REQUEST,
            &slot_positions,
        )?;
        ensure!(
            page_indices.len() == self.max_pages,
            "Kimi batch decode page table length {} must match arena max_pages {}",
            page_indices.len(),
            self.max_pages
        );
        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_len, &mut self.last_page_len_d)?;
        {
            let mut batch_indices_d = self.batch_indices_d.slice_mut(0..self.batch_size);
            ctx.stream
                .memcpy_htod(&batch_indices, &mut batch_indices_d)?;
        }
        {
            let mut positions_d = self.positions_d.slice_mut(0..self.batch_size);
            ctx.stream.memcpy_htod(&row_positions, &mut positions_d)?;
        }
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_chunk_size, &mut self.kv_chunk_size_d)?;
        Ok(())
    }

    pub(super) fn upload_batch_tokens(
        &mut self,
        ctx: &DeviceContext,
        token_ids: &[u32],
    ) -> Result<()> {
        ensure!(
            token_ids.len() <= self.batch_size,
            "Kimi batch token upload length {} exceeds batch_size {}",
            token_ids.len(),
            self.batch_size
        );
        let mut tokens = vec![0u32; self.batch_size];
        tokens[..token_ids.len()].copy_from_slice(token_ids);
        ctx.stream.memcpy_htod(&tokens, &mut self.token_ids_d)?;
        Ok(())
    }

    pub(super) fn append_prefill_layer_kv(
        &mut self,
        ctx: &DeviceContext,
        layer_idx: usize,
        compressed_normed: &GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
        append_kpe: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    ) -> Result<()> {
        ensure!(
            compressed_normed.seq_len <= self.append_capacity,
            "Kimi prefill append seq_len {} exceeds metadata capacity {}",
            compressed_normed.seq_len,
            self.append_capacity
        );
        let layer_cache = self.layer_caches.get_mut(layer_idx).ok_or_else(|| {
            anyhow::anyhow!("Kimi prefill KV layer cache {layer_idx} out of range")
        })?;
        kimi_mla_paged_kv_append(
            ctx,
            &mut layer_cache.ckv_cache,
            &mut layer_cache.kpe_cache,
            self.layout,
            &self.page_indices_d,
            &self.page_indptr_d,
            &self.last_page_len_d,
            compressed_normed,
            append_kpe,
            &self.batch_indices_d,
            &self.positions_d,
        )
    }
}

pub(super) fn build_decode_append_page_metadata(
    batch_size: usize,
    page_size: usize,
    pages_per_request: usize,
    append_positions: &[usize],
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    ensure!(
        append_positions.len() == batch_size,
        "Kimi decode append positions length {} must match batch_size {}",
        append_positions.len(),
        batch_size
    );
    ensure!(page_size > 0, "Kimi decode page_size must be positive");
    ensure!(
        pages_per_request > 0,
        "Kimi decode pages_per_request must be positive"
    );
    let max_pages = batch_size
        .checked_mul(pages_per_request)
        .ok_or_else(|| anyhow::anyhow!("Kimi decode page table max_pages overflow"))?;
    let mut page_indices = vec![0i32; max_pages];
    let mut page_indptr = Vec::with_capacity(batch_size + 1);
    let mut last_page_len = Vec::with_capacity(batch_size);
    let mut cursor = 0usize;

    for (request_idx, position) in append_positions.iter().copied().enumerate() {
        let cached_tokens_after_append = position
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode append position overflow"))?;
        let pages = cached_tokens_after_append.div_ceil(page_size);
        ensure!(
            pages <= pages_per_request,
            "Kimi decode request {request_idx} needs {pages} pages for position {position}, capacity is {pages_per_request}"
        );
        page_indptr.push(cursor as i32);
        let request_page_base = request_idx
            .checked_mul(pages_per_request)
            .ok_or_else(|| anyhow::anyhow!("Kimi decode request page base overflow"))?;
        for local_page in 0..pages {
            page_indices[cursor] = (request_page_base + local_page) as i32;
            cursor += 1;
        }
        let last_len = ((cached_tokens_after_append - 1) % page_size) + 1;
        last_page_len.push(last_len as i32);
    }
    page_indptr.push(cursor as i32);
    Ok((page_indices, page_indptr, last_page_len))
}

impl KimiWorkerDecodeScratch {
    pub(super) fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let marlin_block_size = kimi_marlin_block_size(batch_size);
        let marlin_route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, batch_size, marlin_block_size)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            marlin_route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            marlin_block_size,
        )?;
        Ok(Self {
            mla: crate::typed_scratch::MlaDecodeScratch::new(ctx, batch_size)?,
            dense_mlp: crate::typed_scratch::DenseMlpDecodeScratch::new(ctx, batch_size)?,
            shared_expert: crate::typed_scratch::SharedExpertDecodeScratch::new(ctx, batch_size)?,
            router: crate::typed_scratch::RouterScratch::new(ctx, batch_size)?,
            marlin: crate::typed_scratch::MarlinExpertScratch::new(ctx, batch_size)?,
            marlin_route_workspace,
            marlin_workspace,
            comm: crate::typed_scratch::CommScratch::new(ctx, batch_size)?,
            sampling: crate::typed_scratch::SamplingScratch::new(ctx, batch_size)?,
        })
    }
}
