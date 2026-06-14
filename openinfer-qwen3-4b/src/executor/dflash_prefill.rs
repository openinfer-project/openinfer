use super::{PrefillStepItem, RequestId};

pub(super) fn dflash_prefill_supported(req: &PrefillStepItem) -> bool {
    req.lora_adapter.is_none()
        && req.cached_tokens == 0
        && req.logprobs == 0
        && !req.echo
        && req.params.is_greedy()
}

pub(super) fn dflash_prefill_can_capture(
    req: &PrefillStepItem,
    pending_state_exists: bool,
) -> bool {
    dflash_prefill_supported(req) && (req.chunk_start == 0 || pending_state_exists)
}

pub(super) fn should_capture_dflash_prefill_context(
    requests: &[PrefillStepItem],
    pending_state_exists: impl Fn(RequestId) -> bool,
) -> bool {
    !requests.is_empty()
        && requests
            .iter()
            .all(|req| dflash_prefill_can_capture(req, pending_state_exists(req.request_id)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DFlashPrefillAction {
    MarkReady,
    KeepPending,
    Drop,
}

pub(super) fn dflash_prefill_action(
    captured_context: bool,
    completed: bool,
) -> DFlashPrefillAction {
    match (captured_context, completed) {
        (true, true) => DFlashPrefillAction::MarkReady,
        (true, false) => DFlashPrefillAction::KeepPending,
        (false, _) => DFlashPrefillAction::Drop,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DFlashPrefillAction, dflash_prefill_action, dflash_prefill_can_capture,
        should_capture_dflash_prefill_context,
    };
    use crate::executor::{PrefillStepItem, RequestId};
    use openinfer_core::sampler::SamplingParams;

    #[test]
    fn dflash_prefill_capture_requires_single_supported_request() {
        let greedy = SamplingParams::default();
        let mut non_greedy_params = greedy;
        non_greedy_params.temperature = 0.7;

        let supported = PrefillStepItem {
            request_id: RequestId::new(1),
            prompt_tokens: vec![1, 2, 3],
            max_output_tokens: 8,
            params: greedy,
            logprobs: 0,
            echo: false,
            lora_adapter: None,
            random_val: 0.0,
            cached_tokens: 0,
            chunk_budget: 3,
            chunk_start: 0,
            chunk_tokens: 0,
        };
        let mut second = supported.clone();
        second.request_id = RequestId::new(2);
        let mut non_greedy = supported.clone();
        non_greedy.params = non_greedy_params;

        assert!(should_capture_dflash_prefill_context(
            std::slice::from_ref(&supported),
            |_| false
        ));
        let mut cached = supported.clone();
        cached.cached_tokens = 1;
        assert!(should_capture_dflash_prefill_context(
            &[supported.clone(), second],
            |_| false
        ));
        assert!(!should_capture_dflash_prefill_context(
            &[non_greedy],
            |_| false
        ));
        assert!(!should_capture_dflash_prefill_context(&[cached], |_| false));

        let mut mid_chunk = supported.clone();
        mid_chunk.chunk_start = 2;
        mid_chunk.chunk_tokens = 1;
        assert!(!dflash_prefill_can_capture(&mid_chunk, false));
        assert!(dflash_prefill_can_capture(&mid_chunk, true));
    }

    #[test]
    fn dflash_prefill_capture_allows_multi_request_start_chunks() {
        let supported = PrefillStepItem {
            request_id: RequestId::new(1),
            prompt_tokens: vec![1, 2, 3],
            max_output_tokens: 8,
            params: SamplingParams::default(),
            logprobs: 0,
            echo: false,
            lora_adapter: None,
            random_val: 0.0,
            cached_tokens: 0,
            chunk_budget: 3,
            chunk_start: 0,
            chunk_tokens: 0,
        };
        let mut second = supported.clone();
        second.request_id = RequestId::new(2);

        assert!(should_capture_dflash_prefill_context(
            &[supported.clone(), second.clone()],
            |_| false
        ));

        let mut mid_chunk = second;
        mid_chunk.chunk_start = 3;
        assert!(!should_capture_dflash_prefill_context(
            &[supported.clone(), mid_chunk.clone()],
            |_| false
        ));
        assert!(should_capture_dflash_prefill_context(
            &[supported, mid_chunk],
            |id| id == RequestId::new(2)
        ));
    }

    #[test]
    fn dflash_prefill_keeps_state_until_final_chunk() {
        assert_eq!(
            dflash_prefill_action(true, false),
            DFlashPrefillAction::KeepPending
        );
        assert_eq!(
            dflash_prefill_action(true, true),
            DFlashPrefillAction::MarkReady
        );
        assert_eq!(
            dflash_prefill_action(false, false),
            DFlashPrefillAction::Drop
        );
        assert_eq!(
            dflash_prefill_action(false, true),
            DFlashPrefillAction::Drop
        );
    }
}
