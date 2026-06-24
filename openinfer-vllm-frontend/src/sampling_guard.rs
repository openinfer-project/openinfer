//! Rejects requests the engine would otherwise degrade silently (#243, #355):
//! sampling params that evaporate between the wire protocol and
//! `SamplingParams`, and `max_tokens` beyond the servable cap. One body parse
//! covers both checks on all three generation endpoints (`/v1/completions`,
//! `/v1/chat/completions`, `/inference/v1/generate`).

use std::sync::{Arc, OnceLock};

use axum::Json;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use log::warn;
use serde::Deserialize;

use super::{COMPLETION_ROUTE_BODY_LIMIT, bad_request};

/// Engine-derived cap on a request's generated tokens (`None` = no cap),
/// shared with the guard before the engine finishes loading so the HTTP
/// frontend can start concurrently with the engine. vllm-server builds the
/// router (and this guard) only after the engine bridge registers, which
/// happens after [`ServableCap::set`] — so generation requests always observe
/// the final value. An unset read means that invariant broke; reject loudly
/// instead of serving with a missing cap.
#[derive(Clone, Default)]
pub(crate) struct ServableCap(Arc<OnceLock<Option<u32>>>);

impl ServableCap {
    pub(crate) fn set(&self, cap: Option<u32>) {
        self.0
            .set(cap)
            .expect("servable cap must be set exactly once");
    }

    /// Outer `None` = engine still loading; inner = the engine's cap.
    #[allow(clippy::option_option)]
    fn get(&self) -> Option<Option<u32>> {
        self.0.get().copied()
    }
}

/// OpenAI-style invalid-request error, byte-compatible with vllm-server's
/// `ErrorDetail` wire shape (that type is not exported, so mirror it here).
fn invalid_request_error(param: &'static str, message: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": "invalid_request_error",
            }
        })),
    )
        .into_response()
}

/// Sampling-adjacent fields shared by the OpenAI request types (top level) and
/// `/inference/v1/generate` (nested under `sampling_params`). Field names match
/// both wire formats; fields one surface lacks simply stay `None`.
///
/// `use_beam_search` (both OpenAI endpoints) and `length_penalty` (completions
/// only) are also rejected by the upstream validators; they stay in the probe
/// so chat gets the same coverage and every unsupported param fails with one
/// consistent message.
#[derive(Default, Deserialize)]
struct Probe {
    seed: Option<serde_json::Value>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    repetition_penalty: Option<f64>,
    min_p: Option<f64>,
    min_tokens: Option<u64>,
    length_penalty: Option<f64>,
    #[serde(default)]
    use_beam_search: bool,
    logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    response_format: Option<serde_json::Value>,
    structured_outputs: Option<serde_json::Value>,
    stop_token_ids: Option<Vec<i64>>,
    allowed_token_ids: Option<Vec<i64>>,
    prompt_logprobs: Option<i64>,
    // logprobs is an integer top-k count on /v1/completions and
    // /inference/v1/generate (sampling_params), but a *boolean* on
    // /v1/chat/completions (paired with top_logprobs). Parse it as a raw JSON
    // value so a chat `logprobs: true` does not fail Probe deserialization --
    // a parse failure would make the guard skip every check and silently drop
    // unsupported fields. Only the integer -1 (full vocabulary) is rejected.
    logprobs: Option<serde_json::Value>,
    // cache_salt provides prefix-cache tenant isolation; the engine does not
    // implement tenant-aware caching, so requests with a salt would silently
    // share cache entries with other tenants.
    cache_salt: Option<serde_json::Value>,
    // `/inference/v1/generate` extras the OpenAI lowering never sets.
    bad_words: Option<Vec<serde_json::Value>>,
    logprob_token_ids: Option<Vec<i64>>,
    skip_reading_prefix_cache: Option<bool>,
    max_tokens: Option<u64>,
    // Chat's current spelling; upstream reads it over the deprecated
    // `max_tokens` when both are present.
    max_completion_tokens: Option<u64>,
}

impl Probe {
    fn first_unsupported(&self) -> Option<(&'static str, String)> {
        if self.seed.as_ref().is_some_and(|seed| !seed.is_null()) {
            return Some((
                "seed",
                "per-request seed is not supported by this engine".to_string(),
            ));
        }
        for (param, value, neutral) in [
            ("presence_penalty", self.presence_penalty, 0.0),
            ("frequency_penalty", self.frequency_penalty, 0.0),
            ("repetition_penalty", self.repetition_penalty, 1.0),
            ("min_p", self.min_p, 0.0),
            ("length_penalty", self.length_penalty, 1.0),
        ] {
            if let Some(value) = value
                && value != neutral
            {
                return Some((
                    param,
                    format!("{param} ({value}) is not supported by this engine"),
                ));
            }
        }
        if self.min_tokens.is_some_and(|min_tokens| min_tokens > 0) {
            return Some((
                "min_tokens",
                "min_tokens is not supported by this engine".to_string(),
            ));
        }
        if self.use_beam_search {
            return Some((
                "use_beam_search",
                "beam search is not supported by this engine".to_string(),
            ));
        }
        if self
            .logit_bias
            .as_ref()
            .is_some_and(|bias| !bias.is_empty())
        {
            return Some((
                "logit_bias",
                "logit_bias is not supported by this engine".to_string(),
            ));
        }
        // `{"type": "text"}` is the OpenAI default and a no-op; any other
        // shape (json_object / json_schema / structural_tag) would silently
        // produce unconstrained text.
        if self.response_format.as_ref().is_some_and(|format| {
            !format.is_null() && format.get("type").and_then(|t| t.as_str()) != Some("text")
        }) {
            return Some((
                "response_format",
                "structured output (response_format) is not supported by this engine".to_string(),
            ));
        }
        if self
            .structured_outputs
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Some((
                "structured_outputs",
                "structured_outputs is not supported by this engine".to_string(),
            ));
        }
        // The engine only stops on the model EOS; custom stop token ids would
        // be crossed without stopping.
        if self
            .stop_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
        {
            return Some((
                "stop_token_ids",
                "stop_token_ids is not supported by this engine (only stop strings and the model EOS)"
                    .to_string(),
            ));
        }
        if self
            .allowed_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
        {
            return Some((
                "allowed_token_ids",
                "allowed_token_ids is not supported by this engine".to_string(),
            ));
        }
        // The bridge drops prompt logprobs (TokenEvent::PromptTokens), so an
        // explicit request would return a response without them.
        if self.prompt_logprobs.is_some() {
            return Some((
                "prompt_logprobs",
                "prompt_logprobs is not supported by this engine".to_string(),
            ));
        }
        // logprobs=-1 requests full-vocabulary logprobs; the engine only
        // supports top-k (non-negative values) and would silently return no
        // logprobs if -1 reached the wire protocol (usize::try_from(-1).ok() =
        // None). Only an integer -1 is rejected; a boolean (chat) or any
        // non-negative count passes through.
        if self.logprobs.as_ref().and_then(serde_json::Value::as_i64) == Some(-1) {
            return Some((
                "logprobs",
                "logprobs=-1 (full vocabulary) is not supported by this engine; use a non-negative top-k value".to_string(),
            ));
        }
        // cache_salt enables prefix-cache tenant isolation; the engine does
        // not implement tenant-aware caching, so a salt would be silently
        // discarded and requests from different tenants would share cache
        // entries.
        if self.cache_salt.as_ref().is_some_and(|v| !v.is_null()) {
            return Some((
                "cache_salt",
                "cache_salt is not supported by this engine".to_string(),
            ));
        }
        if self
            .bad_words
            .as_ref()
            .is_some_and(|words| !words.is_empty())
        {
            return Some((
                "bad_words",
                "bad_words is not supported by this engine".to_string(),
            ));
        }
        if self
            .logprob_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
        {
            return Some((
                "logprob_token_ids",
                "logprob_token_ids is not supported by this engine".to_string(),
            ));
        }
        // The engine always reads the prefix cache; an explicit opt-out would
        // be silently ignored.
        if self.skip_reading_prefix_cache == Some(true) {
            return Some((
                "skip_reading_prefix_cache",
                "skip_reading_prefix_cache is not supported by this engine".to_string(),
            ));
        }
        None
    }

    fn over_capacity(&self, servable: Option<u32>) -> Option<(&'static str, u64, u32)> {
        let servable = servable?;
        // Mirror upstream's migration: `max_completion_tokens` wins over the
        // deprecated `max_tokens` when both are present.
        let (param, requested) = match (self.max_completion_tokens, self.max_tokens) {
            (Some(requested), _) => ("max_completion_tokens", requested),
            (None, Some(requested)) => ("max_tokens", requested),
            (None, None) => return None,
        };
        (requested > u64::from(servable)).then_some((param, requested, servable))
    }
}

/// `/inference/v1/generate` nests its sampling fields under `sampling_params`,
/// but `cache_salt` rides at the top level of the request (a sibling of
/// `sampling_params`), so it has to be captured here and folded into the probe.
#[derive(Deserialize)]
struct GenerateProbe {
    sampling_params: Option<Probe>,
    cache_salt: Option<serde_json::Value>,
}

/// The engine honors `temperature` / `top_p` / `top_k` / `ignore_eos`; every
/// other sampling or output-shaping knob would be dropped on the floor between
/// the wire protocol and `SamplingParams`. Silently ignoring a requested seed,
/// penalty, or JSON schema returns output the client explicitly asked us not
/// to produce, so reject non-neutral values up front (#243, #355). Neutral
/// values (penalty 0, repetition 1, empty logit_bias, `response_format:
/// {"type": "text"}`) are no-ops that common client SDKs send by default —
/// those pass. The same parse rejects `max_tokens` over the servable cap;
/// a too-long prompt is rejected later by the upstream tokenized-length check.
pub(crate) async fn guard_generation_request(
    State(cap): State<ServableCap>,
    request: Request,
    next: Next,
) -> Response {
    let nested = match request.uri().path() {
        "/v1/completions" | "/v1/chat/completions" => false,
        "/inference/v1/generate" => true,
        _ => return next.run(request).await,
    };
    let Some(servable) = cap.get() else {
        warn!("generation request arrived before the engine finished loading");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": "the engine is still loading",
                    "type": "service_unavailable_error",
                    "param": null,
                    "code": "service_unavailable_error",
                }
            })),
        )
            .into_response();
    };
    let (parts, body) = request.into_parts();
    let Ok(bytes) = to_bytes(body, COMPLETION_ROUTE_BODY_LIMIT).await else {
        return bad_request("failed to read request body");
    };
    // Malformed JSON falls through to the server's own JsonParseError path.
    let probe = if nested {
        serde_json::from_slice::<GenerateProbe>(&bytes)
            .ok()
            .map(|generate| {
                let mut probe = generate.sampling_params.unwrap_or_default();
                // cache_salt is top-level on the generate request, not inside
                // sampling_params, so it never lands in the nested probe above.
                probe.cache_salt = generate.cache_salt;
                probe
            })
    } else {
        serde_json::from_slice::<Probe>(&bytes).ok()
    };
    if probe.is_none() {
        warn!("sampling guard could not parse the request body; deferring to server validation");
    }
    if let Some(probe) = probe {
        if let Some((param, message)) = probe.first_unsupported() {
            warn!("rejecting request: {message}");
            return invalid_request_error(param, message);
        }
        if let Some((param, requested, servable)) = probe.over_capacity(servable) {
            let message = format!(
                "{param} ({requested}) exceeds the model's maximum context length of {servable} tokens"
            );
            warn!("rejecting request: {message}");
            return invalid_request_error(param, message);
        }
    }
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::middleware::from_fn_with_state;
    use axum::routing::{get, post};
    use tower::ServiceExt;

    use super::*;

    fn guarded_router(servable: Option<u32>) -> Router {
        let cap = ServableCap::default();
        cap.set(servable);
        unset_cap_router(cap)
    }

    fn unset_cap_router(cap: ServableCap) -> Router {
        Router::new()
            .route("/v1/completions", post(|| async { "ok" }))
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .route("/inference/v1/generate", post(|| async { "ok" }))
            .route("/v1/models", get(|| async { "models" }))
            .layer(from_fn_with_state(cap, guard_generation_request))
    }

    #[tokio::test]
    async fn generation_before_engine_ready_is_rejected_not_passed_through() {
        let (status, _) = send_json(
            unset_cap_router(ServableCap::default()),
            "/v1/completions",
            r#"{"max_tokens": 1}"#,
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    async fn send_json(router: Router, path: &str, body: &str) -> (StatusCode, serde_json::Value) {
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("build request"),
            )
            .await
            .expect("middleware response");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("read body");
        let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, value)
    }

    #[tokio::test]
    async fn unsupported_sampling_params_get_standard_openai_errors() {
        let cases = [
            (r#"{"seed": 42}"#, "seed"),
            (r#"{"presence_penalty": 0.5}"#, "presence_penalty"),
            (r#"{"frequency_penalty": -0.5}"#, "frequency_penalty"),
            (r#"{"repetition_penalty": 1.1}"#, "repetition_penalty"),
            (r#"{"min_p": 0.05}"#, "min_p"),
            (r#"{"length_penalty": 0.8}"#, "length_penalty"),
            (r#"{"min_tokens": 2}"#, "min_tokens"),
            (r#"{"use_beam_search": true}"#, "use_beam_search"),
            (r#"{"logit_bias": {"50256": -100}}"#, "logit_bias"),
            (
                r#"{"response_format": {"type": "json_object"}}"#,
                "response_format",
            ),
            (
                r#"{"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {}}}}"#,
                "response_format",
            ),
            (
                r#"{"structured_outputs": {"json": {}}}"#,
                "structured_outputs",
            ),
            (r#"{"stop_token_ids": [151645]}"#, "stop_token_ids"),
            (r#"{"allowed_token_ids": [1, 2]}"#, "allowed_token_ids"),
            (r#"{"prompt_logprobs": 0}"#, "prompt_logprobs"),
            (r#"{"logprobs": -1}"#, "logprobs"),
            (r#"{"cache_salt": "tenant-a"}"#, "cache_salt"),
        ];
        for (body, param) in cases {
            for path in ["/v1/completions", "/v1/chat/completions"] {
                let (status, error) = send_json(guarded_router(None), path, body).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {body}");
                assert_eq!(error["error"]["param"], param, "{body}");
                assert_eq!(error["error"]["type"], "invalid_request_error", "{body}");
                assert!(
                    error["error"]["message"]
                        .as_str()
                        .is_some_and(|m| !m.is_empty()),
                    "{body}"
                );
            }
        }
    }

    #[tokio::test]
    async fn generate_endpoint_rejects_nested_unsupported_params() {
        let cases = [
            (
                r#"{"token_ids": [1], "sampling_params": {"seed": 42}}"#,
                "seed",
            ),
            (
                r#"{"token_ids": [1], "sampling_params": {"min_p": 0.1}}"#,
                "min_p",
            ),
            (
                r#"{"token_ids": [1], "sampling_params": {"bad_words": ["foo"]}}"#,
                "bad_words",
            ),
            (
                r#"{"token_ids": [1], "sampling_params": {"logprob_token_ids": [5]}}"#,
                "logprob_token_ids",
            ),
            (
                r#"{"token_ids": [1], "sampling_params": {"structured_outputs": {"json": {}}}}"#,
                "structured_outputs",
            ),
            (
                r#"{"token_ids": [1], "sampling_params": {"logprobs": -1}}"#,
                "logprobs",
            ),
            // cache_salt rides at the top level of the generate request, not
            // inside sampling_params (mirrors GenerateRequest on the wire).
            (
                r#"{"token_ids": [1], "cache_salt": "tenant-a"}"#,
                "cache_salt",
            ),
        ];
        for (body, param) in cases {
            let (status, error) =
                send_json(guarded_router(None), "/inference/v1/generate", body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error["error"]["param"], param, "{body}");
            assert_eq!(error["error"]["type"], "invalid_request_error", "{body}");
        }
    }

    #[tokio::test]
    async fn max_completion_tokens_wins_over_deprecated_max_tokens() {
        // Upstream lowers max_completion_tokens when present; a small
        // max_tokens next to a huge max_completion_tokens must not mask it.
        let (status, error) = send_json(
            guarded_router(Some(100)),
            "/v1/chat/completions",
            r#"{"max_tokens": 10, "max_completion_tokens": 999999}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["error"]["param"], "max_completion_tokens");

        let (status, _) = send_json(
            guarded_router(Some(100)),
            "/v1/chat/completions",
            r#"{"max_tokens": 999999, "max_completion_tokens": 10}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_boolean_logprobs_does_not_disable_guards() {
        // Chat sends `logprobs: true` (a boolean, paired with top_logprobs),
        // unlike completions/generate where logprobs is an integer. The probe
        // must still parse so the remaining guards run -- otherwise an
        // unsupported field riding alongside it (here cache_salt) slips
        // through the silent drop this guard exists to prevent.
        let (status, error) = send_json(
            guarded_router(Some(100)),
            "/v1/chat/completions",
            r#"{"logprobs": true, "top_logprobs": 5, "cache_salt": "tenant-a"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["error"]["param"], "cache_salt");

        // A valid chat logprobs request (boolean, no unsupported fields) passes.
        let (status, _) = send_json(
            guarded_router(Some(100)),
            "/v1/chat/completions",
            r#"{"logprobs": true, "top_logprobs": 5}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn skip_reading_prefix_cache_opt_out_is_rejected() {
        let (status, error) = send_json(
            guarded_router(None),
            "/inference/v1/generate",
            r#"{"token_ids": [1], "sampling_params": {"skip_reading_prefix_cache": true}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["error"]["param"], "skip_reading_prefix_cache");

        let (status, _) = send_json(
            guarded_router(None),
            "/inference/v1/generate",
            r#"{"token_ids": [1], "sampling_params": {"skip_reading_prefix_cache": false}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn max_tokens_over_servable_cap_is_rejected_on_all_endpoints() {
        let cases = [
            ("/v1/completions", r#"{"max_tokens": 101}"#, "max_tokens"),
            (
                "/v1/chat/completions",
                r#"{"max_tokens": 101}"#,
                "max_tokens",
            ),
            (
                "/v1/chat/completions",
                r#"{"max_completion_tokens": 101}"#,
                "max_completion_tokens",
            ),
            (
                "/inference/v1/generate",
                r#"{"token_ids": [1], "sampling_params": {"max_tokens": 101}}"#,
                "max_tokens",
            ),
        ];
        for (path, body, param) in cases {
            let (status, error) = send_json(guarded_router(Some(100)), path, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
            assert_eq!(error["error"]["param"], param, "{path}");

            // At or under the cap — or with no cap configured — it passes.
            let (status, _) =
                send_json(guarded_router(Some(101)), path, &body.replace("101", "100")).await;
            assert_eq!(status, StatusCode::OK, "{path} under cap");
            let (status, _) = send_json(guarded_router(None), path, body).await;
            assert_eq!(status, StatusCode::OK, "{path} uncapped");
        }
    }

    #[tokio::test]
    async fn neutral_sampling_values_pass_through() {
        // Common SDKs send explicit defaults — those are no-ops, not errors.
        let bodies = [
            r#"{"seed": null}"#,
            r#"{"presence_penalty": 0.0, "frequency_penalty": 0}"#,
            r#"{"repetition_penalty": 1.0}"#,
            r#"{"min_p": 0, "min_tokens": 0}"#,
            r#"{"length_penalty": 1.0}"#,
            r#"{"use_beam_search": false}"#,
            r#"{"logit_bias": {}}"#,
            r#"{"response_format": {"type": "text"}}"#,
            r#"{"response_format": null, "structured_outputs": null}"#,
            r#"{"stop_token_ids": [], "allowed_token_ids": []}"#,
            r#"{"prompt_logprobs": null}"#,
            r#"{"temperature": 0.7, "top_p": 0.9, "top_k": 40}"#,
            // positive logprobs (top-k) is valid; only -1 (full-vocab) is not
            r#"{"logprobs": 5}"#,
            // null cache_salt is a no-op (common SDK default)
            r#"{"cache_salt": null}"#,
        ];
        for body in bodies {
            let (status, _) = send_json(guarded_router(Some(100)), "/v1/completions", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
        // Missing sampling_params: the guard passes it through; the real
        // server's typed parse rejects it (the field is required upstream) —
        // the stub handler's 200 only asserts the guard stays out of the way.
        for body in [
            r#"{"token_ids": [1]}"#,
            r#"{"token_ids": [1], "sampling_params": {"temperature": 0.7, "max_tokens": 50, "ignore_eos": true}}"#,
        ] {
            let (status, _) =
                send_json(guarded_router(Some(100)), "/inference/v1/generate", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
    }

    #[tokio::test]
    async fn guard_ignores_other_routes_and_malformed_json() {
        // Malformed JSON must fall through to the server's own parse error.
        let (status, _) =
            send_json(guarded_router(Some(100)), "/v1/completions", "{not json").await;
        assert_eq!(status, StatusCode::OK);

        let response = guarded_router(None)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }
}
