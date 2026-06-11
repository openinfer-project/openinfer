//! Rejects sampling params the engine would otherwise silently drop (#243).

use axum::Json;
use axum::body::{Body, to_bytes};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use log::warn;
use serde::Deserialize;

use super::{COMPLETION_ROUTE_BODY_LIMIT, bad_request};

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

/// The engine honors `temperature` / `top_p` / `top_k` / `ignore_eos`; every
/// other sampling or output-shaping knob the OpenAI surface accepts would be
/// dropped on the floor between the wire protocol and `SamplingParams`.
/// Silently ignoring a requested seed, penalty, or JSON schema returns output
/// the client explicitly asked us not to produce, so reject non-neutral
/// values up front (#243). Neutral values (penalty 0, repetition 1, empty
/// logit_bias, `response_format: {"type": "text"}`) are no-ops that common
/// client SDKs send by default — those pass.
pub(crate) async fn reject_unsupported_sampling(request: Request, next: Next) -> Response {
    // `use_beam_search` (both endpoints) and `length_penalty` (completions
    // only) are also rejected by the upstream validators; they stay in the
    // probe so chat gets the same coverage and every unsupported param fails
    // with one consistent message.
    #[derive(Deserialize)]
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
            // shape (json_object / json_schema / structural_tag) would
            // silently produce unconstrained text.
            if self.response_format.as_ref().is_some_and(|format| {
                !format.is_null() && format.get("type").and_then(|t| t.as_str()) != Some("text")
            }) {
                return Some((
                    "response_format",
                    "structured output (response_format) is not supported by this engine"
                        .to_string(),
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
            // The engine only stops on the model EOS; custom stop token ids
            // would be crossed without stopping.
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
            // The bridge drops prompt logprobs (TokenEvent::PromptTokens), so
            // an explicit request would return a response without them.
            if self.prompt_logprobs.is_some() {
                return Some((
                    "prompt_logprobs",
                    "prompt_logprobs is not supported by this engine".to_string(),
                ));
            }
            None
        }
    }

    let path = request.uri().path();
    if path != "/v1/completions" && path != "/v1/chat/completions" {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let Ok(bytes) = to_bytes(body, COMPLETION_ROUTE_BODY_LIMIT).await else {
        return bad_request("failed to read request body");
    };
    // Malformed JSON falls through to the server's own JsonParseError path.
    if let Ok(probe) = serde_json::from_slice::<Probe>(&bytes)
        && let Some((param, message)) = probe.first_unsupported()
    {
        warn!("rejecting request: {message}");
        return invalid_request_error(param, message);
    }
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::middleware::from_fn;
    use axum::routing::{get, post};
    use tower::ServiceExt;

    use super::*;

    fn sampling_guard_router() -> Router {
        Router::new()
            .route("/v1/completions", post(|| async { "ok" }))
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .route("/v1/models", get(|| async { "models" }))
            .layer(from_fn(reject_unsupported_sampling))
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
        ];
        for (body, param) in cases {
            for path in ["/v1/completions", "/v1/chat/completions"] {
                let (status, error) = send_json(sampling_guard_router(), path, body).await;
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
        ];
        for body in bodies {
            let (status, _) = send_json(sampling_guard_router(), "/v1/completions", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
    }

    #[tokio::test]
    async fn sampling_guard_ignores_other_routes_and_malformed_json() {
        // Malformed JSON must fall through to the server's own parse error.
        let (status, _) = send_json(sampling_guard_router(), "/v1/completions", "{not json").await;
        assert_eq!(status, StatusCode::OK);

        let response = sampling_guard_router()
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
