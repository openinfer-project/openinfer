use anyhow::Context as _;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::post;
use serde_json::json;
use tower::ServiceExt as _;

const REQUEST_BODY_LIMIT: usize = 16 * 1024 * 1024;

pub(crate) fn prefill_only_routes(vllm_router: Router) -> Router {
    Router::new()
        .route("/v1/completions", post(validate_prefill_only))
        .route("/v1/chat/completions", post(validate_prefill_only))
        .with_state(vllm_router.clone())
        .fallback_service(vllm_router)
}

async fn validate_prefill_only(
    axum::extract::State(vllm_router): axum::extract::State<Router>,
    request: Request,
) -> Response {
    match validate_prefill_only_inner(vllm_router, request).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("{error:#}"),
                    "type": "invalid_request_error",
                    "code": "invalid_request_error"
                }
            })),
        )
            .into_response(),
    }
}

async fn validate_prefill_only_inner(
    vllm_router: Router,
    request: Request,
) -> anyhow::Result<Response> {
    let (mut parts, body) = request.into_parts();
    let body = to_bytes(body, REQUEST_BODY_LIMIT)
        .await
        .context("failed to read OpenAI request body")?;
    let value: serde_json::Value =
        serde_json::from_slice(&body).context("failed to parse OpenAI request JSON")?;
    let max_tokens = value.get("max_tokens").and_then(serde_json::Value::as_u64);
    anyhow::ensure!(
        max_tokens == Some(1),
        "GLM5.2 prefill-only mode requires max_tokens=1, got {}",
        max_tokens.map_or_else(|| "omitted".to_owned(), |value| value.to_string())
    );
    parts.headers.insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&body.len().to_string())
            .context("request body length is invalid")?,
    );
    vllm_router
        .oneshot(Request::from_parts(parts, Body::from(body)))
        .await
        .context("vLLM router failed to handle prefill-only request")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn prefill_only_contract_returns_400_before_engine_submission() {
        let downstream =
            Router::new().route("/v1/chat/completions", post(|| async { StatusCode::OK }));
        let response = prefill_only_routes(downstream)
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"max_tokens":2}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
