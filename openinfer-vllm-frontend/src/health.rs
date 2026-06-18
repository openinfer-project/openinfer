use std::sync::{Arc, OnceLock};

use axum::Json;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use openinfer_engine::engine::{EngineHealth, EngineReadiness};

pub(crate) type HealthProbe = Arc<OnceLock<EngineHealth>>;

pub(crate) async fn guard_health_request(
    State(probe): State<HealthProbe>,
    req: Request,
    next: Next,
) -> Response {
    if req.uri().path() != "/health" {
        return next.run(req).await;
    }

    match probe.get().map(EngineHealth::readiness) {
        Some(EngineReadiness::Healthy) => {
            Json(serde_json::json!({ "status": "ok" })).into_response()
        }
        Some(EngineReadiness::Unhealthy { reason }) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "unhealthy",
                "reason": reason,
            })),
        )
            .into_response(),
        None => next.run(req).await,
    }
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use tower::ServiceExt;

    use super::*;

    async fn get_health(router: Router) -> (StatusCode, serde_json::Value) {
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap();
        (status, body)
    }

    fn router(probe: HealthProbe) -> Router {
        Router::new()
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({"status": "upstream"})) }),
            )
            .layer(from_fn_with_state(probe, guard_health_request))
    }

    #[tokio::test]
    async fn health_guard_reports_unhealthy_engine() {
        let probe = HealthProbe::default();
        let health = EngineHealth::new();
        health.mark_unhealthy("worker died");
        assert!(probe.set(health).is_ok());

        let (status, body) = get_health(router(probe)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "unhealthy");
        assert_eq!(body["reason"], "worker died");
    }

    #[tokio::test]
    async fn health_guard_reports_healthy_engine() {
        let probe = HealthProbe::default();
        assert!(probe.set(EngineHealth::new()).is_ok());

        let (status, body) = get_health(router(probe)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
    }
}
