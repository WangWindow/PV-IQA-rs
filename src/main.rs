mod device;
mod error;
mod model;
mod preprocess;
mod types;

use std::{
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use chrono::Utc;
use rand::random;
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    device::RuntimeDevice,
    error::{AppError, AppResult},
    model::ModelStore,
    preprocess::{collect_image_paths, load_image_to_tensor, load_images_to_tensor},
    types::{
        HealthResponse, ScoreBatchRequest, ScoreBatchResponse, ScoreImageRequest,
        ScoreImageResponse, ScoreResult,
    },
};

#[derive(Clone)]
struct AppState {
    models: Arc<ModelStore>,
    runtime_device: RuntimeDevice,
    started_at: String,
}

impl AppState {
    async fn health_response(&self) -> HealthResponse {
        let loaded_runs = self.models.cached_runs().await;
        HealthResponse {
            status: "ok",
            service: "pv-iqa-rs",
            device: self.runtime_device.resolved_label().to_string(),
            requested_device: self.runtime_device.requested_label().to_string(),
            started_at: self.started_at.clone(),
            cache_size: loaded_runs.len(),
            loaded_runs,
            repo_root: self.models.repo_root().display().to_string(),
        }
    }
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(state.health_response().await)
}

fn resolve_batch_paths(request: &ScoreBatchRequest) -> AppResult<Vec<PathBuf>> {
    if let Some(image_paths) = &request.image_paths {
        if image_paths.is_empty() {
            return Err(AppError::InvalidRequest(
                "image_paths cannot be empty when provided.".to_string(),
            ));
        }
        return Ok(image_paths.iter().map(PathBuf::from).collect());
    }
    if let Some(image_root) = &request.image_root {
        return collect_image_paths(&PathBuf::from(image_root));
    }
    Err(AppError::InvalidRequest(
        "Provide either image_paths or image_root.".to_string(),
    ))
}

async fn score_image(
    State(state): State<AppState>,
    Json(request): Json<ScoreImageRequest>,
) -> AppResult<Json<ScoreImageResponse>> {
    let request_id = format!("{:08x}", random::<u32>());
    let started = Instant::now();
    let run_name = request.run_name.clone();
    let image_path = PathBuf::from(request.image_path.clone());
    let loaded = state.models.get_or_load(&run_name).await?;
    let device = state.runtime_device.handle();

    let result = tokio::task::spawn_blocking(move || {
        let tensor = load_image_to_tensor(image_path.as_path(), &loaded.metadata, device.as_ref())?;
        let score = loaded
            .score_tensor(tensor)?
            .into_iter()
            .next()
            .ok_or_else(|| AppError::InvalidRequest("Model returned no scores.".to_string()))?;
        Ok::<ScoreResult, AppError>(ScoreResult {
            image_path: image_path.display().to_string(),
            quality_score: score,
        })
    })
    .await??;

    let duration_ms = started.elapsed().as_millis();
    info!(
        request_id,
        run_name,
        image_path = %result.image_path,
        duration_ms,
        "rust image inference completed"
    );

    Ok(Json(ScoreImageResponse {
        backend: "rust-candle",
        request_id,
        run_name,
        duration_ms,
        result,
    }))
}

async fn score_batch(
    State(state): State<AppState>,
    Json(request): Json<ScoreBatchRequest>,
) -> AppResult<Json<ScoreBatchResponse>> {
    let request_id = format!("{:08x}", random::<u32>());
    let started = Instant::now();
    let run_name = request.run_name.clone();
    let image_paths = resolve_batch_paths(&request)?;
    let loaded = state.models.get_or_load(&run_name).await?;
    let image_paths_for_inference = image_paths.clone();
    let device = state.runtime_device.handle();

    let results = tokio::task::spawn_blocking(move || {
        if loaded.supports_dynamic_batch() {
            let tensor =
                load_images_to_tensor(&image_paths_for_inference, &loaded.metadata, device.as_ref())?;
            let scores = loaded.score_tensor(tensor)?;
            if scores.len() != image_paths_for_inference.len() {
                return Err(AppError::InvalidRequest(format!(
                    "Score count mismatch: expected {}, got {}",
                    image_paths_for_inference.len(),
                    scores.len()
                )));
            }
            Ok::<Vec<ScoreResult>, AppError>(
                image_paths_for_inference
                    .into_iter()
                    .zip(scores)
                    .map(|(path, score)| ScoreResult {
                        image_path: path.display().to_string(),
                        quality_score: score,
                    })
                    .collect(),
            )
        } else {
            let mut results = Vec::with_capacity(image_paths_for_inference.len());
            for path in image_paths_for_inference {
                let tensor = load_image_to_tensor(path.as_path(), &loaded.metadata, device.as_ref())?;
                let score = loaded
                    .score_tensor(tensor)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        AppError::InvalidRequest("Model returned no scores.".to_string())
                    })?;
                results.push(ScoreResult {
                    image_path: path.display().to_string(),
                    quality_score: score,
                });
            }
            Ok(results)
        }
    })
    .await??;

    let duration_ms = started.elapsed().as_millis();
    info!(
        request_id,
        run_name,
        result_count = results.len(),
        duration_ms,
        "rust batch inference completed"
    );

    Ok(Json(ScoreBatchResponse {
        backend: "rust-candle",
        request_id,
        run_name,
        result_count: results.len(),
        duration_ms,
        results,
    }))
}

fn parse_socket_addr(host: &str, port: u16) -> AppResult<SocketAddr> {
    format!("{host}:{port}")
        .parse::<SocketAddr>()
        .map_err(|error| AppError::InvalidRequest(format!("Invalid socket address: {error}")))
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false))
        .init();
}

#[tokio::main]
async fn main() -> AppResult<()> {
    init_tracing();

    let repo_root = PathBuf::from(
        env::var("PV_IQA_REPO_ROOT").unwrap_or_else(|_| "/root/workspace/PV-IQA".to_string()),
    );
    let host = env::var("PV_IQA_RS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("PV_IQA_RS_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(7007);
    let runtime_device = RuntimeDevice::from_env()?;
    info!(
        requested_device = runtime_device.requested_label(),
        resolved_device = runtime_device.resolved_label(),
        "resolved runtime device"
    );
    let state = AppState {
        models: Arc::new(ModelStore::new(repo_root, runtime_device.handle())),
        runtime_device,
        started_at: Utc::now().to_rfc3339(),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/score/image", post(score_image))
        .route("/score/batch", post(score_batch))
        .with_state(state.clone());

    let addr = parse_socket_addr(&host, port)?;
    warn!("pv-iqa-rs starting on http://{addr}");
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
