use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ScoreImageRequest {
    pub run_name: String,
    pub image_path: String,
}

#[derive(Debug, Deserialize)]
pub struct ScoreBatchRequest {
    pub run_name: String,
    pub image_paths: Option<Vec<String>>,
    pub image_root: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ScoreResult {
    pub image_path: String,
    pub quality_score: f32,
}

#[derive(Debug, Serialize)]
pub struct ScoreImageResponse {
    pub backend: &'static str,
    pub request_id: String,
    pub run_name: String,
    pub duration_ms: u128,
    pub result: ScoreResult,
}

#[derive(Debug, Serialize)]
pub struct ScoreBatchResponse {
    pub backend: &'static str,
    pub request_id: String,
    pub run_name: String,
    pub result_count: usize,
    pub duration_ms: u128,
    pub results: Vec<ScoreResult>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
    pub device: String,
    pub requested_device: String,
    pub started_at: String,
    pub cache_size: usize,
    pub loaded_runs: Vec<String>,
    pub repo_root: String,
}
