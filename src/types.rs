use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ScoreResult {
    pub image_path: String,
    pub quality_score: f32,
}
