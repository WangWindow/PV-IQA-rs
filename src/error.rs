use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    WalkDir(#[from] walkdir::Error),
}

impl From<tokio::task::JoinError> for AppError {
    fn from(e: tokio::task::JoinError) -> Self {
        AppError::InvalidRequest(e.to_string())
    }
}

impl From<jpeg_decoder::Error> for AppError {
    fn from(e: jpeg_decoder::Error) -> Self {
        AppError::InvalidRequest(e.to_string())
    }
}
