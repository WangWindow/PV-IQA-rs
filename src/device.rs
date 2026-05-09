use std::sync::Arc;

use candle_core::Device;

use crate::error::AppResult;

#[derive(Clone)]
pub struct RuntimeDevice {
    handle: Arc<Device>,
}

impl RuntimeDevice {
    pub fn from_env() -> AppResult<Self> {
        let device_str =
            std::env::var("PV_IQA_RS_DEVICE").unwrap_or_else(|_| "auto".into());
        let _ordinal = std::env::var("PV_IQA_RS_CUDA_ORDINAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let device = match device_str.trim().to_ascii_lowercase().as_str() {
            "cpu" => Device::Cpu,
            #[cfg(feature = "cuda")]
            "cuda" => Device::new_cuda(_ordinal).map_err(|e| {
                crate::error::AppError::InvalidRequest(format!("CUDA init failed: {e}"))
            })?,
            "auto" => {
                #[cfg(feature = "cuda")]
                {
                    if let Ok(d) = Device::new_cuda(_ordinal) {
                        return Ok(Self {
                            handle: Arc::new(d),
                        });
                    }
                }
                Device::Cpu
            }
            other => {
                return Err(crate::error::AppError::InvalidRequest(format!(
                    "Unknown device: {other}"
                )));
            }
        };
        Ok(Self {
            handle: Arc::new(device),
        })
    }

    pub fn handle(&self) -> Arc<Device> {
        self.handle.clone()
    }

    pub fn label(&self) -> &str {
        match *self.handle {
            Device::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => "cuda",
            _ => "unknown",
        }
    }
}
