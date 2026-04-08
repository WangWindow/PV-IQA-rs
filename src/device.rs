use std::{env, sync::Arc};

use candle_core::Device;

use crate::error::{AppError, AppResult};

#[derive(Clone)]
pub struct RuntimeDevice {
    requested: String,
    resolved: String,
    handle: Arc<Device>,
}

impl RuntimeDevice {
    pub fn from_env() -> AppResult<Self> {
        let requested = env::var("PV_IQA_RS_DEVICE").unwrap_or_else(|_| "auto".to_string());
        let requested = requested.trim().to_ascii_lowercase();
        let ordinal = env::var("PV_IQA_RS_CUDA_ORDINAL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);

        match requested.as_str() {
            "auto" => Self::auto(ordinal),
            "cpu" => Ok(Self::cpu("cpu")),
            "cuda" => Self::cuda(ordinal),
            other => Err(AppError::InvalidRequest(format!(
                "Unsupported PV_IQA_RS_DEVICE value '{other}'. Use auto, cpu, or cuda."
            ))),
        }
    }

    pub fn handle(&self) -> Arc<Device> {
        self.handle.clone()
    }

    pub fn requested_label(&self) -> &str {
        &self.requested
    }

    pub fn resolved_label(&self) -> &str {
        &self.resolved
    }

    fn cpu(requested: &str) -> Self {
        Self {
            requested: requested.to_string(),
            resolved: "cpu".to_string(),
            handle: Arc::new(Device::Cpu),
        }
    }

    fn auto(ordinal: usize) -> AppResult<Self> {
        #[cfg(feature = "cuda")]
        {
            match Self::cuda(ordinal) {
                Ok(device) => Ok(device),
                Err(_) => Ok(Self::cpu("auto")),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Ok(Self::cpu("auto"))
        }
    }

    fn cuda(ordinal: usize) -> AppResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = Device::new_cuda(ordinal).map_err(|error| {
                AppError::InvalidRequest(format!(
                    "Failed to initialize CUDA device {ordinal}: {error}"
                ))
            })?;
            Ok(Self {
                requested: "cuda".to_string(),
                resolved: format!("cuda:{ordinal}"),
                handle: Arc::new(device),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(AppError::InvalidRequest(
                "This pv-iqa-rs build does not include the `cuda` feature.".to_string(),
            ))
        }
    }
}
