use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use candle_core::{DType, Device, Tensor};
use candle_onnx::{
    dtype,
    onnx::{self, ModelProto, tensor_proto::DataType},
    read_file, simple_eval,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::{AppError, AppResult};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RustModelMetadata {
    pub checkpoint_path: String,
    pub model_path: String,
    pub input_name: String,
    pub output_name: String,
    pub image_size: usize,
    pub grayscale_to_rgb: bool,
    pub normalize_mean: Vec<f32>,
    pub normalize_std: Vec<f32>,
    pub opset_version: i64,
    #[serde(default)]
    pub dynamic_batch: bool,
    #[serde(default)]
    pub export_profile: String,
}

#[derive(Debug)]
pub struct LoadedRun {
    pub run_name: String,
    pub metadata: RustModelMetadata,
    pub model: ModelProto,
    pub initializers: HashMap<String, Tensor>,
}

impl LoadedRun {
    pub fn resolve_artifact_paths(repo_root: &Path, run_name: &str) -> (PathBuf, PathBuf) {
        let checkpoint_path = repo_root
            .join("checkpoints")
            .join(run_name)
            .join("iqa")
            .join("best.pt");
        let onnx_path = checkpoint_path.with_extension("onnx");
        let metadata_path = checkpoint_path.with_extension("onnx.json");
        (onnx_path, metadata_path)
    }

    pub fn from_repo(repo_root: &Path, run_name: &str, device: &Device) -> AppResult<Self> {
        let (model_path, metadata_path) = Self::resolve_artifact_paths(repo_root, run_name);
        if !metadata_path.exists() {
            return Err(AppError::NotFound(format!(
                "Rust metadata not found for run '{run_name}': {}",
                metadata_path.display()
            )));
        }
        if !model_path.exists() {
            return Err(AppError::NotFound(format!(
                "Rust ONNX model not found for run '{run_name}': {}",
                model_path.display()
            )));
        }
        let metadata =
            serde_json::from_str::<RustModelMetadata>(&fs::read_to_string(&metadata_path)?)?;
        let mut model = read_file(&model_path)?;
        let initializers = extract_initializers(&mut model, device)?;
        Ok(Self {
            run_name: run_name.to_string(),
            metadata,
            model,
            initializers,
        })
    }

    pub fn score_tensor(&self, input: Tensor) -> AppResult<Vec<f32>> {
        let mut values = self.initializers.clone();
        values.insert(self.metadata.input_name.clone(), input);
        let outputs = simple_eval(&self.model, values)?;
        let score = outputs.get(&self.metadata.output_name).ok_or_else(|| {
            AppError::NotFound(format!(
                "ONNX output '{}' was not produced for run '{}'",
                self.metadata.output_name, self.run_name
            ))
        })?;
        Ok(score.flatten_all()?.to_vec1::<f32>()?)
    }

    pub fn supports_dynamic_batch(&self) -> bool {
        self.metadata.dynamic_batch
    }
}

#[derive(Clone)]
pub struct ModelStore {
    repo_root: PathBuf,
    device: Arc<Device>,
    cache: Arc<RwLock<HashMap<String, Arc<LoadedRun>>>>,
}

impl ModelStore {
    pub fn new(repo_root: PathBuf, device: Arc<Device>) -> Self {
        Self {
            repo_root,
            device,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_or_load(&self, run_name: &str) -> AppResult<Arc<LoadedRun>> {
        if let Some(existing) = self.cache.read().await.get(run_name) {
            return Ok(existing.clone());
        }

        let repo_root = self.repo_root.clone();
        let device = self.device.clone();
        let run_name_owned = run_name.to_string();
        let loaded = tokio::task::spawn_blocking(move || {
            LoadedRun::from_repo(&repo_root, &run_name_owned, device.as_ref())
        })
        .await??;

        let loaded = Arc::new(loaded);
        let mut cache = self.cache.write().await;
        let entry = cache
            .entry(run_name.to_string())
            .or_insert_with(|| loaded.clone());
        Ok(entry.clone())
    }

    pub async fn cached_runs(&self) -> Vec<String> {
        let cache = self.cache.read().await;
        let mut runs = cache.keys().cloned().collect::<Vec<_>>();
        runs.sort();
        runs
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }
}

fn tensor_from_proto(tensor: &onnx::TensorProto, name: &str, device: &Device) -> AppResult<Tensor> {
    let dims: Vec<usize> = tensor.dims.iter().map(|&value| value as usize).collect();
    match DataType::try_from(tensor.data_type) {
        Ok(DataType::Int32) => {
            if tensor.int32_data.is_empty() {
                let len = tensor.raw_data.len() / 4;
                let data: &[i32] = unsafe {
                    std::slice::from_raw_parts(tensor.raw_data.as_ptr() as *const i32, len)
                };
                let data = data.iter().map(|value| *value as i64).collect::<Vec<_>>();
                Ok(Tensor::from_vec(data, len, device)?)
            } else {
                let data = tensor
                    .int32_data
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>();
                Ok(Tensor::from_vec(data, tensor.int32_data.len(), device)?)
            }
        }
        Ok(data_type) => match dtype(data_type) {
            Some(dtype) => {
                if dtype == DType::F32 && !tensor.float_data.is_empty() {
                    Ok(Tensor::from_slice(
                        &tensor.float_data,
                        dims.as_slice(),
                        device,
                    )?)
                } else if dtype == DType::F64 && !tensor.double_data.is_empty() {
                    Ok(Tensor::from_slice(
                        &tensor.double_data,
                        dims.as_slice(),
                        device,
                    )?)
                } else if dtype == DType::I64 && !tensor.int64_data.is_empty() {
                    Ok(Tensor::from_slice(
                        &tensor.int64_data,
                        dims.as_slice(),
                        device,
                    )?)
                } else {
                    Ok(Tensor::from_raw_buffer(
                        tensor.raw_data.as_slice(),
                        dtype,
                        dims.as_slice(),
                        device,
                    )?)
                }
            }
            None => Err(AppError::InvalidRequest(format!(
                "unsupported tensor data-type {data_type:?} for {name}"
            ))),
        },
        Err(_) => Err(AppError::InvalidRequest(format!(
            "unsupported tensor data-type {} for {name}",
            tensor.data_type
        ))),
    }
}

fn extract_initializers(
    model: &mut ModelProto,
    device: &Device,
) -> AppResult<HashMap<String, Tensor>> {
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| AppError::InvalidRequest("ONNX graph is missing.".to_string()))?;
    let mut initializers = HashMap::with_capacity(graph.initializer.len());
    for tensor in graph.initializer.iter() {
        let value = tensor_from_proto(tensor, tensor.name.as_str(), device)?;
        initializers.insert(tensor.name.clone(), value);
    }
    graph.initializer.clear();
    Ok(initializers)
}
