mod device;
mod error;
mod model;
mod preprocess;
mod types;

use std::{path::PathBuf, sync::Arc};

use clap::{Parser, Subcommand};

use crate::{
    device::RuntimeDevice,
    model::{LoadedRun, ModelStore},
    types::ScoreResult,
};

#[derive(Parser)]
#[command(name = "pv-iqa", about = "Palm Vein Image Quality Assessment CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Inference device: cpu, cuda, or auto
    #[arg(long, default_value = "auto")]
    device: String,

    /// Output format: json or jsonl
    #[arg(long, default_value = "json")]
    output: String,
}

#[derive(Subcommand)]
enum Command {
    /// Score a single palm vein image
    Score {
        /// Path to ONNX model file
        #[arg(long)]
        model: PathBuf,
        /// Path to image file
        #[arg(long)]
        image: PathBuf,
    },
    /// Batch score all images in a directory
    Batch {
        /// Path to ONNX model file
        #[arg(long)]
        model: PathBuf,
        /// Path to image directory
        #[arg(long)]
        dir: PathBuf,
    },
}

fn extract_run_context(model: &PathBuf) -> (PathBuf, String) {
    // model: .../checkpoints/<run>/iqa/best.onnx
    // repo_root goes up: iqa/ → <run>/ → checkpoints/ → <repo>/
    let repo_root = model
        .parent()  // iqa/
        .and_then(|p| p.parent())  // <run>/
        .and_then(|p| p.parent())  // checkpoints/
        .and_then(|p| p.parent())  // <repo>/
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/root/workspace/PV-IQA"));

    let run_name = model
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    (repo_root, run_name)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    unsafe { std::env::set_var("PV_IQA_RS_DEVICE", &cli.device); }
    let runtime_device = RuntimeDevice::from_env()?;
    eprintln!("device: {}", runtime_device.label());

    let device_handle = runtime_device.handle();

    let model_path = match &cli.command {
        Command::Score { model, .. } | Command::Batch { model, .. } => model.clone(),
    };

    let (repo_root, run_name) = extract_run_context(&model_path);
    let model_store = ModelStore::new(repo_root, device_handle.clone());

    let loaded: Arc<LoadedRun> = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(model_store.get_or_load(&run_name))?;

    match &cli.command {
        Command::Score { image, .. } => {
            let tensor = preprocess::load_image_to_tensor(
                image.as_path(),
                &loaded.metadata,
                device_handle.as_ref(),
            )?;
            let score = loaded.score_tensor(tensor)?;
            let result = ScoreResult {
                image_path: image.display().to_string(),
                quality_score: score.into_iter().next().unwrap_or(0.0),
            };
            println!("{}", serde_json::to_string(&result)?);
        }
        Command::Batch { dir, .. } => {
            let image_paths = preprocess::collect_image_paths(dir.as_path())?;
            if image_paths.is_empty() {
                anyhow::bail!("No images found in {}", dir.display());
            }
            let mut results = Vec::with_capacity(image_paths.len());
            for path in &image_paths {
                let tensor = preprocess::load_image_to_tensor(
                    path.as_path(),
                    &loaded.metadata,
                    device_handle.as_ref(),
                )?;
                let score = loaded.score_tensor(tensor)?;
                results.push(ScoreResult {
                    image_path: path.display().to_string(),
                    quality_score: score.into_iter().next().unwrap_or(0.0),
                });
            }
            println!("{}", serde_json::to_string(&results)?);
        }
    }

    Ok(())
}
