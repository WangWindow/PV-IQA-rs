use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use walkdir::WalkDir;

use crate::error::{AppError, AppResult};
use crate::model::RustModelMetadata;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff"];

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| {
            let value = value.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&value.as_str())
        })
        .unwrap_or(false)
}

pub fn collect_image_paths(root: &Path) -> AppResult<Vec<PathBuf>> {
    if !root.exists() {
        return Err(AppError::NotFound(format!(
            "Image root does not exist: {}",
            root.display()
        )));
    }
    let mut paths = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file() && is_image_path(path))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn append_image_to_buffer(
    buffer: &mut Vec<f32>,
    image_path: &Path,
    metadata: &RustModelMetadata,
) -> AppResult<()> {
    let size = metadata.image_size;
    let image = image::open(image_path)?.to_luma8();
    let resized = image::imageops::resize(&image, size as u32, size as u32, FilterType::Triangle);
    if metadata.grayscale_to_rgb {
        for channel_idx in 0..3 {
            let mean = metadata.normalize_mean.get(channel_idx).copied().ok_or_else(|| {
                AppError::InvalidRequest("normalize_mean is missing RGB entries".to_string())
            })?;
            let std = metadata.normalize_std.get(channel_idx).copied().ok_or_else(|| {
                AppError::InvalidRequest("normalize_std is missing RGB entries".to_string())
            })?;
            for pixel in resized.pixels() {
                let value = f32::from(pixel[0]) / 255.0;
                buffer.push((value - mean) / std);
            }
        }
    } else {
        let mean = metadata.normalize_mean.first().copied().unwrap_or(0.0);
        let std = metadata.normalize_std.first().copied().unwrap_or(1.0);
        for pixel in resized.pixels() {
            let value = f32::from(pixel[0]) / 255.0;
            buffer.push((value - mean) / std);
        }
    }
    Ok(())
}

pub fn load_image_to_tensor(
    image_path: &Path,
    metadata: &RustModelMetadata,
    device: &Device,
) -> AppResult<Tensor> {
    let channels = if metadata.grayscale_to_rgb { 3 } else { 1 };
    let size = metadata.image_size;
    let mut buffer = Vec::with_capacity(channels * size * size);
    append_image_to_buffer(&mut buffer, image_path, metadata)?;
    Ok(Tensor::from_vec(buffer, (1, channels, size, size), device)?)
}

pub fn load_images_to_tensor(
    image_paths: &[PathBuf],
    metadata: &RustModelMetadata,
    device: &Device,
) -> AppResult<Tensor> {
    if image_paths.is_empty() {
        return Err(AppError::InvalidRequest(
            "At least one image path is required.".to_string(),
        ));
    }

    let size = metadata.image_size;
    let channels = if metadata.grayscale_to_rgb { 3 } else { 1 };
    let mut buffer = Vec::with_capacity(image_paths.len() * channels * size * size);

    for image_path in image_paths {
        append_image_to_buffer(&mut buffer, image_path.as_path(), metadata)?;
    }

    Ok(Tensor::from_vec(buffer, (image_paths.len(), channels, size, size), device)?)
}
