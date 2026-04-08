use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Mutex, OnceLock},
};

use candle_core::{Device, Tensor};
use fast_image_resize::{FilterType as FastFilterType, ResizeAlg, ResizeOptions, Resizer};
use image::imageops::FilterType as ImageFilterType;
use jpeg_decoder::{ColorTransform, Decoder as JpegDecoder, PixelFormat};
use serde::Serialize;
use walkdir::WalkDir;

use crate::error::{AppError, AppResult};
use crate::model::RustModelMetadata;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff"];

#[derive(Clone, Copy)]
enum ResizeMode {
    ImageTriangle,
    ImageGaussian,
    ImageCatmullRom,
    FastConvolutionBox,
    FastConvolutionBilinear,
    FastConvolutionGaussian,
    FastInterpolationBilinear,
    FastConvolutionHamming,
    FastConvolutionMitchell,
    FastSuperSamplingBilinear2,
    FastSuperSamplingBilinear4,
}

impl ResizeMode {
    fn from_env() -> Self {
        match env::var("PV_IQA_RS_RESIZE_MODE")
            .unwrap_or_else(|_| "fast-convolution-bilinear".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "image-gaussian" => Self::ImageGaussian,
            "image-catmullrom" => Self::ImageCatmullRom,
            "fast-convolution-box" => Self::FastConvolutionBox,
            "fast-convolution-bilinear" => Self::FastConvolutionBilinear,
            "fast-convolution-gaussian" => Self::FastConvolutionGaussian,
            "fast-interpolation-bilinear" => Self::FastInterpolationBilinear,
            "fast-convolution-hamming" => Self::FastConvolutionHamming,
            "fast-convolution-mitchell" => Self::FastConvolutionMitchell,
            "fast-supersampling-bilinear-2" => Self::FastSuperSamplingBilinear2,
            "fast-supersampling-bilinear-4" => Self::FastSuperSamplingBilinear4,
            _ => Self::ImageTriangle,
        }
    }
}

#[derive(Clone, Copy)]
enum DecodeMode {
    ImageCrate,
    JpegDecoderGrayscale,
    JpegDecoderRgbToLuma,
}

impl DecodeMode {
    fn from_env() -> Self {
        match env::var("PV_IQA_RS_DECODE_MODE")
            .unwrap_or_else(|_| "image".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "jpeg-decoder-grayscale" => Self::JpegDecoderGrayscale,
            "jpeg-decoder-rgb-to-luma" => Self::JpegDecoderRgbToLuma,
            _ => Self::ImageCrate,
        }
    }
}

#[derive(Clone, Copy)]
enum PreprocessMode {
    Rust,
    PythonPillow,
}

impl PreprocessMode {
    fn from_env() -> Self {
        match env::var("PV_IQA_RS_PREPROCESS_MODE")
            .unwrap_or_else(|_| "python-pillow".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "rust" => Self::Rust,
            _ => Self::PythonPillow,
        }
    }
}

#[derive(Serialize)]
struct PythonPreprocessRequest {
    image_paths: Vec<String>,
    image_size: usize,
    grayscale_to_rgb: bool,
    normalize_mean: Vec<f32>,
    normalize_std: Vec<f32>,
}

struct PythonPreprocessHelper {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

static PYTHON_PREPROCESS_HELPER: OnceLock<Mutex<Option<PythonPreprocessHelper>>> = OnceLock::new();

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

fn resize_grayscale_image_fast(
    image: &image::GrayImage,
    size: u32,
    image_path: &Path,
    algorithm: ResizeAlg,
) -> AppResult<image::GrayImage> {
    let mut resized = image::GrayImage::new(size, size);
    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(algorithm);
    resizer
        .resize(image, &mut resized, Some(&options))
        .map_err(|error| {
            AppError::InvalidRequest(format!(
                "failed to resize image '{}': {error}",
                image_path.display()
            ))
        })?;
    Ok(resized)
}

fn resize_grayscale_image(
    image: &image::GrayImage,
    size: u32,
    image_path: &Path,
) -> AppResult<image::GrayImage> {
    let resized = match ResizeMode::from_env() {
        ResizeMode::ImageTriangle => {
            image::imageops::resize(image, size, size, ImageFilterType::Triangle)
        }
        ResizeMode::ImageGaussian => {
            image::imageops::resize(image, size, size, ImageFilterType::Gaussian)
        }
        ResizeMode::ImageCatmullRom => {
            image::imageops::resize(image, size, size, ImageFilterType::CatmullRom)
        }
        ResizeMode::FastConvolutionBox => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Convolution(FastFilterType::Box),
        )?,
        ResizeMode::FastConvolutionBilinear => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Convolution(FastFilterType::Bilinear),
        )?,
        ResizeMode::FastConvolutionGaussian => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Convolution(FastFilterType::Gaussian),
        )?,
        ResizeMode::FastInterpolationBilinear => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Interpolation(FastFilterType::Bilinear),
        )?,
        ResizeMode::FastConvolutionHamming => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Convolution(FastFilterType::Hamming),
        )?,
        ResizeMode::FastConvolutionMitchell => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::Convolution(FastFilterType::Mitchell),
        )?,
        ResizeMode::FastSuperSamplingBilinear2 => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::SuperSampling(FastFilterType::Bilinear, 2),
        )?,
        ResizeMode::FastSuperSamplingBilinear4 => resize_grayscale_image_fast(
            image,
            size,
            image_path,
            ResizeAlg::SuperSampling(FastFilterType::Bilinear, 4),
        )?,
    };
    Ok(resized)
}

fn decode_jpeg_with_jpeg_decoder(
    image_path: &Path,
    color_transform: ColorTransform,
) -> AppResult<image::GrayImage> {
    let file = File::open(image_path)?;
    let mut decoder = JpegDecoder::new(BufReader::new(file));
    decoder.set_color_transform(color_transform);
    let pixels = decoder.decode()?;
    let info = decoder.info().ok_or_else(|| {
        AppError::InvalidRequest(format!(
            "missing JPEG metadata for '{}'",
            image_path.display()
        ))
    })?;

    let gray_pixels = match info.pixel_format {
        PixelFormat::L8 => pixels,
        PixelFormat::RGB24 => pixels
            .chunks_exact(3)
            .map(|chunk| {
                let r = u32::from(chunk[0]);
                let g = u32::from(chunk[1]);
                let b = u32::from(chunk[2]);
                (((299 * r) + (587 * g) + (114 * b) + 500) / 1000) as u8
            })
            .collect(),
        pixel_format => {
            return Err(AppError::InvalidRequest(format!(
                "unsupported JPEG pixel format {pixel_format:?} for '{}'",
                image_path.display()
            )));
        }
    };

    image::GrayImage::from_raw(u32::from(info.width), u32::from(info.height), gray_pixels)
        .ok_or_else(|| {
            AppError::InvalidRequest(format!(
                "failed to build grayscale JPEG image for '{}'",
                image_path.display()
            ))
        })
}

fn load_grayscale_image(image_path: &Path) -> AppResult<image::GrayImage> {
    let extension = image_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let decode_mode = DecodeMode::from_env();
    if matches!(extension.as_str(), "jpg" | "jpeg") {
        match decode_mode {
            DecodeMode::ImageCrate => {}
            DecodeMode::JpegDecoderGrayscale => {
                return decode_jpeg_with_jpeg_decoder(image_path, ColorTransform::Grayscale);
            }
            DecodeMode::JpegDecoderRgbToLuma => {
                return decode_jpeg_with_jpeg_decoder(image_path, ColorTransform::RGB);
            }
        }
    }

    Ok(image::open(image_path)?.to_luma8())
}

fn python_helper_slot() -> &'static Mutex<Option<PythonPreprocessHelper>> {
    PYTHON_PREPROCESS_HELPER.get_or_init(|| Mutex::new(None))
}

fn spawn_python_preprocess_helper() -> AppResult<PythonPreprocessHelper> {
    let repo_root =
        env::var("PV_IQA_REPO_ROOT").unwrap_or_else(|_| "/root/workspace/PV-IQA".to_string());
    let mut child = Command::new("uv")
        .current_dir(&repo_root)
        .args([
            "run",
            "python",
            "-u",
            "-m",
            "pv_iqa.utils.rust_preprocess_helper",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child.stdin.take().ok_or_else(|| {
        AppError::InvalidRequest("failed to capture python preprocess stdin".to_string())
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        AppError::InvalidRequest("failed to capture python preprocess stdout".to_string())
    })?;
    Ok(PythonPreprocessHelper {
        child,
        stdin: BufWriter::new(stdin),
        stdout: BufReader::new(stdout),
    })
}

fn parse_f32_payload(payload: Vec<u8>, expected_count: usize) -> AppResult<Vec<f32>> {
    if payload.len() != expected_count * std::mem::size_of::<f32>() {
        return Err(AppError::InvalidRequest(format!(
            "python preprocess payload size mismatch: expected {} bytes, got {} bytes",
            expected_count * std::mem::size_of::<f32>(),
            payload.len()
        )));
    }

    Ok(payload
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn request_python_preprocess_locked(
    helper: &mut PythonPreprocessHelper,
    request: &PythonPreprocessRequest,
    expected_count: usize,
) -> AppResult<Vec<f32>> {
    let request_body = serde_json::to_vec(request)?;
    helper.stdin.write_all(&request_body)?;
    helper.stdin.write_all(b"\n")?;
    helper.stdin.flush()?;

    let mut status = [0_u8; 1];
    helper.stdout.read_exact(&mut status)?;
    let mut len_bytes = [0_u8; 8];
    helper.stdout.read_exact(&mut len_bytes)?;
    let payload_len = u64::from_le_bytes(len_bytes) as usize;
    let mut payload = vec![0_u8; payload_len];
    helper.stdout.read_exact(&mut payload)?;

    if status[0] == 0 {
        return parse_f32_payload(payload, expected_count);
    }

    let message = String::from_utf8_lossy(&payload).into_owned();
    Err(AppError::InvalidRequest(format!(
        "python preprocess failed: {message}"
    )))
}

fn request_python_preprocess(
    image_paths: &[PathBuf],
    metadata: &RustModelMetadata,
) -> AppResult<Vec<f32>> {
    let channels = if metadata.grayscale_to_rgb { 3 } else { 1 };
    let expected_count = image_paths.len() * channels * metadata.image_size * metadata.image_size;
    let request = PythonPreprocessRequest {
        image_paths: image_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        image_size: metadata.image_size,
        grayscale_to_rgb: metadata.grayscale_to_rgb,
        normalize_mean: metadata.normalize_mean.clone(),
        normalize_std: metadata.normalize_std.clone(),
    };

    let slot = python_helper_slot();
    let mut guard = slot
        .lock()
        .map_err(|_| AppError::InvalidRequest("python preprocess mutex poisoned".to_string()))?;

    for attempt in 0..2 {
        if guard.is_none() {
            *guard = Some(spawn_python_preprocess_helper()?);
        }

        let result = {
            let helper = guard.as_mut().expect("python helper should exist");
            request_python_preprocess_locked(helper, &request, expected_count)
        };

        match result {
            Ok(values) => return Ok(values),
            Err(error) if attempt == 0 => {
                if let Some(mut helper) = guard.take() {
                    let _ = helper.child.kill();
                    let _ = helper.child.wait();
                }
                if !matches!(error, AppError::Io(_)) {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }

    Err(AppError::InvalidRequest(
        "python preprocess helper could not be reached".to_string(),
    ))
}

fn load_images_to_tensor_rust(
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

    Ok(Tensor::from_vec(
        buffer,
        (image_paths.len(), channels, size, size),
        device,
    )?)
}

fn load_images_to_tensor_python(
    image_paths: &[PathBuf],
    metadata: &RustModelMetadata,
    device: &Device,
) -> AppResult<Tensor> {
    if image_paths.is_empty() {
        return Err(AppError::InvalidRequest(
            "At least one image path is required.".to_string(),
        ));
    }

    let channels = if metadata.grayscale_to_rgb { 3 } else { 1 };
    let values = request_python_preprocess(image_paths, metadata)?;
    Ok(Tensor::from_vec(
        values,
        (
            image_paths.len(),
            channels,
            metadata.image_size,
            metadata.image_size,
        ),
        device,
    )?)
}

fn append_image_to_buffer(
    buffer: &mut Vec<f32>,
    image_path: &Path,
    metadata: &RustModelMetadata,
) -> AppResult<()> {
    let size = metadata.image_size;
    let image = load_grayscale_image(image_path)?;
    let resized = resize_grayscale_image(&image, size as u32, image_path)?;
    if metadata.grayscale_to_rgb {
        for channel_idx in 0..3 {
            let mean = metadata
                .normalize_mean
                .get(channel_idx)
                .copied()
                .ok_or_else(|| {
                    AppError::InvalidRequest("normalize_mean is missing RGB entries".to_string())
                })?;
            let std = metadata
                .normalize_std
                .get(channel_idx)
                .copied()
                .ok_or_else(|| {
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
    load_images_to_tensor(&[image_path.to_path_buf()], metadata, device)
}

pub fn load_images_to_tensor(
    image_paths: &[PathBuf],
    metadata: &RustModelMetadata,
    device: &Device,
) -> AppResult<Tensor> {
    match PreprocessMode::from_env() {
        PreprocessMode::Rust => load_images_to_tensor_rust(image_paths, metadata, device),
        PreprocessMode::PythonPillow => load_images_to_tensor_python(image_paths, metadata, device),
    }
}
