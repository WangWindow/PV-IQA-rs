# PV-IQA-rs

PV-IQA ONNX 模型的 Rust CLI 推理工具。纯 Rust 图像预处理，CPU / CUDA 双后端。

## 用法

```bash
# 单张图片
pv-iqa score --model checkpoints/20250101-120000/iqa/best.onnx --image palm.jpg

# 文件夹批量
pv-iqa batch --model checkpoints/20250101-120000/iqa/best.onnx --dir ./images/

# 指定设备
pv-iqa --device cuda score --model best.onnx --image palm.jpg

# 静默模式（仅 stdout 输出 JSON）
pv-iqa --quiet score --model best.onnx --image palm.jpg
```

模型路径自动解析 run 上下文 — 从 `checkpoints/<run>/iqa/best.onnx` 推断出仓库根目录和实验名，加载同目录下的 `best.onnx.json` 元数据（输入名、输出名、图像尺寸、归一化参数、灰度转 RGB 标志）。

输出为 JSON，直接写 stdout（进度信息写 stderr）：

```json
{"image_path": "palm.jpg", "quality_score": 53.26}
```

`batch` 子命令输出 JSON 数组；`--output jsonl` 改为逐行 JSON。

## 构建

```bash
./build.sh                    # CPU + CUDA
./build.sh --skip-cuda        # 仅 CPU
./build.sh --skip-cpu         # 仅 CUDA
```

产物默认发布到 PV-IQA 主仓库的 `app/bin/pv-iqa-{cpu,cuda}`。

构建选项：

| 参数 | 说明 |
|------|------|
| `--profile release\|debug` | 构建 profile（默认 release） |
| `--cuda-features <value>` | CUDA Cargo features（默认 `cuda`） |
| `--repo-root <path>` | PV-IQA 主仓库路径 |
| `--publish-dir <path>` | 二进制输出目录 |

## 架构

| 文件 | 职责 |
|------|------|
| `src/main.rs` | CLI 入口（clap），全局参数 + `score`/`batch` 子命令 |
| `src/device.rs` | CPU / CUDA / auto 设备选择（支持 CUDA ordinal） |
| `src/model.rs` | ONNX 加载（`ModelStore` + 内存缓存）、初始器提取、tensor 推理 |
| `src/preprocess.rs` | 灰度图加载 → bilinear 缩放 → 归一化 → Tensor |
| `src/error.rs` | 统一错误类型（`thiserror`） |
| `src/types.rs` | `ScoreResult` 输出结构 |
| `candle-onnx/` | 本地修补的 candle-onnx 0.10.2 |

### 预处理管线

```
JPEG/PNG/BMP/TIFF → Luma8 灰度 → bilinear resize (224×224)
  → /255 归一化 → (value − mean) / std → 三通道复制（grayscale_to_rgb）
  → candle_core Tensor (NCHW, f32)
```

归一化参数与 Python 端一致（`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`），从 `best.onnx.json` 读取。

## candle-onnx CUDA 修复

本仓库包含 candle-onnx 0.10.2 的本地副本，通过 `[patch.crates-io]` 自动替换。

`eval.rs` 中 `Gemm` 算子原实现将 `alpha`/`beta` 标量固定在 CPU 上，CUDA 设备上下文下会导致计算图跨设备报错。修改为从输入 tensor 推断设备：

```diff
- let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
- let beta  = Tensor::full(beta,  c.shape(), &Device::Cpu)?;
+ let alpha = Tensor::full(alpha, a.shape(), a.device())?;
+ let beta  = Tensor::full(beta,  c.shape(), c.device())?;
```

相关 [issue #3491](https://github.com/huggingface/candle/issues/3491)。

## 依赖

- Rust stable (edition 2024)
- CUDA / nvcc（可选，构建 CUDA 版本需 `cuda` feature）
