# PV-IQA-rs

PV-IQA 的 Rust CLI 推理工具。加载 ONNX 模型，纯 Rust 预处理，CPU / CUDA 双后端。

## 使用

```bash
# 单张图片
pv-iqa score --model best.onnx --image palm.jpg

# 文件夹批量
pv-iqa batch --model best.onnx --dir ./images/

# 指定设备
pv-iqa --device cuda score --model best.onnx --image palm.jpg
```

输出为 JSON，直接写入 stdout：

```json
{"image_path": "palm.jpg", "quality_score": 53.26}
```

## 构建

```bash
./build.sh                    # CPU + CUDA
./build.sh --skip-cuda        # 仅 CPU
./build.sh --skip-cpu         # 仅 CUDA
```

产物发布到 `$REPO_ROOT/app/bin/pv-iqa-{cpu,cuda}`。

## 依赖

- Rust stable
- CUDA / nvcc（可选，构建 CUDA 版本）

<details>
<summary>candle-onnx CUDA 补丁</summary>

本项目包含 candle-onnx 0.10.2 的本地副本（`candle-onnx/`），通过 `[patch.crates-io]` 自动替换。

修改了 `eval.rs` 中 `Gemm` 算子的两行，将 `&Device::Cpu` 改为运行时设备引用，使 CUDA 推理可用。详见 [issue #3491](https://github.com/huggingface/candle/issues/3491)。

```diff
- let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
- let beta  = Tensor::full(beta,  c.shape(), &Device::Cpu)?;
+ let alpha = Tensor::full(alpha, a.shape(), a.device())?;
+ let beta  = Tensor::full(beta,  c.shape(), c.device())?;
```
</details>

## 架构

| 文件 | 职责 |
|------|------|
| `src/main.rs` | CLI 入口（clap） |
| `src/model.rs` | ONNX 加载 + 推理 |
| `src/preprocess.rs` | 图像解码 + 缩放 + 归一化 |
| `src/device.rs` | CPU/CUDA 设备选择 |
| `candle-onnx/` | 本地修补的 candle-onnx |
