# PV-IQA-rs

`PV-IQA-rs` 是 `PV-IQA` 的 Rust/Candle 推理服务，负责将已经导出的 IQA ONNX 模型以 **CPU / CUDA** 两种方式提供在线推理能力。

该服务使用：

- `Axum` 提供 HTTP API
- `candle-core / candle-nn / candle-onnx` 执行 ONNX 前向推理
- 与主项目相同的导出产物：`best.onnx` + `best.onnx.json`
- 默认 `python-pillow` 预处理模式，以保持与 Python 推理结果一致

---

## 1. 功能概览

- 单图评分：`POST /score/image`
- 批量评分：`POST /score/batch`
- 健康检查：`GET /health`
- CPU / CUDA 双版本二进制构建与发布
- 与 `PV-IQA` 主仓库中的 run、ONNX 导出和 Bun API 直接集成

---

## 2. 目录与依赖关系

本项目默认依赖主仓库：

```text
/root/workspace/PV-IQA
```

原因：

1. 需要读取 `checkpoints/<run-name>/iqa/best.onnx`
2. 需要读取 `checkpoints/<run-name>/iqa/best.onnx.json`
3. 默认 `python-pillow` 预处理模式会调用主仓库中的 Python helper 模块

---

## 3. 环境要求

### 必需

- Rust stable
- Cargo

### 可选

- CUDA / `nvcc`：构建与运行 CUDA 版
- `uv` + Python 环境：默认 `python-pillow` 预处理模式需要

---

## 4. API

### `GET /health`

返回服务状态、设备信息、缓存中的 run 列表。

### `POST /score/image`

请求体：

```json
{
  "run_name": "20260407-130934",
  "image_path": "/root/workspace/PV-IQA/datasets/ROI_Data/ac_l/1.jpg"
}
```

### `POST /score/batch`

请求体支持两种形式：

```json
{
  "run_name": "20260407-130934",
  "image_paths": [
    "/abs/path/a.jpg",
    "/abs/path/b.jpg"
  ]
}
```

或：

```json
{
  "run_name": "20260407-130934",
  "image_root": "/abs/path/to/folder"
}
```

---

## 5. 运行

### 5.1 直接运行

```bash
cd /root/autodl-tmp/PV-IQA-rs
PV_IQA_REPO_ROOT=/root/workspace/PV-IQA cargo run --release
```

### 5.2 运行设备

- `PV_IQA_RS_DEVICE=auto`
- `PV_IQA_RS_DEVICE=cpu`
- `PV_IQA_RS_DEVICE=cuda`

---

## 6. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PV_IQA_REPO_ROOT` | `/root/workspace/PV-IQA` | 主仓库根目录 |
| `PV_IQA_RS_HOST` | `127.0.0.1` | 服务监听地址 |
| `PV_IQA_RS_PORT` | `7007` | 服务监听端口 |
| `PV_IQA_RS_DEVICE` | `auto` | 运行设备：`auto` / `cpu` / `cuda` |
| `PV_IQA_RS_CUDA_ORDINAL` | `0` | CUDA 设备编号 |
| `PV_IQA_RS_PREPROCESS_MODE` | `python-pillow` | 预处理模式：`python-pillow` / `rust` |
| `PV_IQA_RS_RESIZE_MODE` | `fast-convolution-bilinear` | 仅在 `rust` 预处理模式下生效的 resize 策略 |
| `PV_IQA_RS_DECODE_MODE` | `image` | 仅在 `rust` 预处理模式下生效的 JPEG 解码策略 |

### 预处理模式说明

#### `python-pillow`（默认）

- 通过常驻 Python helper 使用 Pillow 完成图像预处理
- 与 `PV-IQA` Python 推理链保持一致
- 适合需要**严格对齐 Python 分数**的场景

#### `rust`

- 完全使用 Rust 图像解码与缩放链
- 适合调试或纯 Rust 预处理实验
- 可配合 `PV_IQA_RS_RESIZE_MODE` / `PV_IQA_RS_DECODE_MODE` 做 A/B 对比

---

## 7. 构建

### 7.1 手动构建

#### CPU

```bash
cargo build --release
```

#### CUDA

```bash
cargo build --release --features cuda
```

> CPU 版默认**不带** `cuda` feature，以减小体积。

### 7.2 官方构建脚本

项目内置双版本构建脚本：

```bash
./scripts/build-binaries.sh --repo-root /root/workspace/PV-IQA
```

默认行为：

1. 分别构建 CPU 与 CUDA 二进制
2. 生成独立 target 目录，避免两种版本互相覆盖
3. 将产物发布到 `PV-IQA/bin/`
4. 输出形如以下文件：
   - `pv-iqa-rs-release-cpu-<timestamp>`
   - `pv-iqa-rs-release-cuda-<timestamp>`

### 7.3 构建脚本参数

```bash
./scripts/build-binaries.sh --help
```

支持的主要参数：

| 参数 | 说明 |
| --- | --- |
| `--repo-root <path>` | 指定主仓库根目录 |
| `--publish-dir <path>` | 指定二进制发布目录 |
| `--profile release|debug` | 构建 profile，默认 `release` |
| `--cuda-features <features>` | CUDA 版使用的 Cargo features，默认 `cuda` |
| `--skip-cpu` | 跳过 CPU 版 |
| `--skip-cuda` | 跳过 CUDA 版 |
| `--timestamp <value>` | 指定产物时间戳 |

---

## 8. 与 PV-IQA 主项目的关系

`PV-IQA-rs` 不负责训练，而是消费主项目已导出的产物：

```text
checkpoints/<run-name>/iqa/best.onnx
checkpoints/<run-name>/iqa/best.onnx.json
```

因此推荐工作流为：

1. 在 `PV-IQA` 中完成训练
2. 执行 `uv run pv-iqa export-rust-model --run-name <run-name>`
3. 启动或发布 `PV-IQA-rs`

---

## 9. 典型用法

### 手动启动 CUDA 服务

```bash
PV_IQA_REPO_ROOT=/root/workspace/PV-IQA \
PV_IQA_RS_DEVICE=cuda \
PV_IQA_RS_PREPROCESS_MODE=python-pillow \
cargo run --release --features cuda
```

### 构建并发布 CPU / CUDA 双版本

```bash
./scripts/build-binaries.sh --repo-root /root/workspace/PV-IQA
```

### curl 测试

```bash
curl http://127.0.0.1:7007/health
```

```bash
curl -X POST http://127.0.0.1:7007/score/image \
  -H 'content-type: application/json' \
  -d '{
    "run_name": "20260407-130934",
    "image_path": "/root/workspace/PV-IQA/datasets/ROI_Data/ac_l/1.jpg"
  }'
```

---

## 10. 说明

- 默认预处理模式优先保证与 Python 推理结果对齐
- 模型前向仍由 Rust/Candle 执行
- 若只想做纯 Rust 图像链实验，可设置 `PV_IQA_RS_PREPROCESS_MODE=rust`

