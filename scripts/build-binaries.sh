#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROFILE="release"
BUILD_CPU=1
BUILD_CUDA=1
CUDA_FEATURES="${PV_IQA_RS_CARGO_FEATURES:-cuda}"
REPO_ROOT="${PV_IQA_REPO_ROOT:-/root/workspace/PV-IQA}"
PUBLISH_DIR=""
TIMESTAMP="${PV_IQA_RS_BUILD_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%S)}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/build-binaries.sh [options]

Options:
  --repo-root <path>       PV-IQA 主仓库根目录
  --publish-dir <path>     二进制发布目录；默认 <repo-root>/bin
  --profile <release|debug>
                           构建 profile，默认 release
  --cuda-features <value>  CUDA 版使用的 Cargo features，默认 cuda
  --skip-cpu               跳过 CPU 版
  --skip-cuda              跳过 CUDA 版
  --timestamp <value>      指定产物时间戳
  -h, --help               显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --publish-dir)
      PUBLISH_DIR="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --cuda-features)
      CUDA_FEATURES="$2"
      shift 2
      ;;
    --skip-cpu)
      BUILD_CPU=0
      shift
      ;;
    --skip-cuda)
      BUILD_CUDA=0
      shift
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PROFILE}" != "release" && "${PROFILE}" != "debug" ]]; then
  echo "--profile 仅支持 release 或 debug" >&2
  exit 1
fi

if [[ "${BUILD_CPU}" -eq 0 && "${BUILD_CUDA}" -eq 0 ]]; then
  echo "CPU 与 CUDA 不能同时跳过" >&2
  exit 1
fi

if [[ -z "${PUBLISH_DIR}" ]]; then
  PUBLISH_DIR="${REPO_ROOT}/bin"
fi

mkdir -p "${PUBLISH_DIR}"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    BINARY_NAME="pv-iqa-rs.exe"
    ;;
  *)
    BINARY_NAME="pv-iqa-rs"
    ;;
esac

if [[ "${PROFILE}" == "release" ]]; then
  CARGO_PROFILE_ARGS=(--release)
  TARGET_SUBDIR="release"
else
  CARGO_PROFILE_ARGS=()
  TARGET_SUBDIR="debug"
fi

build_variant() {
  local variant="$1"
  local target_dir="$2"
  shift 2
  local cargo_args=("$@")

  echo "[build] ${variant} -> cargo build ${CARGO_PROFILE_ARGS[*]} ${cargo_args[*]}"
  cargo build "${CARGO_PROFILE_ARGS[@]}" "${cargo_args[@]}" --target-dir "${target_dir}"

  local source_path="${target_dir}/${TARGET_SUBDIR}/${BINARY_NAME}"
  local output_name="pv-iqa-rs-${PROFILE}-${variant}-${TIMESTAMP}"
  if [[ "${BINARY_NAME}" == *.exe ]]; then
    output_name="${output_name}.exe"
  fi
  local output_path="${PUBLISH_DIR}/${output_name}"

  cp "${source_path}" "${output_path}"
  chmod +x "${output_path}" || true
  echo "[publish] ${output_path}"
}

cd "${PROJECT_ROOT}"

if [[ "${BUILD_CPU}" -eq 1 ]]; then
  build_variant "cpu" "${PROJECT_ROOT}/target-${PROFILE}-cpu"
fi

if [[ "${BUILD_CUDA}" -eq 1 ]]; then
  build_variant "cuda" "${PROJECT_ROOT}/target-${PROFILE}-cuda" --features "${CUDA_FEATURES}"
fi

echo "[done] timestamp=${TIMESTAMP}"
