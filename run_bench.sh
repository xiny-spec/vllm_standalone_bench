#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — vLLM 多配置并发性能批量测试启动脚本
#
# 使用方法:
#   chmod +x run_bench.sh    # 首次使用需赋予执行权限
#   ./run_bench.sh           # 直接运行
#
# 依赖安装:
#   pip install aiohttp numpy tqdm          # 必须
#   pip install openpyxl                    # 可选：保存 XLSX 格式
#   pip install transformers                # 可选：精确 token 计数
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、服务连接配置
# =============================================================================
#
# 两种连接方式（二选一）：
#   方式A: 填写 HOST + PORT（适合标准 vLLM 部署）
#   方式B: 填写 BASE_URL（适合 HTTPS 域名 / 带路径前缀的网关 / 第三方 API）
#
# 方式B 示例: BASE_URL="https://aicp.teamshub.com/openai/api/v1/openai/v1"
#   末尾不加斜杠，且需包含完整版本路径 /v1
#   最终请求URL = BASE_URL + /completions（或 /chat/completions）
#   如: https://aicp.teamshub.com/openai/api/v1/openai/v1/completions
#
# BASE_URL 非空时自动忽略 HOST 和 PORT

# 完整 URL 前缀（留空则使用下方 HOST:PORT 方式）
BASE_URL=""
# BASE_URL="https://aicp.teamshub.com/openai/api/v1/openai/v1"  # 示例

# vLLM 服务的 IP 地址（BASE_URL 为空时生效）
HOST="10.86.0.32"

# vLLM 服务的端口号（BASE_URL 为空时生效）
PORT=27421

# 是否跳过 HTTPS 证书验证（仅在 BASE_URL 为 https:// 且证书不受信时设为 true）
INSECURE=false

# 模型名称，需与 vllm serve --model 参数完全一致
MODEL="qwen-7"

# 服务端模型别名（若 vllm serve 通过 --served-model-name 指定了别名，填此处）
# 留空则自动使用 MODEL 的值
SERVED_MODEL_NAME=""

# API 鉴权密钥（若服务端开启了 --api-key，设置环境变量 OPENAI_API_KEY；未开启则留空）
# 示例: export OPENAI_API_KEY="your-api-key"
API_KEY="${OPENAI_API_KEY:-}"

# =============================================================================
# ▌ 二、请求接口类型
# =============================================================================
#
#   openai      → /v1/completions       文本补全接口（推荐用于纯性能测试）
#   openai-chat → /v1/chat/completions  对话接口（贴近实际业务场景）
#
BACKEND="openai-chat"

# =============================================================================
# ▌ 三、测试维度配置（核心参数）
# =============================================================================

# 输入 token 长度列表（多个值用空格分隔，单位: token）
# 例: "128 512 1024 2048 4096"
INPUT_LENS="128 512 1024"

# 输出 token 长度列表
# 规则:
#   - 数量与 INPUT_LENS 相同 → 一一配对（第1输入对应第1输出，以此类推）
#   - 只写 1 个值            → 自动广播到所有输入长度
#   - 启用 CROSS_PRODUCT=true → 忽略此规则，改为全组合（见下方）
OUTPUT_LENS="1024"

# 笛卡尔积模式：设为 true 时，测试所有 INPUT_LENS × OUTPUT_LENS 的组合
# 例: INPUT_LENS="512 1024", OUTPUT_LENS="128 256" → 测试 4 种组合
CROSS_PRODUCT=false

# 并发请求数列表（每个值独立运行一次测试，多个值用空格分隔）
# 例: "1 4 8 16 32 64" 表示分别测试 1/4/8/16/32/64 并发
PARALLEL_NUMS="1 4 8"

# 每组配置的重复测试轮数（轮数越多结果越稳定，建议 3~5）
# 注意: 总请求数 = PARALLEL_NUMS × EPOCHS（每轮每个并发发一次请求）
EPOCHS=3

# 每组配置测试完成后的等待时间（秒），让服务端 KV cache 释放、温度恢复
# 配置数量多时建议适当增大（如 5~10s）；快速测试时可设为 0
SLEEP_BETWEEN=2.0

# 前缀缓存比例 [0.0 ~ 1.0]（默认 0.0 = 不使用共享前缀）
#
# 用于测试 vLLM prefix caching（自动前缀缓存 / radix cache）对性能的影响。
#
# 原理：取值 X 时，每组测试中所有请求共享同一段前缀文本（占 input_len 的 X 倍 token），
#        后缀部分每个请求独立随机生成，保证请求间差异性。
#        实际 prompt_len ≈ input_len × (1 + X)
#
# 典型用法：
#   PREFIX_RATIO=0.0   → 全随机，无共享（测试无缓存基准）
#   PREFIX_RATIO=0.5   → 50% 前缀共享（input_len=512 → 前缀256 + 后缀512）
#   PREFIX_RATIO=0.9   → 90% 前缀共享（高缓存命中率场景）
#
# 注意：需同时在 vLLM 服务端启用前缀缓存（--enable-prefix-caching）才有效。
PREFIX_RATIO=0.8
# PREFIX_RATIO=0.5   # 示例：测试 50% 前缀缓存场景

# =============================================================================
# ▌ 四、Tokenizer 配置（可选，用于精确 token 计数）
# =============================================================================
#
# 指定 HuggingFace tokenizer 的本地路径后，工具会生成精确 token 数量的随机文本。
# 不指定则使用空格分隔数字序列近似（token 数量误差较小，通常可接受）。
#
# 通过环境变量 TOKENIZER_PATH 指定；留空则使用近似字符串模式。
# 示例: export TOKENIZER_PATH="/path/to/your/tokenizer"
TOKENIZER="${TOKENIZER_PATH:-}"
# TOKENIZER="/path/to/your/tokenizer"     # 本地 tokenizer 目录
# TOKENIZER="Qwen/Qwen2.5-7B-Instruct"   # HuggingFace Hub 模型名（需网络）

# =============================================================================
# ▌ 五、约束过滤（可选，自动跳过超限配置）
# =============================================================================
#
# 最大允许的 TTFT 均值（毫秒）
# 当某并发数下 TTFT 均值超过此阈值时，该 (输入,输出) 组合的更高并发测试自动跳过。
# 例: 5000 → TTFT 超 5 秒即跳过更高并发
# 留空则不限制
MAX_TTFT_MS="15000"
# MAX_TTFT_MS=5000    # 例: 超过 5000ms 时跳过更高并发

# 最低单用户输出 Token 吞吐量（tok/s）
# 当某并发数下输出 Token 吞吐低于此阈值时，跳过更高并发测试。
# 通常在 parallel=1 时触发：若单用户吞吐已低于预期，继续加压意义不大。
# 例: MIN_THROUGHPUT_TOK_S=20 → 低于 20 tok/s 时跳过更高并发
# 留空则不限制
MIN_THROUGHPUT_TOK_S="5"
# MIN_THROUGHPUT_TOK_S=10    # 例: 单用户吞吐 < 10 tok/s 时跳过

# =============================================================================
# ▌ 六、预热配置
# =============================================================================
#
# 首次测试前发送的预热请求数（让 vLLM 完成 CUDA 图预热、KV cache 初始化）。
# 设为 0 则跳过预热（节省时间，但首组结果可能偏高）。
# 仅第一组 (input, output, parallel) 配置执行预热，后续配置跳过。
#
WARMUP_REQUESTS=1

# =============================================================================
# ▌ 七、输出文件配置
# =============================================================================

# 结果保存目录（不存在时自动创建）
RESULT_DIR="results"

# 文件名时间戳（用于区分多次测试的结果文件）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# CSV 结果文件路径（UTF-8-BOM 编码，Excel 可直接打开）
OUTPUT_CSV="${RESULT_DIR}/bench_${TIMESTAMP}.csv"

# XLSX 结果文件路径（需安装 openpyxl；不需要 XLSX 则注释掉下面这行）
OUTPUT_XLSX="${RESULT_DIR}/bench_${TIMESTAMP}.xlsx"

# =============================================================================
# ▌ 八、以下内容通常无需修改
# =============================================================================

# 脚本所在目录（自动识别，支持软链接）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/run_bench_multi.py"

# 检查 Python 脚本是否存在
if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "[ERROR] 找不到基准测试脚本: ${BENCH_SCRIPT}" >&2
    exit 1
fi

# 创建输出目录
mkdir -p "${RESULT_DIR}"

# ─── 构建命令参数数组 ─────────────────────────────────────────────────────────
CMD=(
    python3 "${BENCH_SCRIPT}"
    --model "${MODEL}"
    --backend "${BACKEND}"
    --input-lens ${INPUT_LENS}       # 不加引号，让 shell 展开为多个参数
    --output-lens ${OUTPUT_LENS}
    --parallel-nums ${PARALLEL_NUMS}
    --epochs "${EPOCHS}"
    --sleep-between "${SLEEP_BETWEEN}"
    --output-csv "${OUTPUT_CSV}"
)

# ── 连接方式：BASE_URL 优先，否则 HOST:PORT ─────────────────────────────────
if [[ -n "${BASE_URL}" ]]; then
    CMD+=(--base-url "${BASE_URL}")
    if [[ "${INSECURE}" == "true" ]]; then
        CMD+=(--insecure)
    fi
else
    CMD+=(--host "${HOST}" --port "${PORT}")
fi

# ── 可选参数（非空时追加）────────────────────────────────────────────────────

# XLSX 输出（如果定义了 OUTPUT_XLSX 变量）
if [[ -n "${OUTPUT_XLSX:-}" ]]; then
    CMD+=(--output-xlsx "${OUTPUT_XLSX}")
fi

# 服务端模型别名
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
    CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

# API 鉴权密钥
if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi

# Tokenizer 路径
if [[ -n "${TOKENIZER}" ]]; then
    CMD+=(--tokenizer "${TOKENIZER}")
fi

# 前缀缓存比例（非 0 时传入）
if [[ "${PREFIX_RATIO}" != "0" && "${PREFIX_RATIO}" != "0.0" ]]; then
    CMD+=(--prefix-ratio "${PREFIX_RATIO}")
fi

# 最大 TTFT 约束
if [[ -n "${MAX_TTFT_MS}" ]]; then
    CMD+=(--max-ttft-ms "${MAX_TTFT_MS}")
fi

# 最低单用户吞吐量约束
if [[ -n "${MIN_THROUGHPUT_TOK_S:-}" ]]; then
    CMD+=(--min-throughput-tok-s "${MIN_THROUGHPUT_TOK_S}")
fi

# 笛卡尔积模式
if [[ "${CROSS_PRODUCT}" == "true" ]]; then
    CMD+=(--cross-product)
fi

# 预热请求数
CMD+=(--warmup-requests "${WARMUP_REQUESTS}")

# ─── 打印配置摘要 ─────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           vLLM 多配置批量基准测试                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
if [[ -n "${BASE_URL}" ]]; then
    printf "║  服务地址  : %-48s║\n" "${BASE_URL}"
else
    printf "║  服务地址  : %-48s║\n" "${HOST}:${PORT}"
fi
printf "║  模型      : %-48s║\n" "${MODEL}"
printf "║  后端类型  : %-48s║\n" "${BACKEND}"
printf "║  输入长度  : %-48s║\n" "${INPUT_LENS}"
printf "║  输出长度  : %-48s║\n" "${OUTPUT_LENS}"
printf "║  并发数    : %-48s║\n" "${PARALLEL_NUMS}"
printf "║  测试轮数  : %-48s║\n" "${EPOCHS}"
printf "║  配置间隔  : %-48s║\n" "${SLEEP_BETWEEN}s"
printf "║  CSV 输出  : %-48s║\n" "${OUTPUT_CSV}"
if [[ -n "${OUTPUT_XLSX:-}" ]]; then
    printf "║  XLSX 输出 : %-48s║\n" "${OUTPUT_XLSX}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
DISPLAY_CMD=("${CMD[@]}")
for ((i = 0; i < ${#DISPLAY_CMD[@]}; i++)); do
    if [[ "${DISPLAY_CMD[$i]}" == "--api-key" && $((i + 1)) -lt ${#DISPLAY_CMD[@]} ]]; then
        DISPLAY_CMD[$((i + 1))]="***"
    fi
done
echo "执行命令:"
echo "  ${DISPLAY_CMD[*]}"
echo ""

# ─── 执行 ─────────────────────────────────────────────────────────────────────
exec "${CMD[@]}"
