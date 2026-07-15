# vLLM/sglang 性能基准测试工具

从 `vllm-main/vllm/benchmarks/serve.py` 提取核心指标计算逻辑，**无需安装 vllm 包**即可运行。

## 🚀 快速开始

```bash
# 查看单次基准测试帮助
python run_bench_serve.py --help

# 查看多配置批量测试帮助
python run_bench_multi.py --help

# 运行脚本中预设的多配置批量测试
./run_bench.sh

# 连接已运行的 vLLM 服务做单次测试
python run_bench_serve.py \
    --backend openai-chat \
    --host localhost \
    --port 8000 \
    --model qwen-7 \
    --dataset-name random \
    --num-prompts 50
```

> 📖 详细参数请查看 `python run_bench_serve.py --help` 和 `python run_bench_multi.py --help`。

## 与 vllm bench serve 的对齐程度

| 能力 | vllm bench serve | 本工具 |
|---|---|---|
| TTFT/TPOT/ITL/E2EL 分位数统计 | ✅ | ✅ |
| Goodput (SLO 约束) | ✅ | ✅ |
| Peak 吞吐 / 并发 | ✅ | ✅ |
| OpenAI Completions API | ✅ | ✅ |
| OpenAI Chat Completions API | ✅ | ✅ |
| Poisson 请求调度 | ✅ | ✅ |
| 并发限制 | ✅ | ✅ |
| 预热 | ✅ | ✅ |
| 端点就绪检查 | ✅ | ✅ |
| Random 数据集 | ✅ | ✅ |
| ShareGPT 数据集 | ✅ | ✅（基础） |
| Ramp-up 策略 | ✅ | ❌ |
| Burstiness (Gamma) | ✅ | ❌ |
| Speculative Decoding 指标 | ✅ | ❌ |
| 多模态 | ✅ | ❌ |
| Timeline Plot | ✅ | ❌ |

## 安装

```bash
pip install -r requirements.txt
```

**不需要安装 vllm！** 仅依赖 `aiohttp`、`numpy`、`tqdm`、`transformers`。如需导出 XLSX，另需安装 `openpyxl`。

## 使用方法

### 基本用法（Random 数据集）

```bash
python run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 \
    --port 8000 \
    --model your-model-name \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 10
```

### Chat Completions 端点

```bash
python run_bench_serve.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --host 127.0.0.1 \
    --port 8000 \
    --model your-model-name \
    --dataset-name random \
    --num-prompts 50 \
    --request-rate 5
```

### 带 Goodput SLO 约束

```bash
python run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 --port 8000 \
    --model your-model \
    --num-prompts 200 \
    --request-rate 20 \
    --goodput ttft:500 tpot:100 e2el:5000
```

### 保存结果

```bash
python run_bench_serve.py \
    --backend openai \
    --host 127.0.0.1 --port 8000 \
    --model your-model \
    --num-prompts 100 \
    --request-rate 10 \
    --save-result \
    --result-dir ./results
```

## 指标说明

| 指标 | 含义 | 计算方式 |
|---|---|---|
| **TTFT** | 首 Token 时延 | `首个 token 到达时间 - 请求发出时间` |
| **TPOT** | 平均输出 Token 时延 | `(E2E - TTFT) / (output_tokens - 1)` |
| **ITL** | Token 间延迟 | `token[i+1] 时刻 - token[i] 时刻` |
| **E2EL** | 端到端延迟 | `最后一个 token 时刻 - 请求发出时刻` |
| **Goodput** | 有效吞吐 | 满足所有 SLO 约束的请求数 / 总时间 |
| **Peak tok/s** | 峰值输出吞吐 | 按秒桶统计的最大每秒输出 token 数 |

## 与 benchmark_tools 的关键差异

1. **TPOT 分母**：本工具用 `output_len - 1`（排除首 token），benchmark_tools 用 `output_len`
2. **ITL**：本工具完整计算，benchmark_tools 缺失
3. **统计维度**：本工具提供 mean/median/std + 完整分位数，benchmark_tools 仅提供均值
4. **吞吐含义**：本工具计算全局系统吞吐，benchmark_tools 计算单请求平均速率
