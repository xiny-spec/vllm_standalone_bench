#!/usr/bin/env python3
"""
run_bench_serve.py — 基于本地 vllm_bench/ 源码进行 OpenAI 兼容接口基准测试，
无需安装 vllm 包或 torch。

目录结构（与本脚本同级）:
    vllm_bench/
        serve.py
        lib/
            utils.py
            endpoint_request_func.py
            ready_checker.py

最小依赖:
    pip install aiohttp numpy tqdm
可选（用于精确 token 计数）:
    pip install transformers

用法（与 vllm bench serve 参数完全兼容）:
    python run_bench_serve.py \\
        --backend openai \\
        --host 127.0.0.1 --port 8000 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset-name random \\
        --num-prompts 100 \\
        --random-input-len 512 \\
        --random-output-len 128 \\
        --request-rate 10

    python run_bench_serve.py \\
        --backend openai-chat \\
        --host 127.0.0.1 --port 8000 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset-name sharegpt \\
        --dataset-path /path/to/ShareGPT_V3.json \\
        --num-prompts 200 \\
        --save-result

原理:
    通过 sys.modules 注入轻量 shim，绕过 vllm/__init__.py 中的 torch 依赖，
    再用 importlib 按文件路径直接加载本目录下 vllm_bench/ 中的源码。

目录结构（与本脚本同级）:
    vllm_bench/
        serve.py                  ← vllm-main/vllm/benchmarks/serve.py
        lib/
            utils.py              ← vllm-main/vllm/benchmarks/lib/utils.py
            endpoint_request_func.py
            ready_checker.py
"""

import sys
import os
import types
import gc
import re
import ssl
import argparse
import logging
import json
import random
import uuid
import warnings
import importlib.util as _ilu
from dataclasses import dataclass, field
from typing import Any

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_BENCH_DIR = os.path.join(_THIS_DIR, 'vllm_bench')

if not os.path.isdir(VLLM_BENCH_DIR):
    raise RuntimeError(
        f"vllm_bench/ 目录未找到: {VLLM_BENCH_DIR}\n"
        "请确保以下文件已复制到 vllm_bench/ 下:\n"
        "  vllm_bench/serve.py\n"
        "  vllm_bench/lib/utils.py\n"
        "  vllm_bench/lib/endpoint_request_func.py\n"
        "  vllm_bench/lib/ready_checker.py"
    )


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def _make_pkg(name: str) -> types.ModuleType:
    """创建空 package 并注入 sys.modules（防止 __init__.py 被触发）"""
    m = types.ModuleType(name)
    m.__path__ = []            # 标记为 package
    m.__package__ = name
    sys.modules[name] = m
    return m


def _make_mod(name: str, **attrs) -> types.ModuleType:
    """创建带属性的 module 并注入 sys.modules"""
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _load_file(module_name: str, file_path: str) -> types.ModuleType:
    """按文件路径加载模块，注册到 sys.modules（支持相对导入）"""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    m = _ilu.module_from_spec(spec)
    # 为相对导入设置正确的 __package__
    if '.' in module_name:
        m.__package__ = module_name.rsplit('.', 1)[0]
    sys.modules[module_name] = m
    spec.loader.exec_module(m)
    return m


# ─── Step 1: 注入 sys.modules shims（必须在任何 vllm 导入之前）─────────────────

# 1a. regex → stdlib re
#     endpoint_request_func.py 仅用 re.search(r"(\d+)$", ...)，完全兼容
sys.modules.setdefault('regex', re)

# 1b. vllm 父包（空壳，跳过 __init__.py 的 torch 依赖）
_make_pkg('vllm')
_make_pkg('vllm.benchmarks')
_make_pkg('vllm.benchmarks.lib')
_make_pkg('vllm.benchmarks.datasets')
_make_pkg('vllm.utils')

# 1c. vllm.logger
_make_mod('vllm.logger', init_logger=logging.getLogger)

# 1d. vllm.utils.gc_utils
_make_mod('vllm.utils.gc_utils', freeze_gc_heap=gc.freeze)

# 1e. vllm.utils.network_utils
def _join_host_port(host: str, port: int) -> str:
    h = str(host)
    # IPv6 地址需要加方括号
    if ':' in h and not h.startswith('['):
        return f'[{h}]:{port}'
    return f'{h}:{port}'

_make_mod('vllm.utils.network_utils', join_host_port=_join_host_port)

# 1f. vllm.utils.argparse_utils
_make_mod('vllm.utils.argparse_utils',
          FlexibleArgumentParser=argparse.ArgumentParser)

# 1g. vllm.utils.import_utils — PlaceholderModule
class _PlaceholderModule:
    """对 vllm 内部可选依赖的占位符，访问时抛出 ImportError"""
    def __init__(self, name: str):
        self._name = name

    def placeholder_attr(self, attr: str):
        def _raise(*a, **kw):
            raise ImportError(f"{self._name}.{attr} 不可用（未安装）")
        return _raise

    def __getattr__(self, name: str):
        raise ImportError(f"模块 {self._name} 未安装")

_make_mod('vllm.utils.import_utils', PlaceholderModule=_PlaceholderModule)

# 1h. vllm.utils.mistral
_make_mod('vllm.utils.mistral', is_mistral_tokenizer=lambda _: False)

# 1i. vllm.utils.torch_utils（env_override 会间接引用，给一个兼容 stub）
def _is_torch_equal(*a, **kw): return False
def _is_torch_equal_or_newer(*a, **kw): return False
_make_mod('vllm.utils.torch_utils',
          is_torch_equal=_is_torch_equal,
          is_torch_equal_or_newer=_is_torch_equal_or_newer)

# 1j. vllm.benchmarks.plot（懒加载分支，仅在 --plot-timeline/--plot-dataset-stats 时调用）
_make_mod('vllm.benchmarks.plot',
          generate_timeline_plot=lambda *a, **kw: None,
          generate_dataset_stats_plot=lambda *a, **kw: None)

# ─── Step 2: vllm.tokenizers — 基于 transformers（可选）────────────────────────
try:
    from transformers import AutoTokenizer as _AutoTok

    def _get_tokenizer(name_or_path: str | None,
                       tokenizer_mode: str = 'auto',
                       trust_remote_code: bool = False,
                       **kw):
        if not name_or_path:
            return None
        try:
            return _AutoTok.from_pretrained(
                name_or_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            warnings.warn(
                f"Tokenizer 加载失败（{name_or_path}）: {e}\n"
                "将使用 None（output_token 计数可能不精确）"
            )
            return None

    class _TokenizerLike:
        """TokenizerLike 协议占位符"""

except ImportError:
    def _get_tokenizer(*a, **kw) -> None:       # type: ignore[misc]
        warnings.warn(
            "transformers 未安装，无法加载 tokenizer。\n"
            "建议: pip install transformers\n"
            "或使用 --skip-tokenizer-init 跳过 tokenizer 初始化。"
        )
        return None

    class _TokenizerLike:                        # type: ignore[no-redef]
        pass

_make_mod('vllm.tokenizers',
          get_tokenizer=_get_tokenizer,
          TokenizerLike=_TokenizerLike)

# ─── Step 3: 轻量 datasets shim（仅支持 random / sharegpt）────────────────────

@dataclass
class SampleRequest:
    """
    与 vllm.benchmarks.datasets.SampleRequest 完全接口兼容。
    serve.py 访问的字段: prompt / prompt_len / expected_output_len /
                         multi_modal_data / request_id / timestamp
    """
    prompt: Any                           # str（纯文本场景）
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Any = None          # 纯文本场景始终为 None
    request_id: str | None = None
    timestamp: float = 0.0               # timed_trace 专用


def add_dataset_parser(parser: argparse.ArgumentParser) -> None:
    """
    向 serve.py 的 ArgumentParser 注入数据集相关参数。
    接口与原版 vllm.benchmarks.datasets.add_dataset_parser 兼容。
    """
    g = parser.add_argument_group('dataset')

    # ── 通用 ──────────────────────────────────────────────────────────────────
    g.add_argument(
        '--dataset-name', type=str, default=None,
        help='数据集类型（shim 支持: random / sharegpt；'
             '其他类型请安装完整 vllm 包）',
    )
    g.add_argument(
        '--dataset-path', type=str, default=None,
        help='数据集文件路径（sharegpt/sonnet 时需要）',
    )
    g.add_argument('--num-prompts', type=int, default=1000,
                   help='基准测试请求总数')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--trust-remote-code', action='store_true', default=False,
                   help='加载 tokenizer 时允许执行自定义代码')

    # ── random dataset ────────────────────────────────────────────────────────
    g.add_argument('--random-input-len', type=int, default=512,
                   help='random 数据集输入 token 数')
    g.add_argument('--random-output-len', type=int, default=128,
                   help='random 数据集输出 token 数')
    g.add_argument('--random-range-ratio', type=float, default=1.0,
                   help='长度随机抖动比例（1.0=固定长度）')
    g.add_argument('--random-prefix-len', type=int, default=0,
                   help='共享前缀长度（测试 prefix caching）')

    # ── sharegpt dataset ──────────────────────────────────────────────────────
    g.add_argument('--sharegpt-output-len', type=int, default=None,
                   help='覆盖 sharegpt 数据集的 output len（None 则用数据集原始值）')

    # ── sonnet dataset（参数占位，get_samples 不支持）────────────────────────
    g.add_argument('--sonnet-input-len', type=int, default=550)
    g.add_argument('--sonnet-output-len', type=int, default=150)
    g.add_argument('--sonnet-prefix-len', type=int, default=200)

    # ── 其他 dataset output_len（serve.py main 会尝试赋值这些 attr）──────────
    g.add_argument('--custom-output-len', type=int, default=None)
    g.add_argument('--hf-output-len', type=int, default=None)
    g.add_argument('--hf-split', type=str, default=None)
    g.add_argument('--hf-subset', type=str, default=None)
    g.add_argument('--spec-bench-output-len', type=int, default=None)
    g.add_argument('--prefix-repetition-output-len', type=int, default=None)

    # ── 多模态（参数占位）────────────────────────────────────────────────────
    g.add_argument('--random-mm-image-size-mean', type=int, default=None)
    g.add_argument('--random-mm-image-size-std', type=int, default=0)
    g.add_argument('--random-mm-min-num-seq', type=int, default=1)
    g.add_argument('--random-mm-max-num-seq', type=int, default=1)


def _generate_random_requests(args: argparse.Namespace,
                               tokenizer) -> list[SampleRequest]:
    """
    生成随机 token 序列请求，每个请求内容均不同。

    前缀缓存支持（random_prefix_len > 0）：
      - 所有请求共享同一段前缀文本（固定生成一次，用于测试 prefix caching 命中率）
      - 每个请求的后缀部分独立随机生成，保证请求间差异性
      - 最终 prompt = shared_prefix + unique_random_suffix
      - prompt_len ≈ random_prefix_len + random_input_len（实际值由 tokenizer 决定）

    无前缀缓存（random_prefix_len == 0）：
      - 每个请求完全随机，互不相同
    """
    prefix_len = getattr(args, 'random_prefix_len', 0)
    range_ratio = getattr(args, 'random_range_ratio', 1.0)

    def _rand_len(base: int) -> int:
        if range_ratio <= 1.0:
            return base
        lo = max(1, int(base / range_ratio))
        hi = int(base * range_ratio)
        return random.randint(lo, hi)

    # ── 预先生成一次共享前缀（所有请求复用，模拟真实前缀缓存场景）──────────────
    shared_prefix_text = ''
    if prefix_len > 0:
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            vocab_size = getattr(tokenizer, 'vocab_size', 32000)
            shared_ids = [random.randrange(vocab_size) for _ in range(prefix_len)]
            shared_prefix_text = tokenizer.decode(shared_ids)
        else:
            shared_prefix_text = ' '.join(
                str(random.randint(0, 31999)) for _ in range(prefix_len)
            )

    requests: list[SampleRequest] = []
    for i in range(args.num_prompts):
        in_len = _rand_len(args.random_input_len)
        out_len = _rand_len(args.random_output_len)

        # ── 生成每个请求独有的后缀（保证请求间内容不同）────────────────────────
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            vocab_size = getattr(tokenizer, 'vocab_size', 32000)
            suffix_ids = [random.randrange(vocab_size) for _ in range(in_len)]
            suffix_text = tokenizer.decode(suffix_ids)
            prompt = shared_prefix_text + suffix_text
            actual_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        else:
            # 无 tokenizer：用空格分隔的数字模拟 token ids
            suffix_text = ' '.join(
                str(random.randint(0, 31999)) for _ in range(in_len)
            )
            if shared_prefix_text:
                prompt = shared_prefix_text + ' ' + suffix_text
            else:
                prompt = suffix_text
            actual_len = prefix_len + in_len

        requests.append(SampleRequest(
            prompt=prompt,
            prompt_len=actual_len,
            expected_output_len=out_len,
            request_id=f'bench-{uuid.uuid4().hex[:8]}-{i}',
        ))
    return requests


def _load_sharegpt_requests(args: argparse.Namespace,
                             tokenizer) -> list[SampleRequest]:
    """加载 ShareGPT 格式数据集"""
    if not args.dataset_path:
        raise ValueError(
            'ShareGPT 数据集需要指定 --dataset-path /path/to/ShareGPT_Vx.json'
        )
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 过滤无效对话
    dataset = [d for d in dataset if len(d.get('conversations', [])) >= 2]
    n = min(args.num_prompts, len(dataset))
    if n < args.num_prompts:
        warnings.warn(
            f'ShareGPT 数据集只有 {len(dataset)} 条有效对话，实际使用 {n} 条',
            stacklevel=2,
        )

    sampled = random.sample(dataset, n)
    out_len = (
        getattr(args, 'sharegpt_output_len', None)
        or getattr(args, 'random_output_len', 128)
        or 128
    )

    requests: list[SampleRequest] = []
    for i, item in enumerate(sampled):
        prompt = item['conversations'][0]['value']
        if tokenizer is not None and hasattr(tokenizer, '__call__'):
            toks = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_len = len(toks)
        else:
            # 粗估：中英文混合平均约 3 字符/token
            prompt_len = max(1, len(prompt) // 3)

        requests.append(SampleRequest(
            prompt=prompt,
            prompt_len=prompt_len,
            expected_output_len=out_len,
            request_id=f'bench-{uuid.uuid4().hex[:8]}-{i}',
        ))
    return requests


def get_samples(args: argparse.Namespace, tokenizer) -> list[SampleRequest]:
    """
    数据集入口函数，与原版 vllm.benchmarks.datasets.get_samples 接口兼容。
    当前 shim 支持: random / sharegpt。
    """
    name = getattr(args, 'dataset_name', 'random') or 'random'

    if name in ('random', 'random-mm'):
        return _generate_random_requests(args, tokenizer)
    elif name == 'sharegpt':
        return _load_sharegpt_requests(args, tokenizer)
    else:
        raise NotImplementedError(
            f"dataset '{name}' 不在 shim 支持范围。\n"
            f"shim 支持的数据集: random / sharegpt\n"
            f"如需 sonnet/burstgpt/huggingface 等，请安装完整 vllm 包后使用 "
            f"`vllm bench serve`。"
        )


# 将 datasets shim 注册到 sys.modules（覆盖前面的空壳 package）
for _ds_name in ('vllm.benchmarks.datasets',
                 'vllm.benchmarks.datasets.datasets'):
    _make_mod(_ds_name,
              SampleRequest=SampleRequest,
              add_dataset_parser=add_dataset_parser,
              get_samples=get_samples,
              # 以下仅为 datasets/__init__.py 中列出的导出项兼容性占位
              DEFAULT_NUM_PROMPTS=1000,
              add_random_dataset_base_args=lambda p: None,
              add_random_multimodal_dataset_args=lambda p: None,
              )

# datasets.utils shim（serve.py 不直接用，但 datasets/__init__.py 会引用）
class _RangeRatio:
    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

_make_mod('vllm.benchmarks.datasets.utils', RangeRatio=_RangeRatio)

# ─── Step 4: 直接按文件路径加载本地 vllm_bench/ 中的 benchmark 源码 ────────────

def _bench_path(*parts: str) -> str:
    """拼接 vllm_bench/ 子目录路径"""
    return os.path.join(VLLM_BENCH_DIR, *parts)


# lib/utils.py — 纯 stdlib，无 vllm 依赖，直接加载
_load_file('vllm.benchmarks.lib.utils',
           _bench_path('lib', 'utils.py'))

# lib/endpoint_request_func.py
#   依赖: aiohttp / tqdm / regex(→shim为stdlib re) / 标准库
_load_file('vllm.benchmarks.lib.endpoint_request_func',
           _bench_path('lib', 'endpoint_request_func.py'))

# lib/ready_checker.py
#   依赖: vllm.logger(→shim) / endpoint_request_func(→已加载)
_load_file('vllm.benchmarks.lib.ready_checker',
           _bench_path('lib', 'ready_checker.py'))

# serve.py — 主基准脚本（所有 vllm.* 依赖已在上面注入）
_serve = _load_file('vllm.benchmarks.serve',
                    _bench_path('serve.py'))

# ─── Step 5: CLI 入口 ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'vLLM serve benchmark — 无需安装 vllm/torch 包\n'
            f'（使用本地工程代码: {VLLM_BENCH_DIR}）'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _serve.add_cli_args(parser)
    args = parser.parse_args()
    _serve.main(args)


if __name__ == '__main__':
    main()
