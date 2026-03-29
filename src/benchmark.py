"""
benchmark.py — Fair evaluation harness for Transformer efficiency comparison.

Measures:
  - Approximate FLOPs  (via a lightweight analytical estimate)
  - Throughput         (tokens / second)
  - Peak memory        (MB, tracked via torch.cuda.memory_stats or tracemalloc)
  - Latency            (ms per batch)

All measurements use the same random token batches so results are directly
comparable across variants.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

try:
    import wandb as _wandb  # optional — only used when log_wandb=True
except ImportError:
    _wandb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    variant: str
    seq_len: int
    batch_size: int
    flops: float
    tokens_per_sec: float
    peak_memory_mb: float
    latency_ms: float
    num_params: int = 0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.variant}] seq={self.seq_len} batch={self.batch_size} | "
            f"FLOPs={self.flops:.3e}  tps={self.tokens_per_sec:.1f}  "
            f"mem={self.peak_memory_mb:.1f}MB  lat={self.latency_ms:.2f}ms  "
            f"params={self.num_params:,}"
        )


# ---------------------------------------------------------------------------
# FLOPs estimator
# ---------------------------------------------------------------------------


def estimate_flops(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
) -> float:
    """Analytical FLOPs estimate for a Transformer encoder forward pass.

    Per layer:
      - QKV projection:  3 × 2 × B × T × d_model²
      - Attention scores: 2 × B × H × T² × d_head
      - Attention × V:   2 × B × H × T² × d_head
      - Output proj:     2 × B × T × d_model²
      - FFN (two linear): 2 × 2 × B × T × d_model × d_ff
    Embedding lookup is negligible and not included.
    """
    d_head = d_model // num_heads

    flops_per_layer = (
        3 * 2 * batch_size * seq_len * d_model * d_model   # QKV
        + 2 * batch_size * num_heads * seq_len * seq_len * d_head  # QK^T
        + 2 * batch_size * num_heads * seq_len * seq_len * d_head  # AV
        + 2 * batch_size * seq_len * d_model * d_model              # out proj
        + 2 * 2 * batch_size * seq_len * d_model * d_ff             # FFN
    )
    return float(num_layers * flops_per_layer)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def benchmark_model(
    model: nn.Module,
    variant: str,
    seq_len: int,
    batch_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    device: Optional[str] = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the benchmark for one (variant, seq_len, batch_size) combination.

    Parameters
    ----------
    model:
        An already-initialised model in eval mode.
    warmup_steps:
        Iterations discarded before timing starts.
    measure_steps:
        Iterations used for timing / memory measurement.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    # Synthetic integer token IDs
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # --- warmup ---
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids)
    if use_cuda:
        torch.cuda.synchronize()

    # --- peak memory tracking ---
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        gc.collect()
        tracemalloc.start()

    # --- timed measurement ---
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_steps):
            _ = model(input_ids)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if use_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_bytes / (1024 ** 2)
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_bytes / (1024 ** 2)

    latency_ms = (elapsed / measure_steps) * 1000
    tokens_per_sec = (batch_size * seq_len * measure_steps) / elapsed
    flops = estimate_flops(model, seq_len, batch_size, d_model, num_heads, num_layers, d_ff)
    num_params = sum(p.numel() for p in model.parameters())

    return BenchmarkResult(
        variant=variant,
        seq_len=seq_len,
        batch_size=batch_size,
        flops=flops,
        tokens_per_sec=tokens_per_sec,
        peak_memory_mb=peak_memory_mb,
        latency_ms=latency_ms,
        num_params=num_params,
    )


# ---------------------------------------------------------------------------
# Multi-config sweep
# ---------------------------------------------------------------------------


def run_sweep(
    variants: dict,
    seq_lens: List[int],
    batch_size: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    device: Optional[str] = None,
    log_wandb: bool = False,
    wandb_project: str = "transformer-efficiency",
) -> List[BenchmarkResult]:
    """Sweep over all (variant, seq_len) combinations and collect results.

    Parameters
    ----------
    variants:
        Dict mapping variant name → model instance.
    seq_lens:
        List of sequence lengths to evaluate.
    log_wandb:
        Whether to log each result to Weights & Biases.
    """
    if log_wandb:
        if _wandb is None:
            raise ImportError("wandb is not installed. Run `pip install wandb`.")
        _wandb.init(project=wandb_project, config={
            "batch_size": batch_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
        })

    results: List[BenchmarkResult] = []
    for name, model in variants.items():
        for seq_len in seq_lens:
            print(f"  Benchmarking variant={name!r}  seq_len={seq_len} …")
            result = benchmark_model(
                model=model,
                variant=name,
                seq_len=seq_len,
                batch_size=batch_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                device=device,
            )
            print(f"    {result}")
            results.append(result)

            if log_wandb:
                _wandb.log({
                    "variant": result.variant,
                    "seq_len": result.seq_len,
                    "flops": result.flops,
                    "tokens_per_sec": result.tokens_per_sec,
                    "peak_memory_mb": result.peak_memory_mb,
                    "latency_ms": result.latency_ms,
                })

    if log_wandb:
        _wandb.finish()

    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def results_to_csv(results: List[BenchmarkResult], path: str) -> None:
    """Persist benchmark results to a CSV file."""
    import csv
    fieldnames = [
        "variant", "seq_len", "batch_size", "flops",
        "tokens_per_sec", "peak_memory_mb", "latency_ms", "num_params",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"Results saved to {path}")
