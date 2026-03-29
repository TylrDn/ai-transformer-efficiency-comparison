"""Tests for src/models.py and src/benchmark.py."""

import pytest
import torch

from src.models import load_model, VanillaTransformer, EfficientTransformer
from src.benchmark import benchmark_model, estimate_flops, run_sweep, results_to_csv


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=["vanilla", "efficient"])
def variant(request):
    return request.param


def test_load_model_returns_correct_type(variant):
    model = load_model(variant, d_model=64, num_heads=4, num_layers=2, d_ff=128,
                       max_seq_len=64, device="cpu")
    if variant == "vanilla":
        assert isinstance(model, VanillaTransformer)
    else:
        assert isinstance(model, EfficientTransformer)


def test_load_model_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        load_model("nonexistent", device="cpu")


def test_forward_pass_shape(variant):
    """Output shape should be (batch, seq_len, d_model)."""
    d_model, seq_len, batch = 64, 16, 2
    model = load_model(variant, d_model=d_model, num_heads=4, num_layers=2,
                       d_ff=128, max_seq_len=64, device="cpu")
    input_ids = torch.randint(0, 100, (batch, seq_len))
    with torch.no_grad():
        out = model(input_ids)
    assert out.shape == (batch, seq_len, d_model)


def test_models_produce_finite_outputs(variant):
    model = load_model(variant, d_model=64, num_heads=4, num_layers=2,
                       d_ff=128, max_seq_len=64, device="cpu")
    input_ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        out = model(input_ids)
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"


def test_model_eval_mode(variant):
    model = load_model(variant, d_model=64, num_heads=4, num_layers=2,
                       d_ff=128, max_seq_len=64, device="cpu")
    assert not model.training, "load_model should return model in eval mode"


# ---------------------------------------------------------------------------
# FLOPs estimator tests
# ---------------------------------------------------------------------------


def test_estimate_flops_positive():
    flops = estimate_flops(
        model=None, seq_len=128, batch_size=4,
        d_model=64, num_heads=4, num_layers=2, d_ff=256,
    )
    assert flops > 0


def test_estimate_flops_scales_with_seq_len():
    base = estimate_flops(model=None, seq_len=64, batch_size=1,
                          d_model=64, num_heads=4, num_layers=1, d_ff=256)
    doubled = estimate_flops(model=None, seq_len=128, batch_size=1,
                             d_model=64, num_heads=4, num_layers=1, d_ff=256)
    # Attention is O(T²), so FLOPs should more than double
    assert doubled > 2 * base


# ---------------------------------------------------------------------------
# Benchmark harness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant_name", ["vanilla", "efficient"])
def test_benchmark_model_returns_result(variant_name, tmp_path):
    model = load_model(variant_name, d_model=64, num_heads=4, num_layers=2,
                       d_ff=128, max_seq_len=64, device="cpu")
    result = benchmark_model(
        model=model,
        variant=variant_name,
        seq_len=16,
        batch_size=2,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        warmup_steps=1,
        measure_steps=2,
        device="cpu",
    )
    assert result.variant == variant_name
    assert result.tokens_per_sec > 0
    assert result.peak_memory_mb >= 0
    assert result.latency_ms > 0
    assert result.flops > 0
    assert result.num_params > 0


def test_run_sweep_returns_all_combinations():
    variants = {
        "vanilla": load_model("vanilla", d_model=64, num_heads=4, num_layers=2,
                              d_ff=128, max_seq_len=64, device="cpu"),
        "efficient": load_model("efficient", d_model=64, num_heads=4, num_layers=2,
                                d_ff=128, max_seq_len=64, device="cpu"),
    }
    seq_lens = [16, 32]
    results = run_sweep(
        variants=variants,
        seq_lens=seq_lens,
        batch_size=2,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        warmup_steps=1,
        measure_steps=2,
        device="cpu",
        log_wandb=False,
    )
    assert len(results) == len(variants) * len(seq_lens)
    variant_names = {r.variant for r in results}
    assert variant_names == {"vanilla", "efficient"}


def test_results_to_csv(tmp_path):
    model = load_model("vanilla", d_model=64, num_heads=4, num_layers=2,
                       d_ff=128, max_seq_len=64, device="cpu")
    result = benchmark_model(
        model=model, variant="vanilla", seq_len=16, batch_size=2,
        d_model=64, num_heads=4, num_layers=2, d_ff=128,
        warmup_steps=1, measure_steps=2, device="cpu",
    )
    csv_path = str(tmp_path / "out.csv")
    results_to_csv([result], csv_path)
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["variant"] == "vanilla"
    assert float(rows[0]["tokens_per_sec"]) > 0
