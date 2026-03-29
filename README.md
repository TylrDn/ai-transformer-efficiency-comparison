# ai-transformer-efficiency-comparison

Compare compute efficiency (FLOPs, tokens/sec, peak memory) of two Transformer
variants: **Vanilla** (standard scaled dot-product attention) versus **Efficient**
(PyTorch's built-in memory-efficient / Flash Attention kernel).

---

## Repository structure

```
.
├── notebooks/
│   └── 01_variant_benchmark.ipynb   # End-to-end benchmark & visualisation
├── src/
│   ├── __init__.py
│   ├── models.py                    # Variant A/B model loaders
│   └── benchmark.py                 # Fair evaluation harness
├── experiments/
│   └── results.csv                  # Benchmark results (generated / placeholder)
├── tests/
│   └── test_models_and_benchmark.py # Unit tests
├── requirements.txt
└── .github/workflows/ci.yml        # Lint + test CI
```

---

## Variants

| # | Name | Attention | Memory complexity |
|---|------|-----------|------------------|
| A | **Vanilla** | Explicit QK^T matrix multiply (`torch.matmul`) | O(T²) |
| B | **Efficient** | `F.scaled_dot_product_attention` (Flash / memory-efficient) | O(T) |

Both variants share identical architecture (embedding, positional encoding,
multi-head attention + FFN layers, LayerNorm) so differences are purely due
to the attention implementation.

---

## Quick start

```bash
# 1 · Install dependencies
pip install -r requirements.txt

# 2 · Run the benchmark from Python
python - <<'EOF'
from src.models import load_model
from src.benchmark import run_sweep, results_to_csv

variants = {
    "vanilla":   load_model("vanilla"),
    "efficient": load_model("efficient"),
}
results = run_sweep(variants, seq_lens=[128, 256, 512, 1024])
results_to_csv(results, "experiments/results.csv")
EOF

# 3 · Or open the notebook for the full analysis + Pareto curve
jupyter notebook notebooks/01_variant_benchmark.ipynb
```

---

## Sample Pareto curve

The chart below shows **throughput vs. peak memory** for each variant across
sequence lengths T ∈ {128, 256, 512, 1024}.  Points closer to the
**top-left corner** are Pareto-efficient (more tokens/sec at less memory).

```
Throughput (tokens/sec)
 ▲
 │  Efficient ◆────◆────◆────◆
 │           /
 │          /
 │  Vanilla ●────●────●────●
 │
 └──────────────────────────► Peak Memory (MB)
      128   256   512  1024  (seq len labels)
```

*Run `notebooks/01_variant_benchmark.ipynb` to generate the actual
`experiments/pareto_curve.png` on your hardware.*

---

## Sample results (placeholder)

| variant   | seq_len | FLOPs      | tokens/sec | peak_mem MB | latency ms |
|-----------|---------|------------|------------|-------------|------------|
| vanilla   | 128     | 1.23 × 10⁹ | 3200       | 512         | 25.0       |
| vanilla   | 1024    | 7.90 × 10¹⁰| 220        | 4096        | 363        |
| efficient | 128     | 9.88 × 10⁸ | 4500       | 320         | 17.8       |
| efficient | 1024    | 6.32 × 10¹⁰| 1100       | 960         | 72.7       |

Full data in [`experiments/results.csv`](experiments/results.csv).

---

## Running tests

```bash
pytest tests/ -v
```

---

## Logging to Weights & Biases

Pass `log_wandb=True` to `run_sweep()` to stream metrics to a W&B project:

```python
run_sweep(variants, seq_lens=[128, 512], log_wandb=True, wandb_project="my-project")
```

