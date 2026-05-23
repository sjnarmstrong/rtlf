# polars-rt

A Rust extension for Polars that pre-compiles a lazy query plan once, then re-executes it at low latency by injecting new DataFrames at runtime — without re-running the optimizer.

## The problem

Every call to `LazyFrame.collect()` runs the Polars optimizer from scratch. For a fixed expression (e.g. a decision tree or scoring function) applied to a stream of small batches, this optimization cost dominates execution time and grows super-linearly with plan complexity.

## How it works

`RealtimeLazyFrame` separates compilation from execution:

1. **Compile once** — you define your query using a placeholder source instead of real data. When you construct a `RealtimeLazyFrame`, it runs the optimizer once and stores the resulting IR plan.

2. **Execute many times** — each call to `collect()` clones the pre-optimized arena, swaps the placeholder's IR node for a real `DataFrameScan`, and runs the physical plan directly.

The placeholder is a fake Parquet scan with a sentinel path token (`_rtlf::placeholder`). The optimizer sees a normal scan source and optimizes around it. At collect time the scan node is replaced before the physical planner ever runs — no re-optimization, no DSL traversal.

```python
import polars as pl
from rtlf import PyRealtimeLazyFrame

# Define schema and expression once
schema = pl.Schema({"feat_0": pl.Int32, "feat_1": pl.Int32})
expr = pl.when(pl.col("feat_0") < 5).then(pl.lit(1)).otherwise(pl.lit(0))

# Compile: runs optimizer once
placeholder = PyRealtimeLazyFrame.read_placeholder("input", schema)
rtlf = PyRealtimeLazyFrame(placeholder.select(expr))

# Execute: no re-optimization
for batch in stream:
    result = rtlf.collect({"input": batch})
```

## Benchmark

Decision tree expressions with width=5 (each node branches 5 ways), batch size=10 rows.
The speedup grows with plan depth because optimizer overhead scales with plan size while execution cost is roughly constant relative to the baseline.

```
depth  lf ms/iter  rtlf ms/iter   speedup
------------------------------------------
    1       1.034         1.009      1.03x
    2       5.642         4.648      1.21x
    3      23.059        17.777      1.30x
    4      70.304        33.631      2.09x
    5     369.320       104.448      3.54x
    6    2606.266       682.040      3.82x
    7   18080.420      4237.845      4.27x
```

At depth 5+ the optimizer is doing most of the work — `rtlf` eliminates it entirely, giving 3.5–4.3x throughput improvement on the same hardware.

## Building

Requires:
- Rust nightly (`nightly-2026-01-09`) — set via `rustup override set nightly-2026-01-09` in `rtlf/`
- Python ≥ 3.12, `uv`

```bash
cd rtlf
uv sync
uv run maturin develop --release
```

Run the benchmark from the project root:

```bash
source rtlf/.venv/bin/activate
python benchmark.py
```

## Implementation notes

**`src/core.rs`** — pure Rust, no pyo3. `read_placeholder()` constructs a `DslPlan::Scan` with a sentinel path encoding the placeholder name. `new()` optimizes the plan and walks the IR arena to find and record the positions of all placeholder nodes. `collect()` clones the arenas, replaces each placeholder node with a live `IR::DataFrameScan`, and runs the physical plan.

**`src/python/mod.rs`** — thin pyo3 wrapper. Accepts `PyLazyFrame` and `HashMap<String, PyDataFrame>` from Python, releases the GIL during `collect()`.

**`src/error.rs`** — newtype around `PolarsError` to map error variants to the appropriate Python exception types (orphan rule workaround).

The Rust crates are patched from the `py-1.38.1` tag of the polars monorepo to guarantee `DSL_SCHEMA_HASH` alignment with the installed Python polars 1.38.1 package. Crates.io published versions do not exactly match any Python release tag and will produce a hash mismatch at runtime.
