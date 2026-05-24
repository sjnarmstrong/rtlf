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

## Benchmarks

All benchmarks use batch size=1000 rows, 10 timed runs per depth point. Results show mean time and speedup relative to `LazyFrame.collect()`.

### Linear chain

A chain of stacked `pl.when(...).then(...).otherwise(...)` calls — depth is the number of nodes. This is the typical structure of a scoring function or feature pipeline.

```
depth_range    lf_ms     rtlf_ms  compiled_ms  rtlf_speedup  compiled_speedup
-------------------------------------------------------------------------------
1–9             1.24        1.25         0.95          0.99x             1.31x
10–99          12.93        9.57         5.78          1.35x             2.24x
100–999       502.97      216.92        21.52          2.32x            23.37x
1000          1545.78      670.64        38.64          2.30x            40.00x
```

At 100+ nodes, `CompiledRealtimeLazyFrame` is **23–40x faster** than plain `LazyFrame`. The optimizer cost grows super-linearly with plan depth; `rtlf` eliminates it entirely. Even the uncompiled `rtlf` gives 2.3x at scale.

### Decision tree

Width-5 branching tree (each depth level fans out 5 ways), so node count is exponential in depth.

```
depth     lf ms    rtlf ms  compiled ms  rtlf speedup  compiled speedup
------------------------------------------------------------------------
    1      1.10       1.07         1.04          1.03x             1.06x
    2      6.01       5.29         3.65          1.14x             1.65x
    3     26.03      18.84        14.29          1.38x             1.82x
    4     73.72      44.32        19.91          1.66x             3.70x
    5    388.34     116.52        14.52          3.33x            26.75x
    6   2817.79     718.58        49.76          3.92x            56.62x
```

Even at shallow depth the exponential node count makes optimizer overhead dominate. By depth 6, `CompiledRealtimeLazyFrame` is **56x faster** than plain `LazyFrame`.

### Running the benchmarks

```bash
uv sync
uv run maturin develop --release
source .venv/bin/activate
python benchmark.py
# outputs: benchmark.png, benchmark_linear.parquet, benchmark_tree.parquet
```

## Building

Requires:
- Rust nightly (`nightly-2026-01-09`) — set via `rustup override set nightly-2026-01-09` in `rtlf/`
- Python ≥ 3.12, `uv`

```bash
cd rtlf
uv sync
uv run maturin develop --release
```

## Implementation notes

**`src/core.rs`** — pure Rust, no pyo3. `read_placeholder()` constructs a `DslPlan::Scan` with a sentinel path encoding the placeholder name. `new()` optimizes the plan and walks the IR arena to find and record the positions of all placeholder nodes. `collect()` clones the arenas, replaces each placeholder node with a live `IR::DataFrameScan`, and runs the physical plan.

**`src/python/mod.rs`** — thin pyo3 wrapper. Accepts `PyLazyFrame` and `HashMap<String, PyDataFrame>` from Python, releases the GIL during `collect()`.

**`src/error.rs`** — newtype around `PolarsError` to map error variants to the appropriate Python exception types (orphan rule workaround).

The Rust crates are patched from the `py-1.38.1` tag of the polars monorepo to guarantee `DSL_SCHEMA_HASH` alignment with the installed Python polars 1.38.1 package. Crates.io published versions do not exactly match any Python release tag and will produce a hash mismatch at runtime.
