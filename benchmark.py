import random
import sys
import time
from functools import cache
from random import randint as randomint

sys.setrecursionlimit(2000)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from rtlf import PyRealtimeLazyFrame

BATCH_SIZE = 1000
NUM_FEATURES = 10
RUNS_PER_DEPTH = 10
LINEAR_BUDGET_S = 2.0
TREE_BUDGET_S = 0.5
MAX_DEPTH = 1000


def depth_sequence():
    """1,2,...,10, 20,30,...,100, 200,300,...,1000."""
    d = 1
    while d <= MAX_DEPTH:
        yield d
        if d < 10:
            d += 1
        else:
            magnitude = 10 ** (len(str(d)) - 1)
            d += magnitude


def make_leaf():
    return pl.struct(
        pl.lit(randomint(0, 10)).alias("a"),
        pl.lit(randomint(0, 10)).alias("b"),
        pl.lit(randomint(0, 10)).alias("c"),
    )


@cache
def build_linear_expr(max_depth: int, version: int):
    """Chain of pl.when(...).then(inner).otherwise(leaf) — one node per depth."""
    if max_depth == 0:
        return make_leaf()
    root = build_linear_expr(max_depth - 1, version)
    feat = f"feat_{(max_depth - 1) % NUM_FEATURES}"
    return pl.when(pl.col(feat) < max_depth).then(root).otherwise(make_leaf())


def build_dtree_expr(max_depth: int, width: int = 5):
    """Exponential tree from benchmark.py."""
    def _build(depth=0):
        if depth >= max_depth:
            return make_leaf()
        expr = pl
        for wi in range(width):
            expr = expr.when(pl.col(f"feat_{depth}") < wi).then(_build(depth + 1))
        return expr.otherwise(_build(depth + 1))
    return _build()


def make_batch(n_features: int) -> pl.DataFrame:
    return pl.DataFrame(
        {f"feat_{d}": [random.randint(0, 9) for _ in range(BATCH_SIZE)] for d in range(n_features)},
        schema={f"feat_{d}": pl.Int32 for d in range(n_features)},
    )


def time_fn(fn, batch: pl.DataFrame) -> float:
    t0 = time.perf_counter()
    fn(batch)
    return time.perf_counter() - t0


def run_one_depth(expr, schema, n_features: int) -> dict[str, list[float]]:
    placeholder_lf = PyRealtimeLazyFrame.read_placeholder("input", schema)
    rtlf = PyRealtimeLazyFrame(placeholder_lf.select(expr))
    compiled = rtlf.compile()

    warmup = make_batch(n_features)
    warmup.lazy().select(expr).collect()
    rtlf.collect({"input": warmup})
    compiled.collect({"input": warmup})

    times: dict[str, list[float]] = {"lf": [], "rtlf": [], "compiled": []}
    for _ in range(RUNS_PER_DEPTH):
        batch = make_batch(n_features)
        times["lf"].append(time_fn(lambda df: df.lazy().select(expr).collect(), batch))
        times["rtlf"].append(time_fn(lambda df: rtlf.collect({"input": df}), batch))
        times["compiled"].append(time_fn(lambda df: compiled.collect({"input": df}), batch))

    return times


def collect_results(build_expr_fn, depth_to_n_features, budget_s: float, label: str) -> pl.DataFrame:
    records = []
    for depth in depth_sequence():
        n_features = depth_to_n_features(depth)
        schema = pl.Schema({f"feat_{d}": pl.Int32 for d in range(n_features)})

        random.seed(depth)
        expr = build_expr_fn(depth, 0)

        print(f"  {label} depth={depth}", end=" ", flush=True)
        times = run_one_depth(expr, schema, n_features)

        median_lf = float(np.median(times["lf"]))
        print(f"lf_median={median_lf * 1000:.1f}ms")

        for method, ts in times.items():
            for t in ts:
                records.append({"depth": depth, "method": method, "time_ms": t * 1000})

        if median_lf >= budget_s:
            break

    return pl.DataFrame(records)


def detailed_table(df: pl.DataFrame, label: str):
    """Print per-depth verbose output matching benchmark.py format, then a summary table."""
    depths = sorted(df["depth"].unique().to_list())

    print(f"\n{label}  batch_size={BATCH_SIZE}\n")
    print(f"{'depth':>5}  {'lf ms':>9}  {'rtlf ms':>9}  {'crtlf ms':>10}  {'rtlf':>13}  {'crtlf':>14}")
    print("-" * 75)

    for d in depths:
        sub = df.filter(pl.col("depth") == d)
        med = {
            row["method"]: row["median"]
            for row in sub.group_by("method").agg(pl.col("time_ms").median().alias("median")).iter_rows(named=True)
        }
        lf, rtlf, crtlf = med.get("lf", float("nan")), med.get("rtlf", float("nan")), med.get("compiled", float("nan"))
        print(f"\ndepth={d}  ({RUNS_PER_DEPTH} runs)")
        print(f"  {'LazyFrame.collect()':<44} {lf:8.3f} ms/iter")
        print(f"  {'RealtimeLazyFrame.collect()':<44} {rtlf:8.3f} ms/iter")
        print(f"  {'CompiledRealtimeLazyFrame.collect()':<44} {crtlf:8.3f} ms/iter")
        print(f"  {'rtlf speedup':<44} {lf / rtlf:8.2f}x")
        print(f"  {'compiled speedup':<44} {lf / crtlf:8.2f}x")

    print("\n" + "=" * 75)
    print(f"{'depth':>5}  {'lf ms':>9}  {'rtlf ms':>9}  {'crtlf ms':>10}  {'rtlf':>13}  {'crtlf':>14}")
    print("-" * 75)
    for d in depths:
        sub = df.filter(pl.col("depth") == d)
        med = {
            row["method"]: row["median"]
            for row in sub.group_by("method").agg(pl.col("time_ms").median().alias("median")).iter_rows(named=True)
        }
        lf, rtlf, crtlf = med.get("lf", float("nan")), med.get("rtlf", float("nan")), med.get("compiled", float("nan"))
        print(f"{d:>5}  {lf:>9.3f}  {rtlf:>9.3f}  {crtlf:>10.3f}  {lf/rtlf:>12.2f}x  {lf/crtlf:>13.2f}x")


def summary_table(df: pl.DataFrame, label: str):
    """Print mean time and speedup bucketed by depth ranges."""
    buckets = [(1, 9), (10, 99), (100, 999), (1000, 9999)]
    rows = []
    for lo, hi in buckets:
        sub = df.filter((pl.col("depth") >= lo) & (pl.col("depth") <= hi))
        if sub.is_empty():
            continue
        means = (
            sub.group_by("method")
            .agg(pl.col("time_ms").mean().alias("mean_ms"))
            .to_dict(as_series=False)
        )
        mean_by = {m: t for m, t in zip(means["method"], means["mean_ms"])}
        if "lf" not in mean_by:
            continue
        rows.append({
            "depth_range": f"{lo}–{hi}",
            "lf_ms": mean_by.get("lf", float("nan")),
            "rtlf_ms": mean_by.get("rtlf", float("nan")),
            "compiled_ms": mean_by.get("compiled", float("nan")),
            "rtlf_speedup": mean_by["lf"] / mean_by["rtlf"] if "rtlf" in mean_by else float("nan"),
            "compiled_speedup": mean_by["lf"] / mean_by["compiled"] if "compiled" in mean_by else float("nan"),
        })

    if not rows:
        return

    col_w = [12, 10, 10, 12, 14, 16]
    headers = ["depth_range", "lf_ms", "rtlf_ms", "compiled_ms", "rtlf_speedup", "compiled_speedup"]
    print(f"\n{label} summary")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "-" * (sum(col_w) + 2 * len(col_w)))
    for r in rows:
        print(
            f"  {r['depth_range']:<{col_w[0]}}"
            f"  {r['lf_ms']:>{col_w[1]}.2f}"
            f"  {r['rtlf_ms']:>{col_w[2]}.2f}"
            f"  {r['compiled_ms']:>{col_w[3]}.2f}"
            f"  {r['rtlf_speedup']:>{col_w[4]}.2f}x"
            f"  {r['compiled_speedup']:>{col_w[5]}.2f}x"
        )


def plot_results(linear_df: pl.DataFrame, tree_df: pl.DataFrame):
    METHOD_LABELS = {"lf": "LazyFrame", "rtlf": "RTLF", "compiled": "Compiled RTLF"}
    PALETTE = {"lf": "#4c72b0", "rtlf": "#dd8452", "compiled": "#55a868"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, df, title in [
        (axes[0], linear_df, f"Linear chain (pl.when stacked, budget={LINEAR_BUDGET_S}s)"),
        (axes[1], tree_df, f"Decision tree (width=5, budget={TREE_BUDGET_S}s)"),
    ]:
        for method in ["lf", "rtlf", "compiled"]:
            sub = df.filter(pl.col("method") == method).sort("depth")
            stats = (
                sub.group_by("depth")
                .agg([
                    pl.col("time_ms").median().alias("median"),
                    pl.col("time_ms").std().alias("std"),
                ])
                .sort("depth")
            )
            d = stats["depth"].to_list()
            med = stats["median"].to_list()
            std = stats["std"].to_list()
            lo = [max(0, m - s) for m, s in zip(med, std)]
            hi = [m + s for m, s in zip(med, std)]

            ax.plot(d, med, label=METHOD_LABELS[method], color=PALETTE[method], marker="o", markersize=4)
            ax.fill_between(d, lo, hi, alpha=0.2, color=PALETTE[method])

        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Depth (nodes, log scale)")
        ax.set_ylabel("Time per batch (ms)")
        ax.legend()
        sns.despine(ax=ax)

    plt.suptitle(f"RTLF benchmark  batch_size={BATCH_SIZE}  runs={RUNS_PER_DEPTH}", fontsize=11)
    plt.tight_layout()
    out = "benchmark.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    print(f"Linear chain benchmark  (budget={LINEAR_BUDGET_S}s, {RUNS_PER_DEPTH} runs/depth)")
    linear_df = collect_results(
        build_expr_fn=build_linear_expr,
        depth_to_n_features=lambda d: min(d, NUM_FEATURES),
        budget_s=LINEAR_BUDGET_S,
        label="linear",
    )
    linear_df.write_parquet("benchmark_linear.parquet")
    print("Saved benchmark_linear.parquet")

    print(f"\nDecision tree benchmark  width=5  (budget={TREE_BUDGET_S}s, {RUNS_PER_DEPTH} runs/depth)")
    tree_df = collect_results(
        build_expr_fn=lambda d, v: build_dtree_expr(d, width=5),
        depth_to_n_features=lambda d: d,
        budget_s=TREE_BUDGET_S,
        label="tree",
    )
    tree_df.write_parquet("benchmark_tree.parquet")
    print("Saved benchmark_tree.parquet")

    summary_table(linear_df, "Linear chain")
    detailed_table(tree_df, "Decision tree  width=5")

    plot_results(linear_df, tree_df)
