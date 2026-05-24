"""
Tests for RealtimeLazyFrame and CompiledRealtimeLazyFrame.

Covers: basic ops, compiled variant, multiple placeholders, joins between a
placeholder and a literal frame, joins between two placeholders, filter/agg,
struct outputs (decision-tree-style), and parquet round-trips.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from rtlf import PyRealtimeLazyFrame


# ── helpers ──────────────────────────────────────────────────────────────────

def rtlf(lf: pl.LazyFrame) -> PyRealtimeLazyFrame:
    return PyRealtimeLazyFrame(lf)


def placeholder(name: str, schema: pl.Schema) -> pl.LazyFrame:
    return PyRealtimeLazyFrame.read_placeholder(name, schema)


# ── basic collect / reuse ─────────────────────────────────────────────────────

def test_collect_simple():
    lf = placeholder("input", pl.Schema({"a": pl.Int64, "b": pl.Float64}))
    rt = rtlf(lf.select(pl.col("a") + 1))

    result = rt.collect({"input": pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})})
    assert result["a"].to_list() == [2, 3, 4]


def test_collect_reuse():
    lf = placeholder("x", pl.Schema({"v": pl.Int64}))
    rt = rtlf(lf.select(pl.col("v") * 2))

    assert rt.collect({"x": pl.DataFrame({"v": [1, 2]})})["v"].to_list() == [2, 4]
    assert rt.collect({"x": pl.DataFrame({"v": [10, 20]})})["v"].to_list() == [20, 40]


def test_compiled_collect_reuse():
    lf = placeholder("x", pl.Schema({"v": pl.Int64}))
    compiled = rtlf(lf.select(pl.col("v") * 3)).compile()

    assert compiled.collect({"x": pl.DataFrame({"v": [1, 2]})})["v"].to_list() == [3, 6]
    assert compiled.collect({"x": pl.DataFrame({"v": [10, 20]})})["v"].to_list() == [30, 60]


def test_compile_matches_rtlf():
    schema = pl.Schema({"a": pl.Int32, "b": pl.Int32})
    lf = placeholder("df", schema)
    expr = pl.col("a") * 2 + pl.col("b")
    rt = rtlf(lf.select(expr))
    compiled = rt.compile()

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, schema=schema)
    r1 = rt.collect({"df": df})
    r2 = compiled.collect({"df": df})
    assert r1.frame_equal(r2)


# ── multiple placeholders ─────────────────────────────────────────────────────

def test_two_placeholders():
    schema = pl.Schema({"v": pl.Int32})
    lf_a = placeholder("a", schema)
    lf_b = placeholder("b", schema)
    rt = rtlf(pl.concat([lf_a, lf_b]))

    result = rt.collect({
        "a": pl.DataFrame({"v": [1, 2]}),
        "b": pl.DataFrame({"v": [3, 4]}),
    })
    assert sorted(result["v"].to_list()) == [1, 2, 3, 4]


def test_two_placeholders_compiled():
    schema = pl.Schema({"v": pl.Int32})
    lf_a = placeholder("a", schema)
    lf_b = placeholder("b", schema)
    compiled = rtlf(pl.concat([lf_a, lf_b])).compile()

    r1 = compiled.collect({"a": pl.DataFrame({"v": [1]}), "b": pl.DataFrame({"v": [2]})})
    r2 = compiled.collect({"a": pl.DataFrame({"v": [9]}), "b": pl.DataFrame({"v": [8]})})
    assert sorted(r1["v"].to_list()) == [1, 2]
    assert sorted(r2["v"].to_list()) == [8, 9]


# ── join: placeholder × literal frame ────────────────────────────────────────

def test_join_placeholder_with_literal():
    """Join a streamed placeholder against a static lookup table embedded in the plan."""
    prices = pl.LazyFrame({"item": ["apple", "banana", "cherry"], "price": [1.0, 0.5, 2.0]})
    orders_schema = pl.Schema({"item": pl.String, "qty": pl.Int32})
    orders_lf = placeholder("orders", orders_schema)

    query = orders_lf.join(prices, on="item", how="left").select(
        pl.col("item"),
        (pl.col("qty") * pl.col("price")).alias("total"),
    )
    rt = rtlf(query)

    df = pl.DataFrame({"item": ["apple", "banana", "apple"], "qty": [2, 3, 1]},
                      schema=orders_schema)
    result = rt.collect({"orders": df})
    assert result["total"].to_list() == pytest.approx([2.0, 1.5, 1.0])


# ── join: placeholder × placeholder ──────────────────────────────────────────

def test_join_two_placeholders():
    users_schema  = pl.Schema({"id": pl.Int32, "name": pl.String})
    orders_schema = pl.Schema({"user_id": pl.Int32, "amount": pl.Float64})

    users_lf  = placeholder("users",  users_schema)
    orders_lf = placeholder("orders", orders_schema)

    query = orders_lf.join(users_lf, left_on="user_id", right_on="id", how="left").select(
        pl.col("name"), pl.col("amount")
    )
    rt = rtlf(query)

    users  = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]},
                          schema=users_schema)
    orders = pl.DataFrame({"user_id": [1, 2, 1], "amount": [10.0, 20.0, 5.0]},
                          schema=orders_schema)

    result = rt.collect({"users": users, "orders": orders})
    totals = dict(zip(result["name"].to_list(), result["amount"].to_list()))
    assert totals["Alice"] in (10.0, 5.0)  # two rows for Alice
    assert totals["Bob"] == 20.0


# ── filter / aggregation ──────────────────────────────────────────────────────

def test_filter_and_agg():
    schema = pl.Schema({"group": pl.String, "val": pl.Int32})
    lf = placeholder("data", schema)
    query = (
        lf.filter(pl.col("val") > 2)
          .group_by("group")
          .agg(pl.col("val").sum().alias("total"))
          .sort("group")
    )
    rt = rtlf(query)
    compiled = rt.compile()

    df = pl.DataFrame(
        {"group": ["a", "a", "b", "b", "b"], "val": [1, 3, 2, 4, 5]},
        schema=schema,
    )
    for result in [rt.collect({"data": df}), compiled.collect({"data": df})]:
        groups = dict(zip(result["group"].to_list(), result["total"].to_list()))
        assert groups["a"] == 3   # only val=3 passes filter
        assert groups["b"] == 9   # val=4 + val=5


# ── struct output (decision-tree style) ──────────────────────────────────────

def test_struct_output():
    schema = pl.Schema({"x": pl.Int32})
    lf = placeholder("input", schema)
    expr = (
        pl.when(pl.col("x") < 5)
          .then(pl.struct(pl.lit(0).alias("bucket"), pl.col("x").alias("val")))
          .otherwise(pl.struct(pl.lit(1).alias("bucket"), pl.col("x").alias("val")))
    )
    rt = rtlf(lf.select(expr.alias("out")))
    compiled = rt.compile()

    df = pl.DataFrame({"x": [3, 7, 1, 9]}, schema=schema)
    for result in [rt.collect({"input": df}), compiled.collect({"input": df})]:
        buckets = result["out"].struct.field("bucket").to_list()
        assert buckets == [0, 1, 0, 1]


# ── parquet round-trip ────────────────────────────────────────────────────────

def test_scan_parquet_mixed_with_placeholder():
    """Real parquet scan joined against a placeholder — exercises the
    non-placeholder-builder code path (the parquet scan should go through
    polars' normal physical planning, our builder only intercepts placeholder
    scans)."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "lookup.parquet"
        pl.DataFrame({"code": [10, 20, 30], "label": ["low", "mid", "high"]}).write_parquet(parquet_path)

        schema = pl.Schema({"code": pl.Int32, "score": pl.Float64})
        events_lf  = placeholder("events", schema)
        lookup_lf  = pl.scan_parquet(parquet_path)

        query = events_lf.join(lookup_lf, on="code", how="left").select("label", "score")

        # RealtimeLazyFrame re-compiles each call so non-placeholder scans
        # go through the normal path — this should work fine.
        rt = rtlf(query)
        df = pl.DataFrame({"code": [10, 30, 20], "score": [0.1, 0.9, 0.5]}, schema=schema)
        result = rt.collect({"events": df}).sort("score")

        assert result["label"].to_list() == ["low", "mid", "high"]


# ── missing placeholder error ─────────────────────────────────────────────────

def test_missing_placeholder_raises():
    schema = pl.Schema({"v": pl.Int32})
    rt = rtlf(placeholder("data", schema).select(pl.col("v")))

    with pytest.raises(Exception, match="data"):
        rt.collect({})


def test_missing_placeholder_raises_compiled():
    schema = pl.Schema({"v": pl.Int32})
    compiled = rtlf(placeholder("data", schema).select(pl.col("v"))).compile()

    with pytest.raises(Exception, match="data"):
        compiled.collect({})
