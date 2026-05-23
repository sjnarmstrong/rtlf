from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

_LIB = Path(__file__).parent


def byte_len(expr: IntoExpr) -> pl.Expr:
    """Return the byte length of each string as UInt32."""
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="byte_len",
        args=[expr],
        is_elementwise=True,
    )
