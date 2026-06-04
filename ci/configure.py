#!/usr/bin/env python3
"""
Mutate Cargo.toml, pyproject.toml, and rust-toolchain.toml so the build
targets a specific polars version.

Usage:
    python ci/configure.py 1.38.1
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent  # repo root = rtlf/

CARGO_TOML   = ROOT / "Cargo.toml"
PYPROJECT    = ROOT / "pyproject.toml"
TOOLCHAIN    = ROOT / "rust-toolchain.toml"
VERSIONS_CFG = ROOT / "ci" / "versions.toml"


def load_versions() -> dict:
    with open(VERSIONS_CFG, "rb") as f:
        return tomllib.load(f)


def find_entry(versions: dict, polars_version: str) -> dict:
    for entry in versions["versions"]:
        if entry["polars"] == polars_version:
            return entry
    available = [e["polars"] for e in versions["versions"]]
    raise SystemExit(f"polars {polars_version!r} not in versions.toml. Available: {available}")


# Features renamed between crate versions: {from_crate: {old_name: new_name}}
# new_name=None means the feature was removed with no replacement.
_FEATURE_RENAMES: dict[str, dict[str, str | None]] = {
    "0.53": {"new_streaming": "streaming"},
}


def patch_cargo(cargo_path: Path, entry: dict) -> None:
    text = cargo_path.read_text()
    crate_ver = entry["crate"]
    polars_tag = f"py-{entry['polars']}"

    # Update crate version numbers in [dependencies]
    # Anchored to line start so "pyo3-polars" (contains "polars") is not matched.
    # Matches: polars-foo = { version = "0.52", ... }
    text = re.sub(
        r'(?m)^(polars[-\w]* = \{ version = )"[^"]+"',
        lambda m: f'{m.group(1)}"{crate_ver}"',
        text,
    )

    # Apply feature renames accumulated up to and including crate_ver.
    for version, renames in _FEATURE_RENAMES.items():
        if crate_ver >= version:
            for old, new in renames.items():
                if new is None:
                    text = re.sub(rf'"\b{re.escape(old)}\b",?\s*', "", text)
                else:
                    text = text.replace(f'"{old}"', f'"{new}"')

    # Update pyo3 and pyo3-polars version constraints in [dependencies]
    pyo3_ver = entry["pyo3"]
    pyo3_polars_ver = entry["pyo3_polars"]
    text = re.sub(
        r'(?m)^(pyo3 = \{ version = )"[^"]+"',
        f'\\1"{pyo3_ver}"',
        text,
    )
    text = re.sub(
        r'(?m)^(pyo3-polars = \{ version = )"[^"]+"',
        f'\\1"{pyo3_polars_ver}"',
        text,
    )

    # Update git tags in [patch.crates-io]
    # Matches: tag = "py-1.38.1"
    text = re.sub(r'(tag = )"py-[^"]+"', f'\\1"{polars_tag}"', text)

    cargo_path.write_text(text)
    print(f"  Cargo.toml  -> crates {crate_ver}, pyo3 {pyo3_ver}, pyo3-polars {pyo3_polars_ver}, tag {polars_tag}")


def patch_pyproject(pyproject_path: Path, entry: dict) -> None:
    text = pyproject_path.read_text()

    # version = "0.38.1"
    text = re.sub(r'^(version = )"[^"]+"', f'\\1"{entry["rtlf"]}"', text, flags=re.MULTILINE)

    # dependencies = ["polars==1.38.1"]
    text = re.sub(
        r'"polars==[^"]+"',
        f'"polars=={entry["polars"]}"',
        text,
    )

    pyproject_path.write_text(text)
    print(f"  pyproject   -> rtlf {entry['rtlf']}, polars {entry['polars']}")


def patch_toolchain(toolchain_path: Path, entry: dict) -> None:
    toolchain_path.write_text(f'[toolchain]\nchannel = "{entry["nightly"]}"\n')
    print(f"  toolchain   -> {entry['nightly']}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: configure.py <polars-version>")

    polars_version = sys.argv[1]
    versions = load_versions()
    entry = find_entry(versions, polars_version)

    print(f"Configuring for polars {polars_version} (rtlf {entry['rtlf']}):")
    patch_cargo(CARGO_TOML, entry)
    patch_pyproject(PYPROJECT, entry)
    patch_toolchain(TOOLCHAIN, entry)
    print("Done.")


if __name__ == "__main__":
    main()
