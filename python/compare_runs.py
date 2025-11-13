#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return the parsed dictionary."""
    with path.open() as f:
        return json.load(f)


def approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """Helper for float comparisons that tolerates tiny rounding noise."""
    return abs(a - b) <= tol


def compare_chromosomes(label: str, lhs: Dict[str, Any], rhs: Dict[str, Any]) -> None:
    """Check every field of two chromosome records."""
    if lhs.get("paths") != rhs.get("paths"):
        raise AssertionError(f"{label} paths differ")
    if len(lhs.get("splitRatios", [])) != len(rhs.get("splitRatios", [])):
        raise AssertionError(f"{label} splitRatios length differ")
    for i, (lv, rv) in enumerate(zip(lhs.get("splitRatios", []), rhs.get("splitRatios", []))):
        if not approx_equal(lv, rv):
            raise AssertionError(f"{label} splitRatio[{i}] mismatch: {lv} vs {rv}")
    for key in ("surplus", "gasUnits", "slippage"):
        lv = lhs.get("fitness", {}).get(key)
        rv = rhs.get("fitness", {}).get(key)
        if not approx_equal(lv, rv):
            raise AssertionError(f"{label} fitness.{key} mismatch: {lv} vs {rv}")
    if lhs.get("rank") != rhs.get("rank"):
        raise AssertionError(f"{label} rank mismatch: {lhs.get('rank')} vs {rhs.get('rank')}")
    cd_left = lhs.get("crowdingDistance")
    cd_right = rhs.get("crowdingDistance")
    if cd_left is None and cd_right is None:
        return
    if cd_left is None or cd_right is None:
        raise AssertionError(f"{label} crowdingDistance mismatch: {cd_left} vs {cd_right}")
    if not approx_equal(cd_left, cd_right):
        raise AssertionError(f"{label} crowdingDistance mismatch: {cd_left} vs {cd_right}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two GA JSON outputs.")
    parser.add_argument("--lhs", required=True, help="Path to the first JSON output.")
    parser.add_argument("--rhs", required=True, help="Path to the second JSON output.")
    args = parser.parse_args()

    lhs_data = load_json(Path(args.lhs))
    rhs_data = load_json(Path(args.rhs))

    for field in ("amount", "generations"):
        if lhs_data.get(field) != rhs_data.get(field):
            raise AssertionError(f"{field} mismatch: {lhs_data.get(field)} vs {rhs_data.get(field)}")

    compare_chromosomes("best", lhs_data["best"], rhs_data["best"])

    lhs_front: List[Dict[str, Any]] = lhs_data.get("paretoFront", [])
    rhs_front: List[Dict[str, Any]] = rhs_data.get("paretoFront", [])
    if len(lhs_front) != len(rhs_front):
        raise AssertionError(f"paretoFront length mismatch: {len(lhs_front)} vs {len(rhs_front)}")
    for idx, (left_chr, right_chr) in enumerate(zip(lhs_front, rhs_front)):
        compare_chromosomes(f"paretoFront[{idx}]", left_chr, right_chr)

    print("GA outputs match for all comparable fields.")


if __name__ == "__main__":
    main()
