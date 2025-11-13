#!/usr/bin/env python3
from __future__ import annotations

"""CLI wrapper for the deterministic DualDecompositionOptimizer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.dual_decomposition_optimizer import DualDecompositionOptimizer, DualOptimizerConfig, ArbitragePath
from python.token_utils import order_size_to_units
from python.path_simulator import build_market_map, simulate_path


def load_instance(path: Path) -> Dict[str, Any]:
    """Read an instance from disk."""
    with path.open() as f:
        return json.load(f)


def convert_path(path: ArbitragePath) -> List[Dict[str, Any]]:
    """Turn the dataclass representation into a JSON-friendly list."""
    hops: List[Dict[str, Any]] = []
    for i in range(len(path.tokens) - 1):
        hops.append(
            {
                "fromToken": path.tokens[i],
                "toToken": path.tokens[i + 1],
                "poolAddress": path.markets[i].marketAddress,
                "protocol": path.markets[i].protocol,
            }
        )
    return hops


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dual Decomposition baseline solver.")
    parser.add_argument("--instance", required=True, help="Path to benchmark instance JSON.")
    parser.add_argument("--max-path-length", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profit-threshold", type=float, default=0.001)
    parser.add_argument("--amount", type=float, help="Override order size (in token units).")
    args = parser.parse_args()

    instance = load_instance(Path(args.instance))
    meta = instance.get("meta", {})
    markets = instance.get("markets", [])
    base_token = meta.get("baseToken")
    if args.amount is not None:
        amount = float(args.amount)
    else:
        wei_val = meta.get("orderSizeWei")
        if not wei_val:
            raise ValueError("orderSizeWei missing; provide --amount")
        amount = order_size_to_units(base_token, wei_val)

    optimizer = DualDecompositionOptimizer(
        DualOptimizerConfig(
            maxIterations=args.iterations,
            maxPathLength=args.max_path_length,
            minProfitThreshold=args.profit_threshold,
        )
    )
    paths = optimizer.find_profitable_paths(markets, base_token, amount)
    market_map = build_market_map(markets)
    if paths:
        paths.sort(key=lambda p: p.expectedProfit, reverse=True)
        best = paths[0]
        hops = convert_path(best)
        gas_units = 21000 + len(hops) * 150000
        output_amount, slippage = simulate_path(hops, amount, market_map)
        result = {
            "method": "DETERMINISTIC",
            "expectedSurplus": output_amount,
            "gasUnits": gas_units,
            "paths": [hops],
            "splitRatios": [1.0],
            "pathLength": len(hops),
            "slippage": slippage,
        }
    else:
        result = {
            "method": "DETERMINISTIC",
            "expectedSurplus": 0.0,
            "gasUnits": 21000,
            "paths": [],
            "splitRatios": [],
            "pathLength": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
