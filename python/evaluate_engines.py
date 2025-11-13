#!/usr/bin/env python3
from __future__ import annotations

"""Batch runner that compares the GA and deterministic baseline on all instances."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.genetic_router_engine import GeneticRouterEngine, GAConfig
from python.dual_decomposition_optimizer import DualDecompositionOptimizer, DualOptimizerConfig, ArbitragePath
from python.seeded_random import SeededRandom
from python.token_utils import order_size_to_units
from python.path_simulator import build_market_map, simulate_path


def load_instance(path: Path) -> Dict[str, Any]:
    """Wrapper around json.load for clarity."""
    with path.open() as f:
        return json.load(f)


def profile_instance(markets: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute simple complexity metrics used in the plots."""
    tokens = {token for market in markets for token in market.get("tokens", [])}
    market_count = len(markets)
    token_count = len(tokens)
    denom = token_count * (token_count - 1) / 2 if token_count > 1 else 1
    avg_markets_per_pair = market_count / denom if denom else market_count
    fragmentation = min(1.0, avg_markets_per_pair / 3.0)
    return {
        "marketCount": market_count,
        "tokenCount": token_count,
        "fragmentationScore": fragmentation,
    }


def convert_path(path: ArbitragePath) -> List[Dict[str, Any]]:
    """Serialise ArbitragePath objects into plain hop dictionaries."""
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


def _avg_hops(paths: Sequence[Sequence[Any]]) -> float:
    if not paths:
        return 0.0
    return sum(len(path) for path in paths) / len(paths)


def run_ga(
    markets: Sequence[Dict[str, Any]],
    base_token: str,
    quote_token: str,
    amount: float,
    seed: int,
    config: GAConfig,
) -> Dict[str, Any]:
    """Execute the GA and collect runtime plus fitness metrics."""
    rng = SeededRandom(seed)
    engine = GeneticRouterEngine(config, rand_fn=rng.random)
    start = time.perf_counter()
    result = engine.optimize(base_token, quote_token, amount, markets)
    elapsed = (time.perf_counter() - start) * 1000.0
    best = result["best"]
    gas = best.fitness.gasUnits or 1.0
    return {
        "surplus": best.fitness.surplus,
        "gasUnits": best.fitness.gasUnits,
        "slippage": best.fitness.slippage,
        "netSurplus": best.fitness.surplus - best.fitness.gasUnits * 30 * 1e-9,
        "runtimeMs": elapsed,
        "pathCount": len(best.paths),
        "avgHopCount": _avg_hops(best.paths),
        "outputPerGas": best.fitness.surplus / gas,
        "splitRatios": best.splitRatios,
        "rank": best.rank,
        "generations": result["generations"],
    }


def run_deterministic(
    markets: Sequence[Dict[str, Any]], base_token: str, amount: float, config: DualOptimizerConfig
) -> Dict[str, Any]:
    """Execute the deterministic baseline and evaluate its best path."""
    optimizer = DualDecompositionOptimizer(config)
    start = time.perf_counter()
    paths = optimizer.find_profitable_paths(markets, base_token, amount)
    elapsed = (time.perf_counter() - start) * 1000.0
    market_map = build_market_map(list(markets))
    if paths:
        paths.sort(key=lambda p: p.expectedProfit, reverse=True)
        best = paths[0]
        hops = convert_path(best)
        gas_units = 21000 + len(hops) * 150000
        output, slippage = simulate_path(hops, amount, market_map)
        gas = gas_units or 1.0
        return {
            "surplus": output,
            "gasUnits": gas_units,
            "runtimeMs": elapsed,
            "pathLength": len(hops),
            "paths": [hops],
            "slippage": slippage,
            "outputPerGas": output / gas,
        }
    return {
        "surplus": 0.0,
        "gasUnits": 21000.0,
        "runtimeMs": elapsed,
        "pathLength": 0,
        "paths": [],
    }


def collect_instances(instances_dir: Path) -> List[Path]:
    """Enumerate all benchmark JSON files."""
    files = []
    for path in instances_dir.glob("*.json"):
        if path.name == "README.md":
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GA and Dual baseline across benchmark instances.")
    parser.add_argument("--instances-dir", default="benchmarks/instances")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-path-length", type=int, default=4)
    parser.add_argument("--time-budget", type=int, default=2000)
    parser.add_argument("--output", required=True, help="Path to write JSON summary.")
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir)
    files = collect_instances(instances_dir)
    if not files:
        raise FileNotFoundError(f"No benchmark instances found in {instances_dir}")

    ga_config = GAConfig(
        populationSize=args.population,
        maxGenerations=args.generations,
        crossoverRate=args.crossover,
        mutationRate=args.mutation,
        eliteCount=5,
        maxPaths=args.max_paths,
        maxPathLength=args.max_path_length,
        timeBudgetMs=args.time_budget,
    )

    dual_config = DualOptimizerConfig(
        maxIterations=100,
        maxPathLength=args.max_path_length,
        minProfitThreshold=0.001,
    )

    results: List[Dict[str, Any]] = []
    base_seed = args.seed

    for idx, path in enumerate(files):
        data = load_instance(path)
        meta = data.get("meta", {})
        markets = data.get("markets", [])
        base_token = meta.get("baseToken")
        quote_token = meta.get("quoteToken")
        order_size = order_size_to_units(base_token, meta["orderSizeWei"])
        profile = profile_instance(markets)

        ga_seed = base_seed + idx
        ga_stats = run_ga(markets, base_token, quote_token, order_size, ga_seed, ga_config)
        det_stats = run_deterministic(markets, base_token, order_size, dual_config)

        entry = {
            "instance": meta.get("id") or path.stem,
            "orderSize": order_size,
            "orderSizeClass": meta.get("orderSizeClass"),
            "fragmentationLabel": meta.get("fragmentation"),
            "ammDiversity": meta.get("ammDiversity"),
            "gasRegime": meta.get("gasRegime"),
            **profile,
            "ga": ga_stats,
            "deterministic": det_stats,
            "surplusDelta": ga_stats["surplus"] - det_stats["surplus"],
            "slippageDelta": ga_stats["slippage"] - det_stats.get("slippage", 0.0),
            "outputPerGasDelta": ga_stats["outputPerGas"] - det_stats.get("outputPerGas", 0.0),
            "runtimeRatio": (ga_stats["runtimeMs"] / det_stats["runtimeMs"]) if det_stats["runtimeMs"] > 0 else None,
        }
        results.append(entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote comparison summary for {len(results)} instances to {output_path}")


if __name__ == "__main__":
    main()
