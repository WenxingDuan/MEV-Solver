#!/usr/bin/env python3
from __future__ import annotations

"""CLI entry point for running the GeneticRouterEngine on a benchmark file."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.genetic_router_engine import GAConfig, GeneticRouterEngine, Chromosome
from python.seeded_random import SeededRandom
from python.token_utils import order_size_to_units


def load_instance(path: Path) -> Dict[str, Any]:
    """Load a benchmark JSON fixture."""
    with path.open() as f:
        return json.load(f)


def sanitize(value: Any) -> Any:
    """Strip NaN/Inf values so JSON dumps stay valid."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value


def chromosome_to_dict(chromosome: Chromosome) -> Dict[str, Any]:
    """Serialise dataclasses into plain dicts for JSON output."""
    data = chromosome.to_dict()
    data = {k: v for k, v in data.items() if v is not None}
    return sanitize(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Python Genetic Router Engine.")
    parser.add_argument("--instance", required=True, help="Path to benchmark instance JSON.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic RNG seed.")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-path-length", type=int, default=4)
    parser.add_argument("--time-budget", type=int, default=2000)
    parser.add_argument("--amount", type=float, help="Override order size (in token units).")
    args = parser.parse_args()

    instance = load_instance(Path(args.instance))
    meta = instance.get("meta", {})
    markets = instance.get("markets", [])
    start_token = meta.get("baseToken")
    end_token = meta.get("quoteToken")
    if args.amount is not None:
        amount = float(args.amount)
    else:
        wei = meta.get("orderSizeWei")
        if not wei:
            raise ValueError("orderSizeWei missing; provide --amount")
        amount = order_size_to_units(start_token, wei)

    config = GAConfig(
        populationSize=args.population,
        maxGenerations=args.generations,
        crossoverRate=args.crossover,
        mutationRate=args.mutation,
        eliteCount=5,
        maxPaths=args.max_paths,
        maxPathLength=args.max_path_length,
        timeBudgetMs=args.time_budget,
    )

    rng = SeededRandom(args.seed)
    engine = GeneticRouterEngine(config, rand_fn=rng.random)
    result = engine.optimize(start_token, end_token, amount, markets)

    best = chromosome_to_dict(result["best"])
    pareto = [chromosome_to_dict(chr_) for chr_ in result["paretoFront"]]
    output = {
        "seed": args.seed,
        "config": config.__dict__,
        "amount": amount,
        "best": best,
        "paretoFront": pareto,
        "generations": result["generations"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
