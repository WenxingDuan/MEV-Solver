# Alpha-Router (Python Only)

This trimmed repo keeps the Python reference implementations needed to experiment with the hybrid routing concepts from *Hybrid Genetic Algorithm for Optimal User Order Routing in CoW Protocol*. Only four capabilities remain:

1. **GeneticRouterEngine** – the NSGA-II multi-objective solver (`python/genetic_router_engine.py`).
2. **DualDecompositionOptimizer** – the deterministic Bellman–Ford baseline (`python/dual_decomposition_optimizer.py`).
3. **Test/data generation** – benchmark instances under `benchmarks/instances/` plus the runners in `python/run_ga.py`, `python/run_dual.py`, `python/evaluate_engines.py`, and `python/compare_runs.py`.
4. **Visualization** – GA vs. Dual trade-off plotter (`python/plot_ga_vs_dual.py`) that produces `figures/ga_vs_dual.png`.

All legacy non-Python components have been removed so the repository focuses purely on the Python tooling.

## Requirements

- Python ≥ 3.9
- Optional plotting: `python3 -m pip install matplotlib`

## Deterministic Seeding

All Python solvers share the same xorshift32 RNG (`python/seeded_random.py`). Supplying identical `--seed` values guarantees that path generation, crossover, and mutation choices stay deterministic across runs, making experiments reproducible.

## Running the Genetic Router (GA)

```bash
python3 python/run_ga.py \
  --instance benchmarks/instances/sample_instance.json \
  --seed 1337 \
  --population 64 \
  --generations 100
```

- Flags mirror the `GAConfig` fields.
- The CLI injects the deterministic RNG and prints a JSON payload containing the best chromosome and the full Pareto front.
- `fitness.surplus` is the simulated quote-token output (after fees and slippage) summed across all active paths; `gasUnits` and `slippage` capture the corresponding objectives.

## Running the Deterministic Baseline

```bash
python3 python/run_dual.py \
  --instance benchmarks/instances/sample_instance.json \
  --max-path-length 4
```

The script ports `DualDecompositionOptimizer`, replays the best arbitrage-style path through the same AMM simulator used by the GA, and reports the resulting quote-token output (`expectedSurplus`), gas estimate, and slippage. When no profitable path exists, it falls back to an empty solution with only the base transaction cost.

## Comparing Two JSON Outputs

Use the comparison helper to ensure two solver runs (e.g., different seeds or configs) produce identical Pareto fronts:

```bash
python3 python/compare_runs.py --lhs run_a.json --rhs run_b.json
```

The script validates every field of the selected best chromosome and each Pareto-front member with a tight floating-point tolerance.

## Benchmarking GA vs. Dual Decomposition

1. Generate a summary across all benchmark instances:

   ```bash
   python3 python/evaluate_engines.py \
     --instances-dir benchmarks/instances \
     --output benchmarks/results/ga_vs_dual_summary.json
   ```

   The script profiles each instance (fragmentation, market count), runs both solvers, records runtime/objective metrics, and writes per-instance comparisons.

2. Plot optimization effectiveness and computational cost:

   ```bash
   python3 python/plot_ga_vs_dual.py \
     --summary benchmarks/results/ga_vs_dual_summary.json \
     --output figures/ga_vs_dual.png
   ```

   The resulting 2×2 figure contrasts: (i) surplus vs. fragmentation, (ii) slippage vs. fragmentation, (iii) runtime vs. market-count complexity, and (iv) capital efficiency (output per gas) vs. market-count, with marker sizes encoding GA path-count diversity. Each vertical segment connects the deterministic baseline to the GA result for a single instance, highlighting how gains/costs evolve with complexity.

## Understanding the Result Fields

Each GA JSON payload contains:

- `seed`: RNG seed that reproduces the run.
- `config`: The `GAConfig` used by the solver.
- `amount`: Order size (in base-token units) fed into the fitness calculation.
- `generations`: Number of completed NSGA-II generations before hitting limits.
- `best`: The chromosome selected from the rank-1 Pareto front.
- `paretoFront`: Every chromosome with `rank === 1`.

Chromosomes include:

- `paths`: Ordered hops, each with `{fromToken, toToken, poolAddress, protocol}`.
- `splitRatios`: Normalised allocation for each path (sums to 1 when paths exist).
- `fitness`: `{surplus, gasUnits, slippage}` where `surplus` is the simulated quote-token output, `gasUnits` is cumulative gas, and `slippage` aggregates price degradation relative to the local mid-price.
- `rank`: Pareto rank (1 is non-dominated).
- `crowdingDistance`: NSGA-II diversity score within the same rank (boundary individuals show `null` because infinite crowding distance serialises to `null`).

When a chromosome cannot discover a valid path between the base and quote token within the maximum path length, it keeps `paths: []` and yields zero surplus. The deterministic baseline likewise returns a zero-surplus solution when no profitable path exists, ensuring a consistent fallback.

## Repository Layout

- `python/` – GA engine, dual baseline, CLI runners, evaluators, plotting code.
- `benchmarks/` – fixed benchmark instances and result placeholders.
- `figures/` – generated figures such as `ga_vs_dual.png`.

License: MIT
